"""Integration tests for Axiom core memoization and safety gates."""

import pytest
from axiom.cache import AxiomCache
from axiom.models import CacheLookupRequest, CacheWriteRequest
from axiom.storage import InMemoryStorage


def test_definition_of_done_test_a_exact_same_request():
    """Test A: Exact same request: MISS -> WRITE -> HIT."""
    cache = AxiomCache(storage=InMemoryStorage())

    req = CacheLookupRequest(
        query="FAA flight restrictions near Boca Chica September 4",
        domain="faa",
        tool_name="expensive_browser_lookup",
        canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"},
        bot_id="bot-a",
    )

    # 1. MISS initially
    res_miss = cache.lookup(req)
    assert res_miss.hit is False

    # 2. WRITE
    payload = {"tfr_active": True, "notam_id": "NOTAM-TX-42"}
    write_res = cache.write(
        CacheWriteRequest(
            query=req.query,
            domain=req.domain,
            tool_name=req.tool_name,
            canonical_args=req.canonical_args,
            payload=payload,
            ttl_seconds=600,
            tags=["faa", "boca_chica"],
            source_bot="research-bot",
        )
    )
    assert write_res.record_id.startswith("rec_")

    # 3. HIT
    res_hit = cache.lookup(req)
    assert res_hit.hit is True
    assert res_hit.payload == payload
    assert res_hit.source_bot == "research-bot"
    assert res_hit.similarity >= 0.95


def test_definition_of_done_test_b_semantic_safe_hit():
    """Test B: Semantically equivalent request: MISS -> WRITE -> semantic SAFE HIT."""
    cache = AxiomCache(storage=InMemoryStorage())

    # Write initial record by Bot A
    cache.write(
        CacheWriteRequest(
            query="Find FAA restrictions around Boca Chica for September 4",
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"},
            payload={"status": "TFR Active", "ceiling": "unlimited"},
            ttl_seconds=600,
            tags=["faa", "boca_chica"],
            source_bot="research-bot",
        )
    )

    # Bot B queries with rephrased query but compatible canonical args
    bot_b_req = CacheLookupRequest(
        query="Are there any airspace restrictions near Starbase on September 4?",
        domain="faa",
        tool_name="expensive_browser_lookup",
        canonical_args={"site": "faa.gov", "location": "boca chica", "date": "2026-09-04"},
        bot_id="ops-bot",
    )

    res = cache.lookup(bot_b_req)
    assert res.hit is True
    assert res.payload == {"status": "TFR Active", "ceiling": "unlimited"}
    assert res.source_bot == "research-bot"
    assert res.similarity is not None


def test_definition_of_done_test_c_semantic_similar_different_date_safe_miss():
    """
    Test C: Semantically similar but different date:
    Similarity is high, but argument compatibility is FALSE -> SAFE MISS!
    """
    cache = AxiomCache(storage=InMemoryStorage())

    # Stored record for September 4
    cache.write(
        CacheWriteRequest(
            query="Find FAA restrictions around Boca Chica for September 4",
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"},
            payload={"status": "TFR Active Sept 4"},
            ttl_seconds=600,
            tags=["faa", "boca_chica"],
            source_bot="research-bot",
        )
    )

    # Query for September 5 (semantically almost identical, but date state is different!)
    query_sept_5 = CacheLookupRequest(
        query="Find FAA restrictions around Boca Chica for September 5",
        domain="faa",
        tool_name="expensive_browser_lookup",
        canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-05"},
        bot_id="planner-bot",
    )

    res = cache.lookup(query_sept_5)
    # MUST BE A SAFE MISS
    assert res.hit is False
    assert res.reason == "argument_mismatch"
    assert res.best_similarity_seen is not None
    assert res.best_similarity_seen > 0.75, f"Expected high similarity but got {res.best_similarity_seen}"


def test_fail_open_on_internal_exception():
    """Cache failure must fail open (return safe miss) rather than throw."""
    class BrokenStorage(InMemoryStorage):
        def get_candidates(self, domain: str, tool_name: str):
            raise RuntimeError("Database disk corruption simulation")

    cache = AxiomCache(storage=BrokenStorage())
    req = CacheLookupRequest(
        query="test query",
        domain="test",
        tool_name="test_tool",
        canonical_args={"a": 1},
    )

    res = cache.lookup(req)
    assert res.hit is False
    assert "fail_open" in res.reason or "error" in res.reason
