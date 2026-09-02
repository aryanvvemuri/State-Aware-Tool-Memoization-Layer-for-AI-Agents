"""Tests for tag-based invalidation (tombstoning)."""

import pytest
from axiom.cache import AxiomCache
from axiom.models import (
    CacheInvalidateRequest,
    CacheLookupRequest,
    CacheWriteRequest,
)
from axiom.storage import InMemoryStorage


def test_tag_invalidation():
    storage = InMemoryStorage()
    cache = AxiomCache(storage=storage)

    # 1. Write record 1 (FAA Boca Chica)
    w1 = cache.write(
        CacheWriteRequest(
            query="FAA flight restrictions near Boca Chica September 4",
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"},
            payload={"status": "TFR ACTIVE"},
            tags=["faa", "boca_chica", "2026-09-04"],
            source_bot="research-bot",
        )
    )

    # 2. Write record 2 (FAA Cape Canaveral)
    w2 = cache.write(
        CacheWriteRequest(
            query="FAA flight restrictions near Cape Canaveral September 4",
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args={"site": "faa.gov", "location": "Cape Canaveral", "date": "2026-09-04"},
            payload={"status": "NO RESTRICTIONS"},
            tags=["faa", "cape_canaveral", "2026-09-04"],
            source_bot="research-bot",
        )
    )

    # Both should be hits initially
    req1 = CacheLookupRequest(
        query="FAA flight restrictions near Boca Chica September 4",
        domain="faa",
        tool_name="expensive_browser_lookup",
        canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"},
    )
    req2 = CacheLookupRequest(
        query="FAA flight restrictions near Cape Canaveral September 4",
        domain="faa",
        tool_name="expensive_browser_lookup",
        canonical_args={"site": "faa.gov", "location": "Cape Canaveral", "date": "2026-09-04"},
    )
    assert cache.lookup(req1).hit is True
    assert cache.lookup(req2).hit is True

    # Invalidate only boca_chica
    inv_res = cache.invalidate(CacheInvalidateRequest(tags=["boca_chica"]))
    assert inv_res.purged_count == 1

    # Record 1 should now MISS due to invalidation
    lookup_after_1 = cache.lookup(req1)
    assert lookup_after_1.hit is False
    assert lookup_after_1.reason == "invalidated"

    # Record 2 should still HIT
    lookup_after_2 = cache.lookup(req2)
    assert lookup_after_2.hit is True
    assert lookup_after_2.payload == {"status": "NO RESTRICTIONS"}
