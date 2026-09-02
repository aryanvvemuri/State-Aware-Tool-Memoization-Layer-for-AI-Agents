"""Tests for SQLiteStorage persistence."""

import os
import pytest
from axiom.models import CacheRecord
from axiom.storage import SQLiteStorage


def test_sqlite_storage_crud_and_invalidation(tmp_path):
    db_file = str(tmp_path / "test_axiom.db")
    storage = SQLiteStorage(db_path=db_file)

    rec = CacheRecord(
        tool_name="expensive_browser_lookup",
        domain="faa",
        canonical_args={"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"},
        semantic_key="FAA flight restrictions near Boca Chica September 4",
        result_payload={"status": "TFR Active"},
        ttl_seconds=600,
        invalidation_tags=["faa", "boca_chica"],
        provenance={"executing_bot": "research-bot"},
        embedding=[0.1, 0.2, 0.3],
    )

    # 1. Save
    rid = storage.save_record(rec)
    assert rid == rec.record_id

    # 2. Get by ID
    loaded = storage.get_record(rid)
    assert loaded is not None
    assert loaded.record_id == rid
    assert loaded.tool_name == "expensive_browser_lookup"
    assert loaded.canonical_args["location"] == "Boca Chica"
    assert loaded.embedding == [0.1, 0.2, 0.3]

    # 3. Get candidates
    cands = storage.get_candidates(domain="faa", tool_name="expensive_browser_lookup")
    assert len(cands) == 1
    assert cands[0].record_id == rid

    # 4. Invalidation
    purged = storage.tombstone_by_tags(["boca_chica"])
    assert purged == 1
    reloaded = storage.get_record(rid)
    assert reloaded.is_tombstoned is True

    # 5. Clear
    storage.clear()
    assert len(storage.get_all_records()) == 0
