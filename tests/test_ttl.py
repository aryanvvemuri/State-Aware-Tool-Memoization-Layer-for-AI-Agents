"""Tests for TTL freshness and expiration behavior."""

import time
import pytest
from axiom.cache import AxiomCache
from axiom.config import get_ttl_for_request
from axiom.models import CacheLookupRequest, CacheRecord, CacheWriteRequest
from axiom.storage import InMemoryStorage


def test_ttl_policy_defaults():
    # FAA domain policy: 600s
    assert get_ttl_for_request(domain="faa", tool_name="other") == 600
    # Weather domain policy: 300s
    assert get_ttl_for_request(domain="weather", tool_name="other") == 300
    # Docs domain policy: 86400s
    assert get_ttl_for_request(domain="docs", tool_name="other") == 86400
    # Custom override
    assert get_ttl_for_request(domain="faa", tool_name="other", requested_ttl=120) == 120


def test_record_expiration():
    storage = InMemoryStorage()
    cache = AxiomCache(storage=storage)

    # Write a record with 1 second TTL
    write_res = cache.write(
        CacheWriteRequest(
            query="current weather in Austin",
            domain="weather",
            tool_name="weather_api",
            canonical_args={"city": "Austin"},
            payload={"temp": 75},
            ttl_seconds=1,
            tags=["weather", "austin"],
            source_bot="test-bot",
        )
    )
    assert write_res.record_id is not None

    # Immediate lookup -> Should HIT
    lookup_req = CacheLookupRequest(
        query="current weather in Austin",
        domain="weather",
        tool_name="weather_api",
        canonical_args={"city": "Austin"},
    )
    res_immediate = cache.lookup(lookup_req)
    assert res_immediate.hit is True
    assert res_immediate.payload == {"temp": 75}

    # Simulate expiration by modifying created_at
    record = storage.get_record(write_res.record_id)
    assert record is not None
    record.created_at = time.time() - 10.0  # 10 seconds ago with 1s TTL

    # Lookup after expiration -> Should MISS due to expired
    res_expired = cache.lookup(lookup_req)
    assert res_expired.hit is False
    assert res_expired.reason == "expired"
