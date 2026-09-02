"""Tests for recursive argument compatibility checking."""

import pytest
from axiom.compatibility import check_argument_compatibility


def test_exact_args_match():
    candidate = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}
    query = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}
    compat, reason = check_argument_compatibility(candidate, query)
    assert compat is True
    assert reason is None


def test_normalization_case_and_whitespace():
    candidate = {"site": "https://faa.gov/", "location": "Boca Chica  ", "date": "2026-09-04"}
    query = {"site": "faa.gov", "location": "boca chica", "date": "2026-09-04"}
    compat, reason = check_argument_compatibility(candidate, query)
    assert compat is True
    assert reason is None


def test_date_mismatch_must_fail():
    """Core assertion: Sept 4 vs Sept 5 must NEVER be treated as compatible."""
    candidate = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}
    query = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-05"}
    compat, reason = check_argument_compatibility(candidate, query)
    assert compat is False
    assert "date" in reason.lower() or "mismatch" in reason.lower()


def test_location_mismatch():
    candidate = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}
    query = {"site": "faa.gov", "location": "Cape Canaveral", "date": "2026-09-04"}
    compat, reason = check_argument_compatibility(candidate, query)
    assert compat is False
    assert "location" in reason.lower()


def test_missing_and_extra_keys():
    candidate = {"site": "faa.gov", "location": "Boca Chica"}
    query = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}
    compat, reason = check_argument_compatibility(candidate, query)
    assert compat is False
    assert "missing" in reason.lower()

    compat_rev, reason_rev = check_argument_compatibility(query, candidate)
    assert compat_rev is False
    assert "extra" in reason_rev.lower() or "missing" in reason_rev.lower()


def test_nested_dict_compatibility():
    candidate = {
        "target": "server-1",
        "filters": {"status": "ACTIVE", "tier": 2},
    }
    query = {
        "target": "server-1",
        "filters": {"status": "active", "tier": 2},
    }
    compat, _ = check_argument_compatibility(candidate, query)
    assert compat is True

    # Nested mismatch
    query_diff = {
        "target": "server-1",
        "filters": {"status": "active", "tier": 3},
    }
    compat_diff, reason = check_argument_compatibility(candidate, query_diff)
    assert compat_diff is False


def test_flexible_keys():
    candidate = {"site": "faa.gov", "session_token": "tok_abc"}
    query = {"site": "faa.gov", "session_token": "tok_xyz"}
    # Without flexible keys, it fails
    compat, _ = check_argument_compatibility(candidate, query)
    assert compat is False

    # With session_token marked flexible, it succeeds
    compat_flex, _ = check_argument_compatibility(candidate, query, flexible_keys={"session_token"})
    assert compat_flex is True
