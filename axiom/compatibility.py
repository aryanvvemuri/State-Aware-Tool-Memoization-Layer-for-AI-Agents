"""Recursive argument compatibility checking for Axiom.

Enforces that semantic similarity alone is insufficient for cache hits.
Canonical tool arguments must be compatible, with strict identity checking
for key state fields (dates, locations, sites, IDs) and normalization
for formatting, whitespace, and case.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Optional, Set, Tuple
from datetime import datetime

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None


# Keys that must match strictly and must never be approximated
EXACT_MATCH_KEYS: Set[str] = {
    "date",
    "location",
    "site",
    "account_id",
    "resource_id",
    "id",
    "day",
    "month",
    "year",
    "target",
}


def _normalize_string(val: str, key_hint: Optional[str] = None) -> str:
    """Normalize string case, whitespace, and obvious protocol/path formatting."""
    s = val.strip().lower()
    s = re.sub(r"\s+", " ", s)

    # URL / domain normalization
    if key_hint in {"site", "url", "domain"} or s.startswith("http://") or s.startswith("https://"):
        s = re.sub(r"^https?://", "", s)
        s = re.sub(r"^www\.", "", s)
        s = s.rstrip("/")

    # Date normalization if dateutil is available
    if key_hint in {"date", "day"} and date_parser is not None:
        try:
            dt = date_parser.parse(s, fuzzy=False)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return s


def _values_are_compatible(
    v1: Any,
    v2: Any,
    key_hint: Optional[str] = None,
    flexible_keys: Optional[Set[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Compare two individual values for compatibility."""
    if flexible_keys and key_hint in flexible_keys:
        return True, None

    # Handle None
    if v1 is None and v2 is None:
        return True, None
    if v1 is None or v2 is None:
        return False, f"Value mismatch for '{key_hint}': one is None ({v1} vs {v2})"

    # Handle booleans
    if isinstance(v1, bool) or isinstance(v2, bool):
        if v1 is v2 or v1 == v2:
            return True, None
        return False, f"Boolean mismatch for '{key_hint}': {v1} != {v2}"

    # Handle numeric
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if abs(float(v1) - float(v2)) < 1e-6:
            return True, None
        return False, f"Numeric mismatch for '{key_hint}': {v1} != {v2}"

    # Handle strings
    if isinstance(v1, str) and isinstance(v2, str):
        n1 = _normalize_string(v1, key_hint)
        n2 = _normalize_string(v2, key_hint)
        if n1 == n2:
            return True, None
        return False, f"String mismatch for '{key_hint}': '{v1}' != '{v2}'"

    # Handle lists
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False, f"List length mismatch for '{key_hint}': {len(v1)} != {len(v2)}"
        for i, (item1, item2) in enumerate(zip(v1, v2)):
            comp, reason = _values_are_compatible(item1, item2, f"{key_hint}[{i}]", flexible_keys)
            if not comp:
                return False, reason
        return True, None

    # Handle dicts recursively
    if isinstance(v1, dict) and isinstance(v2, dict):
        return check_argument_compatibility(v1, v2, flexible_keys=flexible_keys)

    # Fallback equality
    if v1 == v2:
        return True, None
    return False, f"Type or value mismatch for '{key_hint}': {v1} ({type(v1)}) != {v2} ({type(v2)})"


def check_argument_compatibility(
    candidate_args: Dict[str, Any],
    query_args: Dict[str, Any],
    flexible_keys: Optional[Set[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Recursively checks if candidate_args are compatible with query_args.
    
    Returns:
        (is_compatible, mismatch_reason)
    """
    if flexible_keys is None:
        flexible_keys = set()

    # Compare keys
    c_keys = set(candidate_args.keys())
    q_keys = set(query_args.keys())

    # All query keys must be represented in candidate args
    # (A candidate can have extra optional defaults if needed, but required query args must match)
    for k in q_keys:
        if k not in c_keys:
            return False, f"Missing key in candidate: '{k}'"

    for k in c_keys:
        if k not in q_keys:
            return False, f"Extra key in candidate not requested: '{k}'"

    for k in q_keys:
        v_cand = candidate_args[k]
        v_query = query_args[k]
        compatible, reason = _values_are_compatible(v_cand, v_query, key_hint=k, flexible_keys=flexible_keys)
        if not compatible:
            return False, reason

    return True, None
