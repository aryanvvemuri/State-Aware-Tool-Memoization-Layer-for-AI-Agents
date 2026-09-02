"""Configuration and default policies for Axiom."""

import os
from typing import Dict

# Huggingface cache directory inside workspace (for offline/sandboxed operation)
_local_hf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
if os.path.exists(_local_hf) and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _local_hf

# Similarity threshold (can be calibrated via scripts/calibrate_threshold.py)
DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("AXIOM_SIMILARITY_THRESHOLD", "0.82"))

# Embedding model name
DEFAULT_EMBEDDING_MODEL: str = os.getenv("AXIOM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Default TTL in seconds per tool or domain (Section 9)
DEFAULT_TTL_POLICIES: Dict[str, int] = {
    # Domains
    "faa": 600,            # FAA status: 10 minutes
    "weather": 300,        # Weather: 5 minutes
    "docs": 86400,         # Static documentation: 24 hours
    "finance": 60,         # Real-time stock/crypto: 1 minute
    # Tools
    "expensive_browser_lookup": 600,
    "api_query": 1800,
}

DEFAULT_TTL_SECONDS: int = int(os.getenv("AXIOM_DEFAULT_TTL_SECONDS", "3600"))

# Mock execution sleep time for testing/demos
DEFAULT_MOCK_SLEEP_SEC: float = float(os.getenv("AXIOM_MOCK_SLEEP_SEC", "8.0"))

# Server settings
AXIOM_HOST: str = os.getenv("AXIOM_HOST", "0.0.0.0")
AXIOM_PORT: int = int(os.getenv("AXIOM_PORT", "8000"))


def get_ttl_for_request(domain: str, tool_name: str, requested_ttl: int = 0) -> int:
    """Returns the effective TTL considering domain, tool, or requested override."""
    if requested_ttl > 0:
        return requested_ttl
    if domain in DEFAULT_TTL_POLICIES:
        return DEFAULT_TTL_POLICIES[domain]
    if tool_name in DEFAULT_TTL_POLICIES:
        return DEFAULT_TTL_POLICIES[tool_name]
    return DEFAULT_TTL_SECONDS
