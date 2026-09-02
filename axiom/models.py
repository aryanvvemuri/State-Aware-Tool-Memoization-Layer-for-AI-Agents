"""Data models and schemas for Axiom."""

from __future__ import annotations
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MissReason(str, Enum):
    NO_MATCH = "no_match"
    EXPIRED = "expired"
    ARGUMENT_MISMATCH = "argument_mismatch"
    INVALIDATED = "invalidated"
    TOOL_MISMATCH = "tool_mismatch"
    DOMAIN_MISMATCH = "domain_mismatch"


class EventType(str, Enum):
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"
    CACHE_WRITE = "CACHE_WRITE"
    INVALIDATION = "INVALIDATION"
    SINGLE_FLIGHT_WAIT = "SINGLE_FLIGHT_WAIT"
    TOOL_EXECUTION = "TOOL_EXECUTION"


class CacheRecord(BaseModel):
    """Represents a memoized tool execution record."""
    record_id: str = Field(default_factory=lambda: f"rec_{uuid.uuid4().hex[:10]}")
    tool_name: str
    domain: str
    canonical_args: Dict[str, Any]
    semantic_key: str
    result_payload: Any
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: int = 3600
    invalidation_tags: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    is_tombstoned: bool = False
    embedding: Optional[List[float]] = None

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)


class CacheLookupRequest(BaseModel):
    query: str
    domain: str
    tool_name: str
    canonical_args: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    bot_id: Optional[str] = None


class CacheLookupResponse(BaseModel):
    hit: bool
    payload: Optional[Any] = None
    similarity: Optional[float] = None
    age_seconds: Optional[float] = None
    source_bot: Optional[str] = None
    record_id: Optional[str] = None
    best_similarity_seen: Optional[float] = None
    reason: Optional[str] = None


class CacheWriteRequest(BaseModel):
    query: str
    domain: str
    tool_name: str
    canonical_args: Dict[str, Any]
    payload: Any
    ttl_seconds: int = 3600
    tags: List[str] = Field(default_factory=list)
    source_bot: str = "agent"
    provenance: Optional[Dict[str, Any]] = None


class CacheWriteResponse(BaseModel):
    record_id: str


class CacheInvalidateRequest(BaseModel):
    tags: List[str]
    domain: Optional[str] = None


class CacheInvalidateResponse(BaseModel):
    purged_count: int


class TelemetryEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    bot_id: Optional[str] = None
    tool_name: Optional[str] = None
    latency_ms: float = 0.0
    similarity: Optional[float] = None
    record_id: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
