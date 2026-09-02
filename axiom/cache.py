"""Core state-aware tool memoization engine for Axiom.

Combines semantic candidate retrieval, recursive canonical argument compatibility,
freshness (TTL), and tag-based invalidation to determine safe cache hits.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from axiom.compatibility import check_argument_compatibility
from axiom.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    get_ttl_for_request,
)
from axiom.embeddings import EmbeddingEngine
from axiom.models import (
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    CacheLookupRequest,
    CacheLookupResponse,
    CacheRecord,
    CacheWriteRequest,
    CacheWriteResponse,
    EventType,
    MissReason,
)
from axiom.storage import InMemoryStorage, StorageInterface
from axiom.telemetry import TelemetryLogger, global_telemetry

logger = logging.getLogger(__name__)


class AxiomCache:
    """State-Aware Tool Memoization Layer."""

    def __init__(
        self,
        storage: Optional[StorageInterface] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        telemetry: Optional[TelemetryLogger] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.storage = storage or InMemoryStorage()
        self.embeddings = embedding_engine or EmbeddingEngine()
        self.telemetry = telemetry or global_telemetry
        self.similarity_threshold = similarity_threshold

    def lookup(self, req: CacheLookupRequest) -> CacheLookupResponse:
        """
        Perform a state-aware cache lookup.
        
        Evaluates:
        safe_hit = (
            semantic_similarity >= threshold
            and same_tool
            and arguments_compatible
            and not_expired
            and not_invalidated
        )
        """
        t0 = time.perf_counter()
        try:
            # 1. Fetch domain/tool candidates
            candidates = self.storage.get_candidates(domain=req.domain, tool_name=req.tool_name)

            # Check domain/tool mismatches if no candidates for this pair
            if not candidates:
                all_records = self.storage.get_all_records()
                has_tool = any(r.tool_name == req.tool_name for r in all_records)
                has_domain = any(r.domain == req.domain for r in all_records)
                
                reason = MissReason.NO_MATCH.value
                if has_tool and not has_domain:
                    reason = MissReason.DOMAIN_MISMATCH.value
                elif has_domain and not has_tool:
                    reason = MissReason.TOOL_MISMATCH.value

                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.telemetry.record_event(
                    event_type=EventType.CACHE_MISS,
                    bot_id=req.bot_id,
                    tool_name=req.tool_name,
                    latency_ms=latency_ms,
                    similarity=0.0,
                    reason=reason,
                    metadata={"query": req.query, "canonical_args": req.canonical_args},
                )
                return CacheLookupResponse(
                    hit=False,
                    best_similarity_seen=0.0,
                    reason=reason,
                )

            # 2. Generate embedding for current query
            query_vec = self.embeddings.embed(req.query)

            best_similarity_seen: float = 0.0
            best_safe_candidate: Optional[CacheRecord] = None
            best_safe_similarity: float = 0.0

            # Diagnostic reasons for near-miss candidates
            candidate_miss_reasons: List[Tuple[float, str]] = []

            # 3. Evaluate each candidate
            for cand in candidates:
                # Ensure candidate has embedding
                if not cand.embedding:
                    cand.embedding = self.embeddings.embed(cand.semantic_key)

                sim = self.embeddings.cosine_similarity(query_vec, cand.embedding)
                if sim > best_similarity_seen:
                    best_similarity_seen = sim

                # Explicit safety condition evaluation
                same_tool = (cand.tool_name == req.tool_name)
                same_domain = (cand.domain == req.domain)
                not_invalidated = (not cand.is_tombstoned)
                not_expired = (not cand.is_expired)
                args_compatible, mismatch_detail = check_argument_compatibility(
                    candidate_args=cand.canonical_args,
                    query_args=req.canonical_args,
                )
                semantic_ok = (sim >= self.similarity_threshold)

                safe_hit = (
                    semantic_ok
                    and same_tool
                    and same_domain
                    and args_compatible
                    and not_expired
                    and not_invalidated
                )

                if safe_hit:
                    if sim > best_safe_similarity:
                        best_safe_similarity = sim
                        best_safe_candidate = cand
                else:
                    # Diagnose why this candidate failed (especially if similarity is high)
                    if not not_invalidated:
                        candidate_miss_reasons.append((sim, MissReason.INVALIDATED.value))
                    elif not not_expired:
                        candidate_miss_reasons.append((sim, MissReason.EXPIRED.value))
                    elif not args_compatible:
                        candidate_miss_reasons.append((sim, MissReason.ARGUMENT_MISMATCH.value))
                    elif not semantic_ok:
                        candidate_miss_reasons.append((sim, MissReason.NO_MATCH.value))

            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 4. Safe Hit found
            if best_safe_candidate is not None:
                source_bot = best_safe_candidate.provenance.get("executing_bot", "unknown")
                self.telemetry.record_event(
                    event_type=EventType.CACHE_HIT,
                    bot_id=req.bot_id,
                    tool_name=req.tool_name,
                    latency_ms=latency_ms,
                    similarity=best_safe_similarity,
                    record_id=best_safe_candidate.record_id,
                    metadata={
                        "query": req.query,
                        "source_bot": source_bot,
                        "age_seconds": best_safe_candidate.age_seconds,
                    },
                )
                return CacheLookupResponse(
                    hit=True,
                    payload=best_safe_candidate.result_payload,
                    similarity=round(best_safe_similarity, 4),
                    age_seconds=round(best_safe_candidate.age_seconds, 1),
                    source_bot=source_bot,
                    record_id=best_safe_candidate.record_id,
                )

            # 5. Safe Miss
            # Sort near-miss reasons by highest similarity seen
            candidate_miss_reasons.sort(key=lambda x: x[0], reverse=True)
            chosen_reason = candidate_miss_reasons[0][1] if candidate_miss_reasons else MissReason.NO_MATCH.value

            self.telemetry.record_event(
                event_type=EventType.CACHE_MISS,
                bot_id=req.bot_id,
                tool_name=req.tool_name,
                latency_ms=latency_ms,
                similarity=best_similarity_seen,
                reason=chosen_reason,
                metadata={
                    "query": req.query,
                    "canonical_args": req.canonical_args,
                    "best_similarity_seen": best_similarity_seen,
                },
            )
            return CacheLookupResponse(
                hit=False,
                best_similarity_seen=round(best_similarity_seen, 4),
                reason=chosen_reason,
            )

        except Exception as exc:
            # Fail-open design: never block the agent tool if cache fails
            logger.error("Axiom cache lookup error (failing open): %s", exc, exc_info=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return CacheLookupResponse(
                hit=False,
                best_similarity_seen=0.0,
                reason="cache_error_fail_open",
            )

    def write(self, req: CacheWriteRequest) -> CacheWriteResponse:
        """Write a tool execution observation to Axiom."""
        t0 = time.perf_counter()
        # Calculate effective TTL
        effective_ttl = get_ttl_for_request(
            domain=req.domain,
            tool_name=req.tool_name,
            requested_ttl=req.ttl_seconds,
        )

        # Generate embedding
        embedding = self.embeddings.embed(req.query)

        # Build provenance
        provenance = req.provenance or {
            "executing_bot": req.source_bot,
            "tool_used": req.tool_name,
        }

        record = CacheRecord(
            tool_name=req.tool_name,
            domain=req.domain,
            canonical_args=req.canonical_args,
            semantic_key=req.query,
            result_payload=req.payload,
            created_at=time.time(),
            ttl_seconds=effective_ttl,
            invalidation_tags=req.tags,
            provenance=provenance,
            embedding=embedding,
            is_tombstoned=False,
        )

        record_id = self.storage.save_record(record)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.telemetry.record_event(
            event_type=EventType.CACHE_WRITE,
            bot_id=req.source_bot,
            tool_name=req.tool_name,
            latency_ms=latency_ms,
            record_id=record_id,
            metadata={
                "query": req.query,
                "ttl_seconds": effective_ttl,
                "tags": req.tags,
            },
        )

        return CacheWriteResponse(record_id=record_id)

    def invalidate(self, req: CacheInvalidateRequest) -> CacheInvalidateResponse:
        """Tombstone records matching any of the given tags."""
        t0 = time.perf_counter()
        purged_count = self.storage.tombstone_by_tags(tags=req.tags, domain=req.domain)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.telemetry.record_event(
            event_type=EventType.INVALIDATION,
            latency_ms=latency_ms,
            metadata={
                "tags": req.tags,
                "domain": req.domain,
                "purged_count": purged_count,
            },
        )

        return CacheInvalidateResponse(purged_count=purged_count)
