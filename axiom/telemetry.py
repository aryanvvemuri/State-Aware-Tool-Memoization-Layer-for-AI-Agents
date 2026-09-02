"""Telemetry and metrics engine for Axiom.

Tracks all cache operations with high-resolution timestamps, computing
real latency savings and maintaining a live event stream.
"""

from __future__ import annotations
import threading
import time
from typing import Any, Dict, List, Optional
from axiom.models import EventType, TelemetryEvent


class TelemetryLogger:
    """Thread-safe telemetry event logger and metrics aggregator."""

    def __init__(self, max_events: int = 1000):
        self._lock = threading.RLock()
        self._max_events = max_events
        self._events: List[TelemetryEvent] = []
        # Baseline execution latencies observed per tool (for computing real time saved)
        self._baseline_tool_latencies_ms: Dict[str, float] = {
            "expensive_browser_lookup": 8500.0,
            "default": 8000.0,
        }

    def record_event(
        self,
        event_type: EventType,
        bot_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        latency_ms: float = 0.0,
        similarity: Optional[float] = None,
        record_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """Record an event with real measured parameters."""
        event = TelemetryEvent(
            event_type=event_type,
            bot_id=bot_id,
            tool_name=tool_name,
            latency_ms=round(latency_ms, 2),
            similarity=round(similarity, 4) if similarity is not None else None,
            record_id=record_id,
            reason=reason,
            metadata=metadata or {},
        )

        with self._lock:
            # If this was a real tool execution, record its duration as baseline
            if event_type == EventType.TOOL_EXECUTION and tool_name and latency_ms > 0:
                self._baseline_tool_latencies_ms[tool_name] = latency_ms

            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)

        return event

    def get_events(self, limit: int = 100, event_type: Optional[EventType] = None) -> List[TelemetryEvent]:
        """Get recent telemetry events in reverse chronological order."""
        with self._lock:
            evts = self._events[:]
            if event_type is not None:
                evts = [e for e in evts if e.event_type == event_type]
            return list(reversed(evts[-limit:]))

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Calculate real aggregate metrics from the recorded event stream."""
        with self._lock:
            total_lookups = 0
            hits = 0
            misses = 0
            hit_latencies = []
            reason_counts: Dict[str, int] = {}
            invalidations = 0
            single_flight_waits = 0
            total_time_saved_ms = 0.0

            for ev in self._events:
                if ev.event_type == EventType.CACHE_HIT:
                    total_lookups += 1
                    hits += 1
                    hit_latencies.append(ev.latency_ms)
                    # Baseline latency for this tool minus cache hit latency
                    baseline = self._baseline_tool_latencies_ms.get(
                        ev.tool_name or "", self._baseline_tool_latencies_ms["default"]
                    )
                    saved = max(0.0, baseline - ev.latency_ms)
                    total_time_saved_ms += saved

                elif ev.event_type == EventType.CACHE_MISS:
                    total_lookups += 1
                    misses += 1
                    r = ev.reason or "unknown"
                    reason_counts[r] = reason_counts.get(r, 0) + 1

                elif ev.event_type == EventType.INVALIDATION:
                    invalidations += 1

                elif ev.event_type == EventType.SINGLE_FLIGHT_WAIT:
                    single_flight_waits += 1
                    baseline = self._baseline_tool_latencies_ms.get(
                        ev.tool_name or "", self._baseline_tool_latencies_ms["default"]
                    )
                    total_time_saved_ms += baseline

            hit_rate = (hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
            avg_hit_latency = (sum(hit_latencies) / len(hit_latencies)) if hit_latencies else 0.0
            executions_avoided = hits + single_flight_waits

            return {
                "total_requests": total_lookups,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate_pct": round(hit_rate, 1),
                "tool_executions_avoided": executions_avoided,
                "avg_hit_latency_ms": round(avg_hit_latency, 2),
                "estimated_latency_saved_sec": round(total_time_saved_ms / 1000.0, 2),
                "invalidations_count": invalidations,
                "single_flight_collapsed": single_flight_waits,
                "miss_reasons": reason_counts,
            }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


# Global singleton instance
global_telemetry = TelemetryLogger()
