"""Single-flight request coalescing mechanism for Axiom.

Prevents cache stampedes by collapsing concurrent duplicate tool executions
into a single in-flight execution, sharing the result across all callers.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from axiom.models import EventType
from axiom.telemetry import TelemetryLogger, global_telemetry

logger = logging.getLogger(__name__)


def compute_flight_key(tool_name: str, canonical_args: Dict[str, Any]) -> str:
    """Generate a deterministic hash key for a tool and canonical arguments."""
    sorted_args = json.dumps(canonical_args, sort_keys=True)
    h = hashlib.sha256(f"{tool_name}:{sorted_args}".encode("utf-8")).hexdigest()[:16]
    return f"{tool_name}:{h}"


class SingleFlightGroup:
    """
    Manages in-flight asynchronous executions.
    
    If multiple callers request the exact same canonical operation simultaneously,
    only the first caller executes the underlying tool; all subsequent callers
    wait on the same in-flight task and receive the identical result upon completion.
    """

    def __init__(self, telemetry: Optional[TelemetryLogger] = None):
        self._lock = asyncio.Lock()
        self._in_flight: Dict[str, asyncio.Future] = {}
        self.telemetry = telemetry or global_telemetry

    async def execute(
        self,
        tool_name: str,
        canonical_args: Dict[str, Any],
        worker_fn: Callable[[], Coroutine[Any, Any, Any]],
        bot_id: str = "agent",
    ) -> Tuple[Any, bool]:
        """
        Execute worker_fn or await an existing in-flight execution.
        
        Returns:
            (result, was_primary_executor)
        """
        key = compute_flight_key(tool_name, canonical_args)
        t0 = time.perf_counter()

        async with self._lock:
            if key in self._in_flight:
                # Existing in-flight task found: register as waiting caller
                future = self._in_flight[key]
                is_leader = False
            else:
                # Leader: create future and begin execution
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._in_flight[key] = future
                is_leader = True

        if not is_leader:
            # Wait for leader's execution
            result = await future
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.telemetry.record_event(
                event_type=EventType.SINGLE_FLIGHT_WAIT,
                bot_id=bot_id,
                tool_name=tool_name,
                latency_ms=elapsed_ms,
                metadata={
                    "flight_key": key,
                    "canonical_args": canonical_args,
                    "coalesced": True,
                },
            )
            return result, False

        # Leader executes the worker function
        try:
            result = await worker_fn()
            future.set_result(result)
            return result, True
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                if key in self._in_flight:
                    del self._in_flight[key]
