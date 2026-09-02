"""Deterministic mock expensive tool for testing and demonstrating Axiom.

Simulates slow external browser navigation and DOM extraction (~8-12 seconds),
returning realistic structured FAA airspace and TFR data.
"""

from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, Optional

from axiom.config import DEFAULT_MOCK_SLEEP_SEC
from axiom.models import EventType
from axiom.telemetry import TelemetryLogger, global_telemetry


def _build_mock_payload(site: str, location: str, date: str, query: str) -> Dict[str, Any]:
    """Generate realistic structured payload based on query parameters."""
    loc_lower = location.lower()
    is_starbase = "boca" in loc_lower or "starbase" in loc_lower or "brownsville" in loc_lower

    if is_starbase:
        notam_id = f"NOTAM-FAA-{date.replace('-', '')}-TX-0042"
        return {
            "site": site,
            "location": location,
            "date": date,
            "airspace_status": "RESTRICTED",
            "tfr_active": True,
            "notam_id": notam_id,
            "altitude_limits": "Surface to Unlimited",
            "active_window_utc": f"{date}T12:00:00Z to {date}T22:00:00Z",
            "reason": "Space flight operations / Starship flight test security perimeter",
            "issuing_authority": "Federal Aviation Administration (FAA) Air Traffic Organization",
            "hazard_radius_nm": 12.5,
            "details": {
                "source": "FAA Airspace Operations & NOTAM Registry",
                "center": "Houston ARTCC (ZHU)",
                "coordinates": "25.997° N, 97.156° W",
                "vfr_traffic_advisory": "All unauthorized aircraft must avoid perimeter",
            },
        }
    else:
        return {
            "site": site,
            "location": location,
            "date": date,
            "airspace_status": "NORMAL",
            "tfr_active": False,
            "notam_id": f"NOTAM-FAA-{date.replace('-', '')}-GEN-001",
            "altitude_limits": "Standard class G airspace",
            "active_window_utc": None,
            "reason": "Routine airspace advisory",
            "issuing_authority": "Federal Aviation Administration",
            "hazard_radius_nm": 0.0,
            "details": {"source": site, "status": "No active temporary flight restrictions found"},
        }


def expensive_browser_lookup(
    site: str,
    location: str,
    date: str,
    query: str,
    sleep_seconds: Optional[float] = None,
    bot_id: str = "research-bot",
    telemetry: Optional[TelemetryLogger] = None,
) -> Dict[str, Any]:
    """
    Synchronous mock expensive browser lookup tool.
    Sleeps for sleep_seconds (default 8.0s) to simulate heavy browser traversal.
    """
    duration = sleep_seconds if sleep_seconds is not None else DEFAULT_MOCK_SLEEP_SEC
    t0 = time.perf_counter()
    time.sleep(duration)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    tel = telemetry or global_telemetry
    tel.record_event(
        event_type=EventType.TOOL_EXECUTION,
        bot_id=bot_id,
        tool_name="expensive_browser_lookup",
        latency_ms=elapsed_ms,
        metadata={"site": site, "location": location, "date": date, "query": query},
    )

    return _build_mock_payload(site, location, date, query)


async def aexpensive_browser_lookup(
    site: str,
    location: str,
    date: str,
    query: str,
    sleep_seconds: Optional[float] = None,
    bot_id: str = "research-bot",
    telemetry: Optional[TelemetryLogger] = None,
) -> Dict[str, Any]:
    """
    Asynchronous mock expensive browser lookup tool.
    """
    duration = sleep_seconds if sleep_seconds is not None else DEFAULT_MOCK_SLEEP_SEC
    t0 = time.perf_counter()
    await asyncio.sleep(duration)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    tel = telemetry or global_telemetry
    tel.record_event(
        event_type=EventType.TOOL_EXECUTION,
        bot_id=bot_id,
        tool_name="expensive_browser_lookup",
        latency_ms=elapsed_ms,
        metadata={"site": site, "location": location, "date": date, "query": query},
    )

    return _build_mock_payload(site, location, date, query)
