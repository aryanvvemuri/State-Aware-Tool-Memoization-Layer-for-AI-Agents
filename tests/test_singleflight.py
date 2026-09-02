"""Tests for SingleFlightGroup request coalescing."""

import asyncio
import pytest
from axiom.models import EventType
from axiom.singleflight import SingleFlightGroup
from axiom.telemetry import TelemetryLogger


@pytest.mark.asyncio
async def test_single_flight_coalescing():
    telemetry = TelemetryLogger()
    sf = SingleFlightGroup(telemetry=telemetry)

    execution_count = 0

    async def expensive_mock_operation():
        nonlocal execution_count
        execution_count += 1
        await asyncio.sleep(0.3)
        return {"data": "faa_restrictions_result", "counter": execution_count}

    tool = "expensive_browser_lookup"
    args = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}

    # Launch 5 callers concurrently
    async def caller(bot_id: str):
        res, is_leader = await sf.execute(
            tool_name=tool,
            canonical_args=args,
            worker_fn=expensive_mock_operation,
            bot_id=bot_id,
        )
        return res, is_leader

    tasks = [caller(f"bot-{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)

    # 1. Underlying operation must be executed EXACTLY ONCE
    assert execution_count == 1

    # 2. Exactly one caller must be the leader, four must be followers
    leaders = [is_leader for _, is_leader in results if is_leader]
    followers = [is_leader for _, is_leader in results if not is_leader]
    assert len(leaders) == 1
    assert len(followers) == 4

    # 3. All 5 callers received the exact same payload
    for payload, _ in results:
        assert payload["data"] == "faa_restrictions_result"
        assert payload["counter"] == 1

    # 4. Telemetry recorded 4 SINGLE_FLIGHT_WAIT events
    wait_events = telemetry.get_events(event_type=EventType.SINGLE_FLIGHT_WAIT)
    assert len(wait_events) == 4
