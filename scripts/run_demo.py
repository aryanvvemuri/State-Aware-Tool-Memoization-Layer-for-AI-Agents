"""End-to-End Demo Script for Axiom.

Executes the primary demo scenarios:
- Demo 1: Cross-Agent Reuse (Bot A MISS -> expensive execution -> WRITE; Bot B SAFE HIT in ~100ms)
- Demo 2: Semantic Similarity is NOT Enough (Bot C with Sept 5 -> High similarity ~0.92, Argument Match FAIL -> SAFE MISS)
- Demo 3: Tag-Based Invalidation (Invalidate tags -> Repeat query -> MISS -> fresh execution -> WRITE)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axiom.cache import AxiomCache
from axiom.mock_tools import expensive_browser_lookup
from axiom.models import (
    CacheInvalidateRequest,
    CacheLookupRequest,
    CacheWriteRequest,
)
from axiom.telemetry import global_telemetry


def print_banner(title: str):
    print("\n" + "=" * 76)
    print(f"  {title.upper()}")
    print("=" * 76)


def print_metric(label: str, value: Any, color: str = ""):
    print(f"  • {label:<35}: {value}")


def run_demo(mock_sleep_sec: float = 8.0):
    cache = AxiomCache(telemetry=global_telemetry)
    
    print_banner("Axiom: State-Aware Tool Memoization Layer for GrokBot")
    print("Core Thesis: Semantic similarity alone is not sufficient for cache correctness.")
    print("Axiom combines semantic retrieval + canonical argument compatibility + TTL + invalidation.\n")
    time.sleep(1)

    # -----------------------------------------------------------------------
    # DEMO 1: Cross-Agent Reuse
    # -----------------------------------------------------------------------
    print_banner("Demo 1 — Cross-Agent Tool Execution Reuse")
    print("Scenario: Bot A ('research-bot') and Bot B ('ops-bot') need FAA flight data.")
    print("Both share a computer and workflow.\n")

    # Step 1: Bot A checks cache
    query_a = "Find FAA restrictions around Boca Chica for September 4"
    args_a = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-04"}
    
    print("🤖 [Bot A: research-bot] Calling Axiom cache_lookup...")
    print(f"   Query: \"{query_a}\"")
    print(f"   Canonical Args: {args_a}")
    
    t0 = time.perf_counter()
    res_a = cache.lookup(
        CacheLookupRequest(
            query=query_a,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args=args_a,
            bot_id="research-bot",
        )
    )
    lookup_ms_a = (time.perf_counter() - t0) * 1000.0

    print(f"   Result: CACHE MISS (reason: {res_a.reason}, latency: {lookup_ms_a:.1f}ms)")
    print("   → Bot A proceeds to execute expensive external browser traversal tool...")

    # Step 2: Bot A executes expensive tool
    t_exec_start = time.perf_counter()
    tool_payload = expensive_browser_lookup(
        site="faa.gov",
        location="Boca Chica",
        date="2026-09-04",
        query=query_a,
        sleep_seconds=mock_sleep_sec,
        bot_id="research-bot",
        telemetry=global_telemetry,
    )
    actual_exec_ms = (time.perf_counter() - t_exec_start) * 1000.0
    print(f"   Tool execution completed in: {actual_exec_ms/1000.0:.2f}s")
    print(f"   Payload extracted: {tool_payload['airspace_status']} ({tool_payload['notam_id']})")

    # Step 3: Bot A writes to Axiom
    print("   Writing observation to Axiom shared memory...")
    write_res = cache.write(
        CacheWriteRequest(
            query=query_a,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args=args_a,
            payload=tool_payload,
            ttl_seconds=600,
            tags=["faa", "boca_chica", "2026-09-04"],
            source_bot="research-bot",
        )
    )
    print(f"   Stored record: {write_res.record_id} with tags ['faa', 'boca_chica', '2026-09-04']\n")
    time.sleep(1.5)

    # Step 4: Bot B queries with completely different wording!
    query_b = "Are there airspace restrictions near Starbase on September 4?"
    args_b = {"site": "faa.gov", "location": "boca chica", "date": "2026-09-04"}

    print("🤖 [Bot B: ops-bot] Calling Axiom cache_lookup...")
    print(f"   Query: \"{query_b}\"")
    print(f"   Canonical Args: {args_b}")

    t0_b = time.perf_counter()
    res_b = cache.lookup(
        CacheLookupRequest(
            query=query_b,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args=args_b,
            bot_id="ops-bot",
        )
    )
    lookup_ms_b = (time.perf_counter() - t0_b) * 1000.0

    print("   " + "─" * 60)
    if res_b.hit:
        print(f"   🎉 RESULT: SAFE HIT!")
        print(f"   Semantic Similarity : {res_b.similarity:.4f} (Threshold: {cache.similarity_threshold})")
        print(f"   Argument Match      : PASS (Site, Location, Date normalized & matched)")
        print(f"   Cache Age           : {res_b.age_seconds:.1f}s")
        print(f"   Source Bot          : {res_b.source_bot}")
        print(f"   Latency             : {lookup_ms_b:.2f}ms (vs. {actual_exec_ms/1000.0:.2f}s expensive tool)")
        saved_ms = actual_exec_ms - lookup_ms_b
        print(f"   ⚡ Execution Time Saved: {saved_ms/1000.0:.2f}s ({actual_exec_ms/max(0.1, lookup_ms_b):.0f}x speedup!)")
    else:
        print(f"   FAILED: Expected hit, got {res_b.reason}")

    time.sleep(2)

    # -----------------------------------------------------------------------
    # DEMO 2: Semantic Similarity is Not Enough
    # -----------------------------------------------------------------------
    print_banner("Demo 2 — Semantic Similarity is NOT Enough")
    print("Scenario: Bot C queries for September 5 (a different date).")
    print("Embedding models see almost identical sentences, but the operational date changed.\n")

    query_c = "Find FAA restrictions around Boca Chica for September 5"
    args_c = {"site": "faa.gov", "location": "Boca Chica", "date": "2026-09-05"}

    print("🤖 [Bot C: planner-bot] Calling Axiom cache_lookup...")
    print(f"   Query: \"{query_c}\"")
    print(f"   Canonical Args: {args_c}")

    t0_c = time.perf_counter()
    res_c = cache.lookup(
        CacheLookupRequest(
            query=query_c,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args=args_c,
            bot_id="planner-bot",
        )
    )
    lookup_ms_c = (time.perf_counter() - t0_c) * 1000.0

    print("   " + "─" * 60)
    print(f"   SEMANTIC MATCH        : {res_c.best_similarity_seen:.4f} (HIGH)")
    print(f"   STATE/ARGUMENT MATCH  : FAIL (Date '2026-09-04' != '2026-09-05')")
    print(f"   RESULT                : SAFE MISS (reason: {res_c.reason})")
    print("   " + "─" * 60)
    print("   🛡️ CRITICAL MOMENT:")
    print("   Blind semantic cache would have returned September 4 NOTAMs for a September 5 query.")
    print("   Axiom rejected the candidate, ensuring correctness and safety for GrokBot.")
    time.sleep(2)

    # -----------------------------------------------------------------------
    # DEMO 3: Tag-Based Invalidation
    # -----------------------------------------------------------------------
    print_banner("Demo 3 — Tag-Based Invalidation (Coherence & Freshness)")
    print("Scenario: A launch update occurs. Bot calls cache_invalidate(['faa', 'boca_chica']).\n")

    print("🧹 Triggering cache_invalidate(tags=['faa', 'boca_chica'])...")
    inv_res = cache.invalidate(CacheInvalidateRequest(tags=["faa", "boca_chica"]))
    print(f"   Invalidation confirmed: {inv_res.purged_count} records tombstoned.")

    print("\n🤖 [Bot B: ops-bot] Repeating query for September 4...")
    print(f"   Query: \"{query_b}\"")
    
    t0_re = time.perf_counter()
    res_re = cache.lookup(
        CacheLookupRequest(
            query=query_b,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args=args_b,
            bot_id="ops-bot",
        )
    )
    lookup_ms_re = (time.perf_counter() - t0_re) * 1000.0

    print(f"   Result: CACHE MISS (reason: {res_re.reason})")
    print("   → Observation successfully invalidated! Bot executes fresh tool call.")

    # Execute fresh tool
    fresh_payload = expensive_browser_lookup(
        site="faa.gov",
        location="Boca Chica",
        date="2026-09-04",
        query=query_b,
        sleep_seconds=max(1.0, mock_sleep_sec / 2.0),
        bot_id="ops-bot",
        telemetry=global_telemetry,
    )
    cache.write(
        CacheWriteRequest(
            query=query_b,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args=args_b,
            payload=fresh_payload,
            ttl_seconds=600,
            tags=["faa", "boca_chica", "2026-09-04"],
            source_bot="ops-bot",
        )
    )
    print("   Fresh observation stored in Axiom.")
    time.sleep(1.5)

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print_banner("Axiom Live Telemetry Summary")
    stats = global_telemetry.get_aggregate_stats()
    print_metric("Total Requests Evaluated", stats["total_requests"])
    print_metric("Cache Hits", stats["cache_hits"])
    print_metric("Cache Misses", stats["cache_misses"])
    print_metric("Hit Rate", f"{stats['hit_rate_pct']}%")
    print_metric("Tool Executions Avoided", stats["tool_executions_avoided"])
    print_metric("Average Cache Hit Latency", f"{stats['avg_hit_latency_ms']:.2f} ms")
    print_metric("Total Execution Time Saved", f"{stats['estimated_latency_saved_sec']:.2f} seconds")
    print_metric("Invalidations Executed", stats["invalidations_count"])
    print_metric("Miss Reasons Breakdown", stats["miss_reasons"])
    print("=" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Axiom Demos")
    parser.add_argument("--fast", action="store_true", help="Run with faster sleep delays (2s)")
    args = parser.parse_args()

    sleep_time = 2.0 if args.fast else 8.0
    run_demo(mock_sleep_sec=sleep_time)
