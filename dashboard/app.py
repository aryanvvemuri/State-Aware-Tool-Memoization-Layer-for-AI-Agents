"""Streamlit Live Telemetry Dashboard for Axiom.

Provides real-time visualization of cache hits, safe misses (specifically argument mismatches),
invalidation events, real latency savings, and active memoized records.
"""

from __future__ import annotations
import json
import time
import pandas as pd
import streamlit as st

from axiom.cache import AxiomCache
from axiom.config import DEFAULT_SIMILARITY_THRESHOLD
from axiom.models import (
    CacheInvalidateRequest,
    CacheLookupRequest,
    CacheWriteRequest,
    EventType,
)
from axiom.telemetry import global_telemetry

# Page configuration
st.set_page_config(
    page_title="Axiom | State-Aware Tool Memoization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e212b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3440;
    }
    .badge-hit {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-miss {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-mismatch {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-invalidation {
        background-color: #8b5cf6;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_axiom_cache():
    """Singleton cache reference stored in streamlit session state."""
    if "axiom_cache" not in st.session_state:
        st.session_state.axiom_cache = AxiomCache(telemetry=global_telemetry)
    return st.session_state.axiom_cache


cache = get_axiom_cache()
telemetry = global_telemetry

# Sidebar
st.sidebar.title("⚡ Axiom Control")
st.sidebar.markdown(
    "**State-Aware Tool Memoization Layer for GrokBot**\n\n"
    "> *Semantic similarity alone is not sufficient for cache correctness.*"
)

auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", value=False)
if auto_refresh:
    time.sleep(2)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Cache Parameters")
st.sidebar.write(f"**Similarity Threshold:** `{cache.similarity_threshold}`")
st.sidebar.write(f"**Embedding Model:** `{cache.embeddings.model_name}`")

if st.sidebar.button("🧹 Clear Telemetry & Cache"):
    telemetry.clear()
    cache.storage.clear()
    st.sidebar.success("Cache cleared!")
    st.rerun()


# Header & Thesis
st.title("⚡ Axiom: State-Aware Tool Memoization")
st.caption(
    "Coordination layer for multi-agent tool execution reuse. Prevents unsafe reuse by pairing "
    "semantic candidate retrieval with strict canonical argument compatibility."
)

stats = telemetry.get_aggregate_stats()

# Top KPI Metric Row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Requests", stats["total_requests"])
m2.metric("Cache Hits", stats["cache_hits"], f"{stats['hit_rate_pct']}% Hit Rate")
m3.metric("Executions Avoided", stats["tool_executions_avoided"])
m4.metric("Time Saved", f"{stats['estimated_latency_saved_sec']}s")
m5.metric("Avg Hit Latency", f"{stats['avg_hit_latency_ms']}ms")
m6.metric("Invalidations", stats["invalidations_count"])

st.markdown("---")

# Main Content: Two Columns
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📡 Live Telemetry Event Stream")
    events = telemetry.get_events(limit=25)

    if not events:
        st.info("No telemetry events recorded yet. Run a demo or trigger a lookup below.")
    else:
        for ev in events:
            with st.container():
                c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 2])
                
                # Badge formatting
                if ev.event_type == EventType.CACHE_HIT:
                    badge = f'<span class="badge-hit">CACHE HIT</span>'
                elif ev.event_type == EventType.CACHE_MISS:
                    if ev.reason == "argument_mismatch":
                        badge = f'<span class="badge-mismatch">SAFE MISS: ARG MISMATCH</span>'
                    else:
                        badge = f'<span class="badge-miss">MISS: {ev.reason or "NO MATCH"}</span>'
                elif ev.event_type == EventType.INVALIDATION:
                    badge = f'<span class="badge-invalidation">INVALIDATION</span>'
                elif ev.event_type == EventType.CACHE_WRITE:
                    badge = f'<span style="background-color:#3b82f6;color:white;padding:4px 8px;border-radius:12px;font-size:0.8rem;">CACHE WRITE</span>'
                else:
                    badge = f'<span>{ev.event_type.value}</span>'

                c1.markdown(badge, unsafe_allow_html=True)
                c2.markdown(f"**Bot:** `{ev.bot_id or 'system'}`")
                
                sim_str = f"Sim: `{ev.similarity}`" if ev.similarity is not None else ""
                c3.markdown(f"**Latency:** `{ev.latency_ms}ms` {sim_str}")
                
                ts_str = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
                c4.caption(f"Time: {ts_str} | Tool: `{ev.tool_name or '-'}`")

                if ev.metadata:
                    with st.expander(f"Metadata ({ev.event_id})", expanded=False):
                        st.json(ev.metadata)
                st.divider()

with right_col:
    st.subheader("🎯 Cache Safety Analysis")
    
    # Reason Breakdown
    reasons = stats.get("miss_reasons", {})
    if reasons:
        st.markdown("**Miss Reasons Breakdown:**")
        df_reasons = pd.DataFrame(list(reasons.items()), columns=["Reason", "Count"])
        st.bar_chart(df_reasons.set_index("Reason"))

        if "argument_mismatch" in reasons and reasons["argument_mismatch"] > 0:
            st.success(
                f"🛡️ **Safety Guardrail Active:** Axiom safely caught **{reasons['argument_mismatch']}** "
                f"queries that had high semantic similarity but incompatible arguments (e.g. date mismatch)!"
            )
    else:
        st.write("No misses recorded yet.")

    st.markdown("---")
    st.subheader("📦 Active Memoized Records")
    records = cache.storage.get_all_records()
    if not records:
        st.write("No records stored in cache.")
    else:
        rec_data = []
        for r in records:
            rec_data.append({
                "ID": r.record_id,
                "Domain": r.domain,
                "Query": r.semantic_key[:35] + "..." if len(r.semantic_key) > 35 else r.semantic_key,
                "Date": r.canonical_args.get("date", "-"),
                "Location": r.canonical_args.get("location", "-"),
                "Age (s)": round(r.age_seconds, 1),
                "TTL (s)": r.ttl_seconds,
                "Status": "🪦 TOMBSTONED" if r.is_tombstoned else ("⌛ EXPIRED" if r.is_expired else "✅ ACTIVE"),
                "Bot": r.provenance.get("executing_bot", "-"),
            })
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True)

st.markdown("---")

# Interactive Workbench
st.subheader("🧪 Interactive Cache Playground")
st.caption("Simulate queries from different bots and observe how Axiom enforces state correctness.")

tab1, tab2, tab3 = st.tabs(["1. Cache Lookup", "2. Cache Write", "3. Invalidate Tags"])

with tab1:
    with st.form("lookup_form"):
        l_bot = st.selectbox("Calling Bot", ["research-bot", "ops-bot", "planner-bot", "executive-bot"])
        l_query = st.text_input("Semantic Query", "Find FAA flight restrictions near Boca Chica for September 4")
        col_a, col_b, col_c = st.columns(3)
        l_site = col_a.text_input("Canonical Arg: Site", "faa.gov")
        l_loc = col_b.text_input("Canonical Arg: Location", "Boca Chica")
        l_date = col_c.text_input("Canonical Arg: Date", "2026-09-04")
        lookup_btn = st.form_submit_button("Execute cache_lookup")

    if lookup_btn:
        req = CacheLookupRequest(
            query=l_query,
            domain="faa",
            tool_name="expensive_browser_lookup",
            canonical_args={"site": l_site, "location": l_loc, "date": l_date},
            bot_id=l_bot,
        )
        res = cache.lookup(req)
        if res.hit:
            st.success(f"🎉 **CACHE HIT** (Similarity: {res.similarity}, Age: {res.age_seconds}s, Source: {res.source_bot})")
            st.json(res.payload)
        else:
            st.warning(f"⚠️ **SAFE MISS** (Reason: `{res.reason}`, Best Similarity Seen: `{res.best_similarity_seen}`)")
            if res.reason == "argument_mismatch":
                st.info("💡 Semantic similarity was high, but canonical arguments differed (e.g. dates didn't match). Axiom prevented unsafe reuse!")

with tab2:
    with st.form("write_form"):
        w_bot = st.selectbox("Executing Bot", ["research-bot", "ops-bot"])
        w_query = st.text_input("Query / Semantic Key", "Find FAA flight restrictions near Boca Chica for September 4")
        c1, c2, c3 = st.columns(3)
        w_site = c1.text_input("Site", "faa.gov", key="w_site")
        w_loc = c2.text_input("Location", "Boca Chica", key="w_loc")
        w_date = c3.text_input("Date", "2026-09-04", key="w_date")
        w_ttl = st.number_input("TTL (seconds)", value=600)
        w_tags = st.text_input("Tags (comma separated)", "faa, boca_chica, 2026-09-04")
        w_payload = st.text_area("Result Payload (JSON)", '{"airspace_status": "RESTRICTED", "tfr_active": true, "notam_id": "NOTAM-TX-0042"}')
        write_btn = st.form_submit_button("Execute cache_write")

    if write_btn:
        try:
            payload_data = json.loads(w_payload)
            tags_list = [t.strip() for t in w_tags.split(",") if t.strip()]
            w_res = cache.write(
                CacheWriteRequest(
                    query=w_query,
                    domain="faa",
                    tool_name="expensive_browser_lookup",
                    canonical_args={"site": w_site, "location": w_loc, "date": w_date},
                    payload=payload_data,
                    ttl_seconds=w_ttl,
                    tags=tags_list,
                    source_bot=w_bot,
                )
            )
            st.success(f"Saved observation with Record ID: `{w_res.record_id}`")
        except Exception as e:
            st.error(f"Error parsing JSON payload: {e}")

with tab3:
    with st.form("inv_form"):
        inv_tags_input = st.text_input("Tags to Invalidate (comma separated)", "faa, boca_chica")
        inv_btn = st.form_submit_button("Execute cache_invalidate")

    if inv_btn:
        tags_to_inv = [t.strip() for t in inv_tags_input.split(",") if t.strip()]
        inv_res = cache.invalidate(CacheInvalidateRequest(tags=tags_to_inv))
        st.success(f"Invalidated {inv_res.purged_count} records with tags: {tags_to_inv}")
        st.rerun()
