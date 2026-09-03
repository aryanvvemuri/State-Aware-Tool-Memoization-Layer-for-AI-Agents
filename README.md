# Axiom: State-Aware Tool Memoization Layer for GrokBot

[![CI](https://github.com/aryanvvemuri/State-Aware-Tool-Memoization-Layer-for-AI-Agents/actions/workflows/ci.yml/badge.svg)](https://github.com/aryanvvemuri/State-Aware-Tool-Memoization-Layer-for-AI-Agents/actions/workflows/ci.yml)

> **"Semantic similarity alone is not sufficient for cache correctness. Axiom combines semantic matching with canonical tool arguments, freshness, and invalidation metadata to determine whether a previous tool execution can safely be reused."**

---

## 1. Problem

Modern multi-agent architectures (such as GrokBot) empower autonomous agents to share a computer, browse the web, interact with remote APIs, and hand off tasks. However, multiple agents frequently encounter identical or overlapping sub-tasks (e.g. retrieving FAA flight restrictions, inspecting cloud infrastructure, or querying financial filings).

Without coordination, each agent executes the same expensive browser navigation or API traversal repeatedly:
- High latency overhead (8–15 seconds per browser lookup)
- Wasted API rate limits and token budgets
- Redundant external network traffic

---

## 2. Existing Limitation: Why Semantic Similarity Fails

Naive agent caching relies solely on embedding cosine similarity:
```python
# INSUFFICIENT AND UNSAFE
if embedding_similarity(query, candidate.semantic_key) >= threshold:
    return cached_payload
```

In agent tool execution, **semantic similarity alone causes catastrophic cache corruption**:
- **Date Changes**: *"Find FAA restrictions near Boca Chica for September 4"* and *"Find FAA restrictions near Boca Chica for September 5"* have a semantic similarity of **0.93+**, yet returning Sept 4 data for Sept 5 is hazardous.
- **Entity & Resource IDs**: *"Check payment intent pi_1234"* and *"Check payment intent pi_5678"* have **0.95+** similarity.
- **Environment Drift**: *"Status of worker node on prod"* vs. *"Status of worker node on staging"*.

---

## 3. What Axiom Is

> **GrokBot provides shared access to a computer and supports multi-agent workflows; Axiom adds a shared layer for reusing expensive tool executions across those agents.**

Axiom operates as a lightweight, non-invasive Model Context Protocol (MCP) server that memoizes **tool executions** (not LLM prose). It evaluates a multi-factor correctness condition before returning any cached observation:

```python
safe_hit = (
    semantic_similarity >= threshold
    and same_tool
    and arguments_compatible
    and not_expired
    and not_invalidated
)
```

```text
                   GrokBot Agent A (research-bot)      GrokBot Agent B (ops-bot)
                                  │                               │
                                  └───────────────┬───────────────┘
                                                  │ MCP (SSE / stdio)
                                                  ▼
                                       ┌─────────────────────┐
                                       │  Axiom MCP Server   │
                                       └──────────┬──────────┘
                                                  │
                                 ┌────────────────┼────────────────┐
                                 ▼                ▼                ▼
                              Semantic          State            TTL &
                              Retrieval       Matching        Invalidation
                         (MiniLM Embeddings)  (Canonical Args) (Tombstoning)
                                 │                │                │
                                 └────────────────┼────────────────┘
                                                  ▼
                                      SAFE HIT vs. SAFE MISS
                                     (safe_hit decision gate)
                                                  │
                                 ┌────────────────┴────────────────┐
                                 ▼                                 ▼
                             SAFE HIT                          SAFE MISS
                         (cached payload,               (reasons: argument_mismatch,
                          latency ~100ms)                expired, invalidated, etc.)
                                                                   │
                                                                   ▼
                                                            expensive_tool
                                                            (~8-10s execution)
                                                                   │
                                                                   ▼
                                                              cache_write
                                                                   │
                                                                   ▼
                                                             Shared Axiom
                                                            (Shared Memory)
```

---

## 4. Key Capabilities & Design

1. **Candidate Retrieval via Local Embeddings**:
   Uses `sentence-transformers` (`all-MiniLM-L6-v2`) locally to find semantically relevant tool execution candidates.
2. **Recursive Structured Argument Compatibility**:
   Strict identity matching for exact state fields (`date`, `location`, `site`, `account_id`, `resource_id`) alongside case/whitespace/URL normalization.
3. **Domain-Specific Freshness (TTL)**:
   Configurable TTL per domain/tool (e.g., FAA status: 10m, weather: 5m, static docs: 24h).
4. **Tag-Based Invalidation (Tombstoning)**:
   Purges observations across agents when external state changes without permanent destruction, enabling auditability.
5. **Real Telemetry & Savings**:
   Tracks actual observed timestamps and tool durations. Real latency savings are measured against true baseline executions.
6. **Fail-Open Architecture**:
   If the cache layer encounters any internal exception, it immediately returns a safe miss so agent execution is never blocked.

---

## 5. Repository Structure

```text
.
├── axiom/
│   ├── __init__.py
│   ├── server.py              # MCP Server (stdio + HTTP/SSE endpoints)
│   ├── cache.py               # Core AxiomCache decision engine
│   ├── compatibility.py       # Recursive argument compatibility checker
│   ├── models.py              # Pydantic schemas and event types
│   ├── embeddings.py          # Sentence-transformers and cosine similarity
│   ├── telemetry.py           # Real telemetry logger & aggregate metrics
│   ├── storage.py             # In-memory storage & candidate retrieval
│   ├── mock_tools.py          # Deterministic expensive browser tool (~8s)
│   ├── config.py              # Domain TTL policies & server config
│   └── requirements.txt       # Python dependencies
├── dashboard/
│   └── app.py                 # Streamlit live telemetry & playground
├── scripts/
│   ├── calibrate_threshold.py # Empirical similarity calibration script
│   └── run_demo.py            # Automated end-to-end runnable demo
├── tests/
│   ├── test_cache.py          # Core memoization and safety gates
│   ├── test_compatibility.py  # Strict argument compatibility & dates
│   ├── test_ttl.py            # TTL expiration and domain policies
│   └── test_invalidation.py   # Tag tombstoning and re-query misses
└── README.md
```

---

## 6. Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/aryanvvemuri/State-Aware-Tool-Memoization-Layer-for-GrokBot.git
cd State-Aware-Tool-Memoization-Layer-for-GrokBot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Automated Tests

```bash
pytest -v tests/
```

### Running Threshold Calibration

```bash
python scripts/calibrate_threshold.py
```

### Running the End-to-End Demo

```bash
# Run full demo with real ~8s tool delays
python scripts/run_demo.py

# Or run with fast 2s delays
python scripts/run_demo.py --fast
```

### Launching the Live Telemetry Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 7. Empirical Threshold Calibration

To prevent arbitrary threshold selection, Axiom includes an empirical calibration benchmark (`scripts/calibrate_threshold.py`) evaluating 60 query pairs with `all-MiniLM-L6-v2`:
- **Equivalent Pairs Mean Similarity**: `0.8584` (Min: `0.6611`, Max: `0.9480`)
- **Near-Miss Pairs Mean Similarity**: `0.7927` (Min: `0.5107`, Max: `0.9847`)
- **Unrelated Pairs Mean Similarity**: `0.0265` (Min: `-0.1836`, Max: `0.1881`)

### The Calibration Insight: Why Semantic Similarity Inverts Safety
At the calibrated retrieval threshold of **`0.75`**:
- **90.0% Recall (TPR)** on true equivalent paraphrases.
- **0.0% False Positives (FPR)** on unrelated queries.
- **60.0% Near-Miss Passthrough**: 60% of near-miss queries (e.g. changing `September 4` to `September 5` or swapping resource IDs) pass the semantic similarity check because they share ~90% identical tokens.

This empirically proves Axiom's thesis: **semantic retrieval must act strictly as a high-recall candidate filter, while structured canonical argument compatibility guarantees correctness and safety**.

---

## 8. Measured Demo Results
| :--- | :--- | :--- | :--- | :--- |
| **Demo 1 (Bot A)** | *"Find FAA restrictions around Boca Chica for Sept 4"* | **8,012 ms** | Cache Miss → Tool Executed → Write | Stored for reuse |
| **Demo 1 (Bot B)** | *"Are there FAA airspace restrictions around Boca Chica for Sept 4?"* | **40 ms** | **SAFE HIT** (Sim: 0.913) | **197x Speedup / ~7.97s saved** |
| **Demo 2 (Bot C)** | *"Find FAA restrictions around Boca Chica for Sept 5"* | **0.4 ms** | **SAFE MISS** (Sim: 0.985, Arg Mismatch) | **Prevented stale Sept 4 reuse** |
| **Demo 3 (Invalidation)** | `cache_invalidate(["faa", "boca_chica"])` | **0.2 ms** | 1 Record Tombstoned | Freshness maintained |
| **Demo 3 (Bot B Repeat)**| *"Are there FAA airspace restrictions around Boca Chica for Sept 4?"* | **4,015 ms** | Cache Miss → Fresh Tool Executed | Data updated after invalidation |
| **Demo 4 (Concurrency)** | 5 Concurrent callers (`alpha`, `beta`, `gamma`, `delta`, `epsilon`) | **1,003 ms batch** | 1 Leader execution, 4 Coalesced | **4 tool executions avoided** |

---

## 9. GrokBot Integration Guide

To connect GrokBot agents to Axiom:

1. **Run Axiom MCP Server**:
   ```bash
   python -m axiom.server --mode http --port 8000
   ```
2. **Expose with a Tunnel (or Deploy to Cloud)**:
   ```bash
   ngrok http 8000
   ```
3. **Register Custom MCP Server in GrokBot**:
   Add MCP Server endpoint:
   ```json
   {
     "mcpServers": {
       "axiom": {
         "url": "https://<your-ngrok-domain>.ngrok-free.app/mcp/sse"
       }
     }
   }
   ```
4. **Agent System Instructions (for `research-bot` and `ops-bot`)**:
   ```markdown
   Before executing any expensive browser navigation or external API traversal:
   1. Call `cache_lookup(query=..., domain=..., tool_name=..., canonical_args=...)`.
   2. If `hit == true`: Reuse `payload` immediately.
   3. If `hit == false`: Execute the real tool, then call `cache_write(query=..., domain=..., tool_name=..., canonical_args=..., payload=...)`.
   4. When you perform an action that alters state, call `cache_invalidate(tags=[...])`.
   ```
