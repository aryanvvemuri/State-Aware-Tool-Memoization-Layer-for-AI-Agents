"""Axiom MCP Server & REST API.

Exposes cache_lookup, cache_write, cache_invalidate, and expensive_browser_lookup
via Model Context Protocol (MCP) and HTTP/SSE for multi-agent coordination.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from axiom.cache import AxiomCache
from axiom.config import AXIOM_HOST, AXIOM_PORT, DEFAULT_MOCK_SLEEP_SEC
from axiom.mock_tools import expensive_browser_lookup
from axiom.models import (
    CacheInvalidateRequest,
    CacheLookupRequest,
    CacheWriteRequest,
)
from axiom.telemetry import global_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("axiom.server")

# Global Cache Instance
axiom_cache = AxiomCache()

# FastAPI App for remote HTTP/SSE access
app = FastAPI(
    title="Axiom: State-Aware Tool Memoization Layer for GrokBot",
    description="Shared memory coordination layer for multi-agent tool execution memoization.",
    version="0.1.0",
)

app.add_middleware(
    CORSMSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core Tool Functions
# ---------------------------------------------------------------------------

def handle_cache_lookup(
    query: str,
    domain: str,
    tool_name: str,
    canonical_args: Dict[str, Any],
    tags: Optional[List[str]] = None,
    bot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """MCP tool implementation for cache_lookup."""
    req = CacheLookupRequest(
        query=query,
        domain=domain,
        tool_name=tool_name,
        canonical_args=canonical_args,
        tags=tags or [],
        bot_id=bot_id,
    )
    res = axiom_cache.lookup(req)
    return res.model_dump()


def handle_cache_write(
    query: str,
    domain: str,
    tool_name: str,
    canonical_args: Dict[str, Any],
    payload: Any,
    ttl_seconds: int = 3600,
    tags: Optional[List[str]] = None,
    source_bot: str = "agent",
) -> Dict[str, Any]:
    """MCP tool implementation for cache_write."""
    req = CacheWriteRequest(
        query=query,
        domain=domain,
        tool_name=tool_name,
        canonical_args=canonical_args,
        payload=payload,
        ttl_seconds=ttl_seconds,
        tags=tags or [],
        source_bot=source_bot,
    )
    res = axiom_cache.write(req)
    return res.model_dump()


def handle_cache_invalidate(
    tags: List[str],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """MCP tool implementation for cache_invalidate."""
    req = CacheInvalidateRequest(tags=tags, domain=domain)
    res = axiom_cache.invalidate(req)
    return res.model_dump()


def handle_expensive_browser_lookup(
    site: str,
    location: str,
    date: str,
    query: str,
    sleep_seconds: Optional[float] = None,
    bot_id: str = "research-bot",
) -> Dict[str, Any]:
    """Deterministic mock tool for testing."""
    return expensive_browser_lookup(
        site=site,
        location=location,
        date=date,
        query=query,
        sleep_seconds=sleep_seconds,
        bot_id=bot_id,
        telemetry=axiom_cache.telemetry,
    )


# ---------------------------------------------------------------------------
# HTTP REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Axiom", "records": len(axiom_cache.storage.get_all_records())}


@app.post("/api/lookup")
def api_lookup(req: CacheLookupRequest):
    res = axiom_cache.lookup(req)
    return res.model_dump()


@app.post("/api/write")
def api_write(req: CacheWriteRequest):
    res = axiom_cache.write(req)
    return res.model_dump()


@app.post("/api/invalidate")
def api_invalidate(req: CacheInvalidateRequest):
    res = axiom_cache.invalidate(req)
    return res.model_dump()


@app.get("/api/telemetry")
def api_telemetry(limit: int = 100):
    events = axiom_cache.telemetry.get_events(limit=limit)
    return [e.model_dump() for e in events]


@app.get("/api/stats")
def api_stats():
    return axiom_cache.telemetry.get_aggregate_stats()


@app.post("/api/tool/expensive_browser_lookup")
def api_expensive_tool(body: Dict[str, Any]):
    return handle_expensive_browser_lookup(
        site=body.get("site", "faa.gov"),
        location=body.get("location", "Boca Chica"),
        date=body.get("date", "2026-09-04"),
        query=body.get("query", "FAA flight restrictions"),
        sleep_seconds=body.get("sleep_seconds"),
        bot_id=body.get("bot_id", "research-bot"),
    )


# ---------------------------------------------------------------------------
# MCP JSON-RPC Protocol (over stdio and HTTP/SSE)
# ---------------------------------------------------------------------------

MCP_TOOLS_MANIFEST = [
    {
        "name": "cache_lookup",
        "description": "State-aware cache lookup. Evaluates semantic similarity, canonical arguments, TTL, and invalidation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query or intent"},
                "domain": {"type": "string", "description": "Domain scope (e.g. 'faa', 'weather')"},
                "tool_name": {"type": "string", "description": "Canonical tool name (e.g. 'expensive_browser_lookup')"},
                "canonical_args": {"type": "object", "description": "Key state arguments (e.g. {'site': ..., 'location': ..., 'date': ...})"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional dependency tags"},
                "bot_id": {"type": "string", "description": "Identifier of the calling bot"},
            },
            "required": ["query", "domain", "tool_name", "canonical_args"],
        },
    },
    {
        "name": "cache_write",
        "description": "Store a tool execution observation in Axiom shared memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query or intent"},
                "domain": {"type": "string", "description": "Domain scope"},
                "tool_name": {"type": "string", "description": "Canonical tool name"},
                "canonical_args": {"type": "object", "description": "Key state arguments"},
                "payload": {"type": "object", "description": "Tool result payload"},
                "ttl_seconds": {"type": "integer", "description": "Time to live in seconds"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Invalidation tags"},
                "source_bot": {"type": "string", "description": "Bot that executed the tool"},
            },
            "required": ["query", "domain", "tool_name", "canonical_args", "payload"],
        },
    },
    {
        "name": "cache_invalidate",
        "description": "Invalidate (tombstone) cached observations matching specified tags.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to invalidate"},
                "domain": {"type": "string", "description": "Optional domain filter"},
            },
            "required": ["tags"],
        },
    },
    {
        "name": "expensive_browser_lookup",
        "description": "Execute mock expensive browser navigation tool (~8s execution).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "site": {"type": "string"},
                "location": {"type": "string"},
                "date": {"type": "string"},
                "query": {"type": "string"},
                "sleep_seconds": {"type": "number"},
                "bot_id": {"type": "string"},
            },
            "required": ["site", "location", "date", "query"],
        },
    },
]


def dispatch_jsonrpc(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a JSON-RPC 2.0 request according to MCP specifications."""
    req_id = request_data.get("id")
    method = request_data.get("method")
    params = request_data.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "axiom-memoization", "version": "0.1.0"},
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS_MANIFEST},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "cache_lookup":
                res = handle_cache_lookup(
                    query=arguments.get("query", ""),
                    domain=arguments.get("domain", ""),
                    tool_name=arguments.get("tool_name", ""),
                    canonical_args=arguments.get("canonical_args", {}),
                    tags=arguments.get("tags"),
                    bot_id=arguments.get("bot_id"),
                )
            elif tool_name == "cache_write":
                res = handle_cache_write(
                    query=arguments.get("query", ""),
                    domain=arguments.get("domain", ""),
                    tool_name=arguments.get("tool_name", ""),
                    canonical_args=arguments.get("canonical_args", {}),
                    payload=arguments.get("payload", {}),
                    ttl_seconds=arguments.get("ttl_seconds", 3600),
                    tags=arguments.get("tags"),
                    source_bot=arguments.get("source_bot", "agent"),
                )
            elif tool_name == "cache_invalidate":
                res = handle_cache_invalidate(
                    tags=arguments.get("tags", []),
                    domain=arguments.get("domain"),
                )
            elif tool_name == "expensive_browser_lookup":
                res = handle_expensive_browser_lookup(
                    site=arguments.get("site", "faa.gov"),
                    location=arguments.get("location", "Boca Chica"),
                    date=arguments.get("date", "2026-09-04"),
                    query=arguments.get("query", ""),
                    sleep_seconds=arguments.get("sleep_seconds"),
                    bot_id=arguments.get("bot_id", "bot"),
                )
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                },
            }
        except Exception as e:
            logger.error("Tool execution error in %s: %s", tool_name, e, exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    elif method == "notifications/initialized":
        return None

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method '{method}' not found"},
    }


# SSE Stream for MCP
@app.post("/mcp/messages")
async def mcp_messages(request: Request):
    body = await request.json()
    resp = dispatch_jsonrpc(body)
    return JSONResponse(content=resp if resp is not None else {})


@app.get("/mcp/sse")
async def mcp_sse(request: Request):
    """Standard MCP Server-Sent Events endpoint."""
    async def event_generator():
        # Yield initial endpoint event
        endpoint_event = {
            "type": "endpoint",
            "endpoint": "/mcp/messages",
        }
        yield f"event: endpoint\ndata: /mcp/messages\n\n"
        # Keep-alive loop
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(15)
            yield ": keepalive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def run_stdio():
    """Run MCP server in standard I/O mode."""
    logger.info("Axiom MCP Server starting in stdio mode...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = dispatch_jsonrpc(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            logger.error("Failed processing stdio line: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Axiom MCP Server")
    parser.add_argument("--mode", choices=["http", "stdio"], default="http", help="Server mode")
    parser.add_argument("--host", default=AXIOM_HOST, help="HTTP host")
    parser.add_argument("--port", type=int, default=AXIOM_PORT, help="HTTP port")
    args = parser.parse_args()

    if args.mode == "stdio":
        run_stdio()
    else:
        logger.info("Starting Axiom HTTP/SSE server on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
