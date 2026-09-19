"""A Streamable-HTTP MCP server exposing evotrader runs and market data to Claude.

Read-only by design.  The run tools answer questions about runs that have
already happened; the market tools (see tradingview.py) read public TradingView
quotes, technicals, candles and screeners.  Nothing starts a run, places an order or spends
Claude credits.  The URL this is served on is public (see
scripts/tunnel-up.sh), so the tool surface is the security boundary.

Standard library only - no dependency beyond what evotrader already needs.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import tradingview, tvarchive
from .config import EvolutionConfig
from .report import lineage, markdown_report, run_summary
from .store import Store

# Claude sends whichever version it likes; we accept any and answer with one we
# know.  Rejecting an unfamiliar version is what makes the connector check fail
# with an unexplained 400.
DEFAULT_PROTOCOL = "2025-06-18"
KNOWN_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}

SERVER_INFO = {"name": "evotrader", "version": "0.1.0"}


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def _db_path() -> str:
    return os.environ.get("EVOTRADER_DB") or EvolutionConfig().db_path


def _open_store() -> Store:
    """A Store per call.

    http.server hands every request its own thread, and a sqlite3 connection
    belongs to the thread that made it.  Sharing one Store across requests
    raises "SQLite objects created in a thread can only be used in that same
    thread" - but only under concurrency, so sequential testing hides it.
    """
    return Store(_db_path())


def _latest_run_id(store: Store) -> Optional[str]:
    rows = store.list_runs(1)
    return rows[0]["id"] if rows else None


def tool_list_runs(limit: int = 25) -> str:
    with _open_store() as store:
        rows = store.list_runs(int(limit))
        if not rows:
            return "No runs in the database yet."
        out = []
        for row in rows:
            cfg = json.loads(row["config"])
            out.append(
                f"{row['id']}  {row['status']:<11} gens={row['generations'] or 0:<5} "
                f"pop={cfg['population']:<4} breeder={cfg['breeder']:<8} {row['note'][:40]}"
            )
        return "\n".join(out)


def tool_run_report(run_id: str = "", limit: int = 15) -> str:
    with _open_store() as store:
        rid = run_id or _latest_run_id(store)
        if not rid:
            return "No runs in the database yet."
        return markdown_report(store, rid, limit=int(limit))


def tool_run_summary(run_id: str = "") -> str:
    with _open_store() as store:
        rid = run_id or _latest_run_id(store)
        if not rid:
            return "No runs in the database yet."
        return json.dumps(run_summary(store, rid), indent=2, default=str)


def tool_leaderboard(run_id: str = "", limit: int = 20) -> str:
    with _open_store() as store:
        rid = run_id or _latest_run_id(store)
        if not rid:
            return "No runs in the database yet."
        rows = store.leaderboard(rid, limit=int(limit))
        if not rows:
            return f"Run {rid} has no evaluated genomes."
        out = [f"leaderboard for run {rid}"]
        for i, row in enumerate(rows, 1):
            out.append(f"{i:>3}. {row['score']:+8.3f}  gen {row['generation']:>3}  "
                       f"{row['genome_id']}  {(row['name'] or '')[:44]}")
        return "\n".join(out)


def tool_generation_history(run_id: str = "", limit: int = 0) -> str:
    with _open_store() as store:
        rid = run_id or _latest_run_id(store)
        if not rid:
            return "No runs in the database yet."
        hist = store.generation_history(rid, limit=int(limit))
        if not hist:
            return f"Run {rid} has no completed generations."
        return json.dumps(hist, indent=2, default=str)


def tool_inspect_genome(genome_id: str, trades: int = 20) -> str:
    if not genome_id:
        return "genome_id is required."
    with _open_store() as store:
        genome = store.get_genome(genome_id)
        if genome is None:
            return f"Unknown genome {genome_id!r}."
        parts = [genome.describe()]
        if genome.rationale:
            parts.append(f"\nrationale: {genome.rationale}")
        rows = store.conn.execute(
            "SELECT * FROM evaluations WHERE genome_id=?", (genome_id,)).fetchall()
        for row in rows:
            metrics = json.dumps(json.loads(row["metrics"]), indent=2)
            parts.append(f"\ngeneration {row['generation']}: score {row['score']:+.3f}\n{metrics}")
        trade_rows = store.conn.execute(
            "SELECT * FROM trades WHERE genome_id=? ORDER BY seq LIMIT ?",
            (genome_id, int(trades))).fetchall()
        if trade_rows:
            parts.append(f"\nfirst {len(trade_rows)} trades")
            for t in trade_rows:
                parts.append(
                    f"  {t['symbol']:<5} {t['entry_date']} -> {t['exit_date']} "
                    f"({t['bars_held']:>3}b) {t['ret'] * 100:+6.1f}%  "
                    f"in: {t['entry_reason']}  out: {t['exit_reason']}")
        chain = lineage(store, genome_id)
        if len(chain) > 1:
            parts.append("\nancestry (newest first)")
            for g in chain:
                parts.append(f"  gen {g.generation:>4} {g.origin:<9} "
                             f"{g.name[:40]:<40} {g.id}")
        return "\n".join(parts)


def tool_genome_trades(genome_id: str, limit: int = 100) -> str:
    if not genome_id:
        return "genome_id is required."
    with _open_store() as store:
        rows = store.conn.execute(
            "SELECT * FROM trades WHERE genome_id=? ORDER BY seq LIMIT ?",
            (genome_id, int(limit))).fetchall()
        if not rows:
            return f"No trades recorded for genome {genome_id!r}."
        return json.dumps([dict(r) for r in rows], indent=2, default=str)


def tool_generation_reflections(run_id: str = "", limit: int = 5) -> str:
    with _open_store() as store:
        rid = run_id or _latest_run_id(store)
        if not rid:
            return "No runs in the database yet."
        rows = store.generation_reflections(rid, limit=int(limit))
        if not rows:
            return f"Run {rid} has no breeder reflections (mutation-only runs have none)."
        return json.dumps(rows, indent=2, default=str)


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_runs",
        "description": "List evolution runs in the database, newest first.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 25}},
        },
        "handler": tool_list_runs,
    },
    {
        "name": "run_report",
        "description": "Full markdown report for a run: config, generation history "
                       "and leaderboard. Defaults to the most recent run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "defaults to latest run"},
                "limit": {"type": "integer", "default": 15},
            },
        },
        "handler": tool_run_report,
    },
    {
        "name": "run_summary",
        "description": "Compact JSON summary of one run. Defaults to the most recent.",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
        },
        "handler": tool_run_summary,
    },
    {
        "name": "leaderboard",
        "description": "Top genomes in a run, ranked by fitness score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
        },
        "handler": tool_leaderboard,
    },
    {
        "name": "generation_history",
        "description": "Per-generation best/mean scores for a run, as JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "default": 0},
            },
        },
        "handler": tool_generation_history,
    },
    {
        "name": "inspect_genome",
        "description": "One genome in full: its strategy rules, thesis, scores, "
                       "first trades and ancestry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "genome_id": {"type": "string"},
                "trades": {"type": "integer", "default": 20},
            },
            "required": ["genome_id"],
        },
        "handler": tool_inspect_genome,
    },
    {
        "name": "genome_trades",
        "description": "The trade journal for one genome, as JSON rows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "genome_id": {"type": "string"},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["genome_id"],
        },
        "handler": tool_genome_trades,
    },
    {
        "name": "generation_reflections",
        "description": "What the breeder said about each generation - why it wrote "
                       "the agents it did.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
        },
        "handler": tool_generation_reflections,
    },
    {
        "name": "quote",
        "description": "Live TradingView quote for one or more tickers, comma-separated. "
                       "Use EXCHANGE:SYMBOL (NASDAQ:AAPL, CME_MINI:NQ1!, BINANCE:BTCUSDT, "
                       "FX:EURUSD, TVC:DXY) or a bare symbol, which is resolved via "
                       "TradingView search. Price, change, OHLC, volume, 52w range, "
                       "performance and TradingView's technical rating.",
        "inputSchema": {
            "type": "object",
            "properties": {"symbols": {"type": "string"}},
            "required": ["symbols"],
        },
        "handler": tradingview.tool_quote,
    },
    {
        "name": "technicals",
        "description": "TradingView's Technicals page for a ticker on a timeframe: "
                       "summary / moving-average / oscillator ratings (strong sell to "
                       "strong buy) plus RSI, Stoch, CCI, ADX, MACD, EMAs/SMAs 10-200, "
                       "Ichimoku, ATR, Bollinger and pivots.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {"type": "string", "enum": list(tradingview.TIMEFRAMES),
                              "default": "1D"},
            },
            "required": ["symbol"],
        },
        "handler": tradingview.tool_technicals,
    },
    {
        "name": "bars",
        "description": "Historical OHLCV candles from TradingView's chart feed, as CSV "
                       "oldest first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {"type": "string", "enum": list(tradingview.TIMEFRAMES),
                              "default": "1D"},
                "count": {"type": "integer", "default": 60,
                          "description": "number of candles (1-1000)"},
            },
            "required": ["symbol"],
        },
        "handler": tradingview.tool_bars,
    },
    {
        "name": "screener",
        "description": "Run a TradingView screener over a whole market and return the "
                       "top matches.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {"type": "string", "enum": list(tradingview._PRESETS),
                           "default": "gainers"},
                "market": {"type": "string", "enum": list(tradingview.MARKETS),
                           "default": "america"},
                "limit": {"type": "integer", "default": 20},
                "min_price": {"type": "number", "default": 0},
                "min_market_cap": {"type": "number", "default": 0},
            },
        },
        "handler": tradingview.tool_screener,
    },
    {
        "name": "search_symbol",
        "description": "Find TradingView tickers (EXCHANGE:SYMBOL) by name or partial "
                       "symbol. Optional asset_type: stock, futures, crypto, forex, "
                       "index, fund, bond.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "asset_type": {"type": "string", "default": ""},
            },
            "required": ["query"],
        },
        "handler": tradingview.tool_search,
    },
    {
        "name": "archive_status",
        "description": "What TradingView history is stored locally by the batch "
                       "puller (`evotrader tv-archive`): every series and timeframe "
                       "with bar counts, date span and CSV path. The stitched files "
                       "are the long NQ history to backtest on (pass the path to "
                       "`python -m strategies.nq_sweeps --csv`).",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": lambda: tvarchive.format_status(tvarchive.status()),
    },
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def _public_tools() -> List[Dict[str, Any]]:
    return [{k: v for k, v in t.items() if k != "handler"} for t in TOOLS]


# --------------------------------------------------------------------------
# JSON-RPC dispatch
# --------------------------------------------------------------------------

def _result(msg_id: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": payload}


def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a response object, or None for a notification."""
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # A notification has no id and never gets a reply.
    if msg_id is None and isinstance(method, str) and method.startswith("notifications/"):
        return None

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in KNOWN_PROTOCOLS else DEFAULT_PROTOCOL
        return _result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })

    if method in ("tools/list", "tools/list_all"):
        return _result(msg_id, {"tools": _public_tools()})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            return _error(msg_id, -32602, f"unknown tool {name!r}")
        try:
            text = tool["handler"](**args)
        except TypeError as exc:
            return _error(msg_id, -32602, f"bad arguments for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, don't kill the server
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })
        return _result(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})

    if method in ("ping", "notifications/ping"):
        return _result(msg_id, {})

    if method in ("resources/list", "prompts/list"):
        key = method.split("/")[0]
        return _result(msg_id, {key: []})

    return _error(msg_id, -32601, f"method not found: {method}")


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):
    server_version = "evotrader-mcp/0.1"
    protocol_version = "HTTP/1.1"

    endpoint: str = "/mcp"
    token: str = ""
    allowed_origins: Tuple[str, ...] = ()

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _read_body(self) -> bytes:
        """Read the body, chunked or not.

        Claude's client may send Transfer-Encoding: chunked with no
        Content-Length.  Reading only Content-Length then yields an empty body,
        the message never runs, and the server still answers 200 - a connector
        that attaches fine and whose tools silently do nothing.
        """
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            chunks: List[bytes] = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                try:
                    size = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization, MCP-Protocol-Version, Mcp-Session-Id")
        self.send_header("Access-Control-Max-Age", "86400")

    def _respond(self, code: int, payload: Optional[Dict[str, Any]] = None,
                 content_type: str = "application/json") -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(code)
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _authorised(self) -> bool:
        """Check the bearer token and the Origin header.

        Both checks run after the body has been drained by the caller: replying
        without reading the body leaves those bytes at the head of the next
        request on a keep-alive connection, and every later request fails to
        parse for no visible reason.
        """
        if self.token:
            header = self.headers.get("Authorization") or ""
            if header != f"Bearer {self.token}":
                return False
        origin = self.headers.get("Origin")
        if origin and self.allowed_origins and origin not in self.allowed_origins:
            return False
        return True

    def _on_endpoint(self) -> bool:
        return self.path.split("?")[0].rstrip("/") in (self.endpoint.rstrip("/"), "")

    # -- methods -----------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._read_body()
        self.send_response(204)
        self.send_header("Allow", "POST, GET, DELETE, OPTIONS, HEAD")
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self._read_body()
        self._respond(200)

    def do_GET(self) -> None:  # noqa: N802
        self._read_body()
        if self.path.split("?")[0].rstrip("/") == "/health":
            self._respond(200, {"status": "ok", "server": SERVER_INFO,
                                "db": _db_path(), "tools": len(TOOLS)})
            return
        if not self._on_endpoint():
            self._respond(404, {"error": "not found"})
            return
        if not self._authorised():
            self._respond(401, {"error": "unauthorised"})
            return
        # This server never pushes; 405 is the spec's answer for "no SSE here".
        self._respond(405, {"error": "this server does not stream"})

    def do_DELETE(self) -> None:  # noqa: N802
        self._read_body()
        self._respond(204)

    def do_POST(self) -> None:  # noqa: N802
        raw = self._read_body()

        if not self._authorised():
            self._respond(401, {"error": "unauthorised"})
            return

        if self.path.split("?")[0].rstrip("/") == "/health":
            self._respond(200, {"status": "ok"})
            return

        if not self._on_endpoint():
            self._respond(404, {"error": "not found"})
            return

        # An empty POST is a reachability probe, not a message.  Handing it to
        # a JSON parser returns 400, which the connector reports as the
        # thoroughly misleading "Not found".
        if not raw.strip():
            self._respond(200, {"status": "ok", "server": SERVER_INFO})
            return

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._respond(200, _error(None, -32700, f"parse error: {exc}"))
            return

        if isinstance(msg, list):
            responses = [r for r in (handle_message(m) for m in msg) if r is not None]
            if not responses:
                self._respond(202)
                return
            self._respond(200, responses)  # type: ignore[arg-type]
            return

        if not isinstance(msg, dict):
            self._respond(200, _error(None, -32600, "invalid request"))
            return

        response = handle_message(msg)
        if response is None:
            self._respond(202)
            return
        self._respond(200, response)


def serve(host: str = "127.0.0.1", port: int = 8787, endpoint: str = "/mcp",
          token: str = "", allowed_origins: Tuple[str, ...] = ()) -> int:
    MCPHandler.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
    MCPHandler.token = token
    MCPHandler.allowed_origins = allowed_origins

    httpd = ThreadingHTTPServer((host, port), MCPHandler)
    httpd.daemon_threads = True

    print(f"evotrader MCP server on http://{host}:{port}{MCPHandler.endpoint}")
    print(f"database: {_db_path()}")
    print(f"tools:    {', '.join(t['name'] for t in TOOLS)}")
    print(f"auth:     {'bearer token required' if token else 'NONE - anyone with the URL can call every tool'}")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0
