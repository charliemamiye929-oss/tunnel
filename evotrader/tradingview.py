"""TradingView market data for the MCP server.

Three public TradingView services, none of which needs a login:

* the scanner (``scanner.tradingview.com/global/scan``) - live quotes, the
  technical ratings shown on TradingView's "Technicals" gauge, indicator
  values, performance and fundamentals, and screening across a whole market;
* symbol search (``symbol-search.tradingview.com``) - turns "nvidia" or "NQ"
  into a full ``EXCHANGE:SYMBOL`` ticker;
* the chart websocket (``data.tradingview.com``) - historical OHLCV candles,
  the same feed the chart page uses.

Unauthenticated data carries TradingView's normal delays (e.g. ~10 min on CME
futures).  Everything here reads; nothing places orders.  Standard library
only, including a minimal websocket client, so the server gains no dependency.
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import socket
import ssl
import string
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCAN_URL = "https://scanner.tradingview.com/{market}/scan"
SEARCH_URL = "https://symbol-search.tradingview.com/symbol_search/v3/"
WS_HOST = "data.tradingview.com"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Origin": "https://www.tradingview.com",
            "Referer": "https://www.tradingview.com/"}

# TradingView tickers: NASDAQ:AAPL, CME_MINI:NQ1!, BINANCE:BTCUSDT, FX:EURUSD,
# TVC:DXY, or a bare symbol that gets resolved through search.
_TICKER_RE = re.compile(r"^[A-Za-z0-9_]{1,24}:[A-Za-z0-9_.!&\-]{1,32}$|^[A-Za-z0-9_.!&\-]{1,32}$")
MAX_SYMBOLS = 12
CACHE_TTL = 20.0

# Chart-timeframe name -> (scanner column suffix, websocket resolution).
TIMEFRAMES: Dict[str, Tuple[str, str]] = {
    "1m": ("|1", "1"), "5m": ("|5", "5"), "15m": ("|15", "15"), "30m": ("|30", "30"),
    "1h": ("|60", "60"), "2h": ("|120", "120"), "4h": ("|240", "240"),
    "1D": ("", "1D"), "1W": ("|1W", "1W"), "1M": ("|1M", "1M"),
}
_TF_ALIASES = {"1d": "1D", "d": "1D", "1w": "1W", "w": "1W", "1mo": "1M", "60m": "1h",
               "60": "1h", "240": "4h", "1": "1m", "5": "5m", "15": "15m", "30": "30m"}

MARKETS = ("america", "crypto", "futures", "forex", "cfd", "uk", "germany", "india",
           "canada", "japan", "australia", "global")


class TVError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

_cache: Dict[str, Tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _http_json(url: str, body: Optional[Dict[str, Any]] = None, *,
               timeout: int = 15) -> Any:
    key = url + (json.dumps(body, sort_keys=True) if body is not None else "")
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
    data = json.dumps(body).encode() if body is not None else None
    headers = dict(_HEADERS)
    if data is not None:
        headers["Content-Type"] = "application/json"
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if 400 <= exc.code < 500:
                raise TVError(f"TradingView rejected the request ({exc.code}): {detail}")
            last = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        time.sleep(0.5 * 2 ** attempt)
    else:
        raise TVError(f"TradingView request failed: {last}")
    with _cache_lock:
        _cache[key] = (time.time(), payload)
    return payload


def _timeframe(tf: str) -> str:
    tf = (tf or "1D").strip()
    tf = _TF_ALIASES.get(tf, _TF_ALIASES.get(tf.lower(), tf))
    if tf not in TIMEFRAMES:
        raise TVError(f"timeframe must be one of {', '.join(TIMEFRAMES)}")
    return tf


# --------------------------------------------------------------------------
# symbol resolution
# --------------------------------------------------------------------------

def search(query: str, limit: int = 10, asset_type: str = "") -> List[Dict[str, Any]]:
    params = {"text": query[:60], "hl": "0", "exchange": "", "lang": "en",
              "search_type": asset_type, "domain": "production"}
    payload = _http_json(SEARCH_URL + "?" + urllib.parse.urlencode(params))
    items = payload.get("symbols", []) if isinstance(payload, dict) else payload
    return list(items or [])[:limit]


def _full_ticker(item: Dict[str, Any], wanted: str = "") -> str:
    exchange = item.get("prefix") or item.get("source_id") or item.get("exchange", "")
    symbol = item["symbol"]
    contracts = [c.get("symbol") for c in item.get("contracts") or [] if c.get("symbol")]
    if contracts:
        symbol = wanted if wanted in contracts else contracts[0]
    return f"{exchange}:{symbol}"


def resolve(symbol: str) -> str:
    """Return EXCHANGE:SYMBOL, searching when no exchange was given."""
    sym = (symbol or "").strip().upper()
    if not _TICKER_RE.match(sym):
        raise TVError(f"not a valid TradingView ticker: {symbol!r}")
    if ":" in sym:
        return sym
    base = re.sub(r"\d+!$", "", sym)
    results = search(base, limit=10)
    if not results:
        raise TVError(f"no TradingView symbol matches {symbol!r}")
    exact = [r for r in results if r.get("symbol", "").upper() == base]
    return _full_ticker((exact or results)[0], wanted=sym)


def _resolve_many(symbols: str | Sequence[str]) -> List[str]:
    items = symbols.split(",") if isinstance(symbols, str) else list(symbols)
    items = [s.strip() for s in items if str(s).strip()]
    if not items:
        raise TVError("give at least one ticker")
    if len(items) > MAX_SYMBOLS:
        raise TVError(f"at most {MAX_SYMBOLS} tickers per call")
    return list(dict.fromkeys(resolve(s) for s in items))


def scan(tickers: Sequence[str], columns: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    payload = _http_json(SCAN_URL.format(market="global"),
                         {"symbols": {"tickers": list(tickers)}, "columns": list(columns)})
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("data") or []:
        out[row["s"]] = dict(zip(columns, row["d"]))
    return out


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _num(x: Any, digits: int = 2) -> str:
    if not isinstance(x, (int, float)) or x != x:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.{min(digits, 2)}f}"
    if abs(x) >= 10:
        return f"{x:.{digits}f}"
    # FX and small-cap crypto live in the fourth or fifth decimal place.
    return f"{x:.{max(digits + 3, 6)}g}"


def _pct(x: Any) -> str:
    return f"{x:+.2f}%" if isinstance(x, (int, float)) and x == x else "n/a"


def _big(x: Any) -> str:
    if not isinstance(x, (int, float)) or x != x:
        return "n/a"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= size:
            return f"{x / size:.2f}{unit}"
    return f"{x:.0f}"


def rating_label(value: Any) -> str:
    """TradingView's own buckets for Recommend.* values in [-1, 1]."""
    if not isinstance(value, (int, float)) or value != value:
        return "n/a"
    if value >= 0.5:
        return "STRONG BUY"
    if value >= 0.1:
        return "BUY"
    if value > -0.1:
        return "NEUTRAL"
    if value > -0.5:
        return "SELL"
    return "STRONG SELL"


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

_QUOTE_COLS = ["description", "type", "exchange", "currency", "close", "change",
               "change_abs", "open", "high", "low", "volume", "price_52_week_high",
               "price_52_week_low", "market_cap_basic", "Recommend.All",
               "Perf.W", "Perf.1M", "Perf.YTD", "Perf.Y"]


def tool_quote(symbols: str) -> str:
    tickers = _resolve_many(symbols)
    rows = scan(tickers, _QUOTE_COLS)
    out = []
    for t in tickers:
        r = rows.get(t)
        if not r:
            out.append(f"{t}: TradingView returned no data")
            continue
        out.append(
            f"{t} - {r['description']} ({r['type']}, {r['currency']})\n"
            f"  last {_num(r['close'])}  {_pct(r['change'])} ({_num(r['change_abs'])})"
            f"   O {_num(r['open'])} H {_num(r['high'])} L {_num(r['low'])}"
            f"   vol {_big(r['volume'])}\n"
            f"  52w {_num(r['price_52_week_low'])} - {_num(r['price_52_week_high'])}"
            f"   mkt cap {_big(r['market_cap_basic'])}"
            f"   perf 1W {_pct(r['Perf.W'])} 1M {_pct(r['Perf.1M'])} "
            f"YTD {_pct(r['Perf.YTD'])} 1Y {_pct(r['Perf.Y'])}\n"
            f"  TradingView rating (1D): {rating_label(r['Recommend.All'])}")
    out.append("\nSource: TradingView (unauthenticated feed; some exchanges are delayed).")
    return "\n".join(out)


_TECH_BASE = ["Recommend.All", "Recommend.MA", "Recommend.Other", "close",
              "RSI", "Stoch.K", "Stoch.D", "CCI20", "ADX", "ADX+DI", "ADX-DI", "AO",
              "Mom", "MACD.macd", "MACD.signal", "W.R", "BBPower", "UO",
              "EMA10", "SMA10", "EMA20", "SMA20", "EMA50", "SMA50", "EMA100",
              "SMA100", "EMA200", "SMA200", "Ichimoku.BLine", "VWMA", "HullMA9",
              "ATR", "BB.upper", "BB.lower", "Pivot.M.Classic.S1",
              "Pivot.M.Classic.Middle", "Pivot.M.Classic.R1"]


def tool_technicals(symbol: str, timeframe: str = "1D") -> str:
    tf = _timeframe(timeframe)
    suffix = TIMEFRAMES[tf][0]
    ticker = resolve(symbol)
    cols = [c + suffix for c in _TECH_BASE]
    row = scan([ticker], cols).get(ticker)
    if not row:
        return f"{ticker}: TradingView returned no technicals."
    v = {c: row[c + suffix] for c in _TECH_BASE}
    price = v["close"]
    out = [f"{ticker} technicals on the {tf} timeframe (TradingView)",
           f"price {_num(price)}",
           "",
           f"SUMMARY:         {rating_label(v['Recommend.All'])} ({_num(v['Recommend.All'], 3)})",
           f"moving averages: {rating_label(v['Recommend.MA'])} ({_num(v['Recommend.MA'], 3)})",
           f"oscillators:     {rating_label(v['Recommend.Other'])} ({_num(v['Recommend.Other'], 3)})",
           "",
           "oscillators",
           f"  RSI(14) {_num(v['RSI'])}   Stoch %K/%D {_num(v['Stoch.K'])}/{_num(v['Stoch.D'])}"
           f"   CCI(20) {_num(v['CCI20'])}   W%R {_num(v['W.R'])}",
           f"  MACD {_num(v['MACD.macd'], 3)} vs signal {_num(v['MACD.signal'], 3)}"
           f"   ADX {_num(v['ADX'])} (+DI {_num(v['ADX+DI'])} / -DI {_num(v['ADX-DI'])})",
           f"  AO {_num(v['AO'], 3)}   Momentum {_num(v['Mom'], 3)}   Bull/Bear power "
           f"{_num(v['BBPower'], 3)}   UO {_num(v['UO'])}",
           "",
           "moving averages"]
    for n in (10, 20, 50, 100, 200):
        parts = []
        for kind in ("EMA", "SMA"):
            ma = v[f"{kind}{n}"]
            side = "" if not isinstance(ma, (int, float)) or not isinstance(price, (int, float)) \
                else (" above" if price > ma else " below")
            parts.append(f"{kind}{n} {_num(ma)}{side}")
        out.append("  " + "   ".join(parts))
    out += [f"  Ichimoku base {_num(v['Ichimoku.BLine'])}   VWMA(20) {_num(v['VWMA'])}"
            f"   Hull(9) {_num(v['HullMA9'])}",
            "",
            f"volatility: ATR(14) {_num(v['ATR'])}   Bollinger {_num(v['BB.lower'])} - "
            f"{_num(v['BB.upper'])}",
            f"monthly pivots (classic): S1 {_num(v['Pivot.M.Classic.S1'])}   P "
            f"{_num(v['Pivot.M.Classic.Middle'])}   R1 {_num(v['Pivot.M.Classic.R1'])}",
            "",
            "Ratings are TradingView's mechanical indicator vote, not advice."]
    return "\n".join(out)


# -- chart websocket -------------------------------------------------------

def _ws_connect(timeout: float) -> ssl.SSLSocket:
    raw = socket.create_connection((WS_HOST, 443), timeout=timeout)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=WS_HOST)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((
        f"GET /socket.io/websocket?from=chart%2F&type=chart HTTP/1.1\r\n"
        f"Host: {WS_HOST}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        f"Origin: https://www.tradingview.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(1)
        if not chunk:
            raise TVError("TradingView closed the chart connection during handshake")
        head += chunk
    status = head.split(b"\r\n", 1)[0].decode("latin-1")
    if " 101 " not in status:
        raise TVError(f"chart websocket refused: {status}")
    return sock


def _ws_send(sock: ssl.SSLSocket, text: str) -> None:
    data = text.encode()
    n = len(data)
    if n < 126:
        header = bytes([0x81, 0x80 | n])
    elif n < 65536:
        header = bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
    else:
        header = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", n)
    mask = os.urandom(4)
    sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))


def _recv_exact(sock: ssl.SSLSocket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise TVError("TradingView closed the chart connection")
        buf += chunk
    return buf


def _ws_recv(sock: ssl.SSLSocket) -> str:
    """Return one text message, reassembling fragments; skip control frames."""
    parts: List[bytes] = []
    while True:
        b0, b1 = _recv_exact(sock, 2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", _recv_exact(sock, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
        payload = _recv_exact(sock, n)
        opcode = b0 & 0x0F
        if opcode == 0x8:
            raise TVError("TradingView closed the chart connection")
        if opcode in (0x9, 0xA):
            continue
        parts.append(payload)
        if b0 & 0x80:
            return b"".join(parts).decode("utf-8", "replace")


def _frame(method: str, params: List[Any]) -> str:
    body = json.dumps({"m": method, "p": params}, separators=(",", ":"))
    return f"~m~{len(body)}~m~{body}"


def fetch_bars(ticker: str, timeframe: str = "1D", count: int = 100,
               timeout: float = 20.0) -> List[Tuple[float, float, float, float, float, float]]:
    """(unix time, open, high, low, close, volume) candles, oldest first."""
    resolution = TIMEFRAMES[_timeframe(timeframe)][1]
    session = "cs_" + "".join(random.choices(string.ascii_lowercase, k=12))
    symbol_spec = "=" + json.dumps({"symbol": ticker, "adjustment": "splits",
                                    "session": "regular"}, separators=(",", ":"))
    sock = _ws_connect(timeout)
    bars: Dict[float, Tuple[float, ...]] = {}
    try:
        for method, params in (
                ("set_auth_token", ["unauthorized_user_token"]),
                ("chart_create_session", [session, ""]),
                ("resolve_symbol", [session, "sym1", symbol_spec]),
                ("create_series", [session, "s1", "s1", "sym1", resolution, int(count)])):
            _ws_send(sock, _frame(method, params))
        deadline = time.time() + timeout
        while time.time() < deadline:
            for part in re.split(r"~m~\d+~m~", _ws_recv(sock)):
                if not part:
                    continue
                if part.startswith("~h~"):  # heartbeat: echo it or get dropped
                    _ws_send(sock, f"~m~{len(part)}~m~{part}")
                    continue
                try:
                    msg = json.loads(part)
                except json.JSONDecodeError:
                    continue
                method = msg.get("m")
                if method in ("symbol_error", "series_error", "critical_error"):
                    raise TVError(f"{ticker}: TradingView {method}: {msg.get('p')}")
                if method in ("timescale_update", "du"):
                    series = (msg.get("p") or [None, {}])[1].get("s1", {})
                    for bar in series.get("s") or []:
                        vals = bar.get("v") or []
                        if len(vals) >= 5:
                            bars[vals[0]] = tuple(vals[:5]) + (vals[5] if len(vals) > 5 else 0.0,)
                if method == "series_completed":
                    return [bars[t] for t in sorted(bars)]
        raise TVError(f"{ticker}: timed out waiting for TradingView candles")
    finally:
        try:
            sock.close()
        except OSError:
            pass


def tool_bars(symbol: str, timeframe: str = "1D", count: int = 60) -> str:
    tf = _timeframe(timeframe)
    ticker = resolve(symbol)
    count = max(1, min(int(count), 1000))
    bars = fetch_bars(ticker, tf, count)
    if not bars:
        return f"{ticker}: no candles on the {tf} timeframe."
    intraday = tf not in ("1D", "1W", "1M")
    fmt = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
    out = [f"{ticker} {tf} candles from TradingView, {len(bars)} bars"
           + (" (times UTC)" if intraday else ""),
           "time,open,high,low,close,volume"]
    for t, o, h, l, c, v in bars:
        stamp = datetime.fromtimestamp(t, tz=timezone.utc).strftime(fmt)
        out.append(f"{stamp},{o:.10g},{h:.10g},{l:.10g},{c:.10g},{v:.0f}")
    return "\n".join(out)


def tool_search(query: str, limit: int = 10, asset_type: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "query is required."
    results = search(q, limit=max(1, min(int(limit), 30)), asset_type=asset_type)
    if not results:
        return f"No TradingView symbols match {q!r}."
    return "\n".join(f"{_full_ticker(r):<26} {r.get('type', ''):<10} "
                     f"{r.get('description', '')}" for r in results)


# -- screener -------------------------------------------------------------

_PRESETS: Dict[str, Tuple[str, str, List[Dict[str, Any]]]] = {
    "gainers": ("change", "desc", []),
    "losers": ("change", "asc", []),
    "most_active": ("volume", "desc", []),
    "strong_buy": ("Recommend.All", "desc",
                   [{"left": "Recommend.All", "operation": "egreater", "right": 0.5}]),
    "strong_sell": ("Recommend.All", "asc",
                    [{"left": "Recommend.All", "operation": "eless", "right": -0.5}]),
    "oversold": ("RSI", "asc", [{"left": "RSI", "operation": "less", "right": 30}]),
    "overbought": ("RSI", "desc", [{"left": "RSI", "operation": "greater", "right": 70}]),
    "new_52w_high": ("change", "desc",
                     [{"left": "close", "operation": "egreater", "right": "price_52_week_high"}]),
    "new_52w_low": ("change", "asc",
                    [{"left": "close", "operation": "eless", "right": "price_52_week_low"}]),
}

# Keep US stock screens to real, tradeable listings rather than every warrant.
_STOCK_FILTERS = [{"left": "type", "operation": "in_range", "right": ["stock", "dr", "fund"]},
                  {"left": "is_primary", "operation": "equal", "right": True},
                  {"left": "volume", "operation": "greater", "right": 100_000}]

# The crypto market lists every DEX pool and perpetual swap; keep screens to
# spot pairs on major exchanges that actually trade.
_CRYPTO_FILTERS = [{"left": "type", "operation": "equal", "right": "spot"},
                   {"left": "exchange", "operation": "in_range",
                    "right": ["BINANCE", "COINBASE", "KRAKEN", "BYBIT", "OKX", "BITSTAMP"]},
                   {"left": "24h_vol|5", "operation": "greater", "right": 1_000_000}]


def tool_screener(preset: str = "gainers", market: str = "america", limit: int = 20,
                  min_price: float = 0.0, min_market_cap: float = 0.0) -> str:
    if preset not in _PRESETS:
        raise TVError(f"preset must be one of {', '.join(_PRESETS)}")
    if market not in MARKETS:
        raise TVError(f"market must be one of {', '.join(MARKETS)}")
    sort_by, order, filters = _PRESETS[preset]
    filters = list(filters)
    if market == "crypto":
        filters += _CRYPTO_FILTERS
    elif market == "futures":
        # Thousands of listed expiries never trade; only screen live contracts.
        filters.append({"left": "volume", "operation": "greater", "right": 1000})
    elif market not in ("forex", "cfd"):
        filters += _STOCK_FILTERS
    if min_price:
        filters.append({"left": "close", "operation": "egreater", "right": float(min_price)})
    if min_market_cap:
        filters.append({"left": "market_cap_basic", "operation": "egreater",
                        "right": float(min_market_cap)})
    cols = ["name", "description", "close", "change", "volume", "market_cap_basic",
            "RSI", "Recommend.All"]
    body = {"filter": filters, "columns": cols,
            "sort": {"sortBy": sort_by, "sortOrder": order},
            "range": [0, max(1, min(int(limit), 100))]}
    payload = _http_json(SCAN_URL.format(market=market), body)
    rows = payload.get("data") or []
    if not rows:
        return f"No {market} symbols match the {preset} screen."
    out = [f"TradingView screener: {preset} in {market} "
           f"({payload.get('totalCount', len(rows))} matches, showing {len(rows)})",
           f"{'ticker':<24}{'last':>12}{'chg':>9}{'volume':>10}{'mkt cap':>10}"
           f"{'RSI':>7}  rating       name"]
    for row in rows:
        r = dict(zip(cols, row["d"]))
        out.append(f"{row['s']:<24}{_num(r['close']):>12}{_pct(r['change']):>9}"
                   f"{_big(r['volume']):>10}{_big(r['market_cap_basic']):>10}"
                   f"{_num(r['RSI'], 1):>7}  {rating_label(r['Recommend.All']):<12} "
                   f"{(r['description'] or '')[:36]}")
    return "\n".join(out)
