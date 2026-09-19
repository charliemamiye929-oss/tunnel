"""Pull TradingView candles with no login and no dependencies.

    python tv_bars.py CME_MINI:NQ1! 5 > nq_5m.csv
    python tv_bars.py CME_MINI:NQZ2024 5 > nqz24_5m.csv    # expired contract
Resolutions: 1 5 15 30 60 240 1D 1W 1M
"""
import base64, json, os, random, re, socket, ssl, string, struct, sys

def _connect():
    host = "data.tradingview.com"
    s = ssl.create_default_context().wrap_socket(
        socket.create_connection((host, 443), timeout=30), server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /socket.io/websocket?from=chart%2F&type=chart HTTP/1.1\r\nHost: {host}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
               f"Sec-WebSocket-Version: 13\r\nOrigin: https://www.tradingview.com\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(1)
    return s

def _send(s, text):
    data, mask = text.encode(), os.urandom(4)
    n = len(data)
    hdr = bytes([0x81, 0x80 | n]) if n < 126 else bytes([0x81, 0xFE]) + struct.pack(">H", n)
    s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

def _exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf += chunk
    return buf

def _recv(s):
    parts = []
    while True:
        b0, b1 = _exact(s, 2)
        n = b1 & 0x7F
        if n == 126: n = struct.unpack(">H", _exact(s, 2))[0]
        elif n == 127: n = struct.unpack(">Q", _exact(s, 8))[0]
        parts.append(_exact(s, n))
        if b0 & 0x80:
            return b"".join(parts).decode("utf-8", "replace")

def _msg(m, p):
    body = json.dumps({"m": m, "p": p}, separators=(",", ":"))
    return f"~m~{len(body)}~m~{body}"

def fetch_bars(ticker, resolution="5", count=20000):
    """Return [(unix_ts, open, high, low, close, volume), ...] oldest first."""
    cs = "cs_" + "".join(random.choices(string.ascii_lowercase, k=12))
    spec = "=" + json.dumps({"symbol": ticker, "adjustment": "splits"}, separators=(",", ":"))
    s = _connect()
    for m, p in [("set_auth_token", ["unauthorized_user_token"]),  # or your session token
                 ("chart_create_session", [cs, ""]),
                 ("resolve_symbol", [cs, "sym1", spec]),
                 ("create_series", [cs, "s1", "s1", "sym1", resolution, count])]:
        _send(s, _msg(m, p))
    bars = {}
    try:
        while True:
            for part in re.split(r"~m~\d+~m~", _recv(s)):
                if part.startswith("~h~"):                 # heartbeat: echo it back
                    _send(s, f"~m~{len(part)}~m~{part}")
                    continue
                if not part.startswith("{"):
                    continue
                msg = json.loads(part)
                if msg.get("m") in ("symbol_error", "series_error", "critical_error"):
                    raise RuntimeError(f"{ticker}: {msg}")
                if msg.get("m") in ("timescale_update", "du"):
                    for b in msg["p"][1].get("s1", {}).get("s", []):
                        v = b["v"]
                        bars[v[0]] = tuple(v[:5]) + (v[5] if len(v) > 5 else 0,)
                if msg.get("m") == "series_completed":
                    return [bars[t] for t in sorted(bars)]
    finally:
        s.close()

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "CME_MINI:NQ1!"
    res = sys.argv[2] if len(sys.argv) > 2 else "5"
    print("ts,open,high,low,close,volume")
    for row in fetch_bars(ticker, res):
        print(",".join(f"{x:.10g}" if i else str(int(x)) for i, x in enumerate(row)))
