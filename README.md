# tunnel

A local MCP server, reachable from Claude on the web, desktop and phone through
a Cloudflare tunnel — plus the TradingView market-data tools it serves and a
batch puller that builds years of NQ candles out of a feed that only hands out
5,000 bars at a time.

Three parts, each usable on its own:

| part | what it is |
|---|---|
| [`TUNNEL-GUIDE.md`](TUNNEL-GUIDE.md) + `scripts/tunnel*` | how to run any MCP server as a Claude **custom connector** over a Cloudflare tunnel, and every failure met on the way |
| `evotrader/tradingview.py` | TradingView quotes, technicals, candles, screener and symbol search — no login, no dependencies |
| `evotrader/tvarchive.py` | the batch puller: every quarterly futures contract since 2015, merged and stitched into CSVs that grow each run |
| `scripts/tv_bars.py` | one standalone file that pulls candles from TradingView. No repo, no install. |

Everything is standard-library Python (numpy is only needed by the evolution
engine, not by the data tools).

---

## Quick start

```bash
python scripts/tv_bars.py CME_MINI:NQ1! 5 > nq_5m.csv    # ~5,500 NQ 5m candles
```

Serve it to Claude as a connector:

```bash
pip install -e .
evotrader serve-http                 # http://127.0.0.1:8787/mcp
scripts/tunnel                       # public URL to paste into Claude
```

Build the long NQ history:

```bash
evotrader tv-archive                 # ~7 min, ~200 pulls
evotrader tv-archive --status        # what is stored
```

---

## Why the archive exists

Without a login TradingView serves only its newest bars per series — about
5,500 NQ 5m candles, four weeks. Measured limits:

| timeframe | bars per series | reaches back |
|---|---|---|
| 1m | 6,900 | ~1 week |
| 5m | 5,472 | ~4 weeks |
| 1h | 10,127 | ~20 months |
| 1D | 6,884 | ~27 years |

Every **expired quarterly contract** (`CME_MINI:NQZ2024`, `NQH2019`, …) is its
own series with its own window ending at its expiry, and those resolve back to
2015. Pulling all of them turns four weeks into years. The archive also merges
every run into the stored CSV, so the continuous series grows without gaps from
the day you start.

One full NQ run (203 pulls, ~7 minutes) produces:

| timeframe | bars | sessions | span |
|---|---|---|---|
| 1m | 327,242 | 317 | 2015-03 → 2026-09 |
| 5m | 250,368 | 960 | 2015-02 → 2026-09 |
| 15m | 255,321 | 2,822 | 2015-01 → 2026-09 |
| 1h | 70,896 | 3,265 | 2013-12 → 2026-09 |
| 1D | 6,884 | 6,884 | 1999-06 → 2026-09 |

**The intraday history is chunked, not continuous.** Each contract keeps only
its last stretch, so 5m arrives as ~19-session blocks near each quarterly
expiry. Sessions are whole, which is what matters for intraday strategies; 15m
and 1h are close to gap-free.

Files are raw candles, **not back-adjusted**. Contracts trade at different
levels, so prices jump between sessions taken from different contracts — never
inside a session. The `source` column names the contract behind each bar.

```
data/tradingview/CME_MINI_NQ1_/5m.csv          one series, one timeframe
data/tradingview/CME_MINI_NQZ2024/5m.csv
data/tradingview/stitched/CME_MINI_NQ/5m.csv   one bar per timestamp
```

Any quarterly future works: `evotrader tv-archive --root ES`,
`--exchange COMEX --root GC`.

---

## The MCP tools

| tool | what Claude gets |
|---|---|
| `quote` | price, change, OHLC, volume, 52-week range, performance, TradingView's rating, for up to 12 tickers |
| `technicals` | TradingView's Technicals gauge on any timeframe: summary / MA / oscillator ratings, RSI, Stoch, CCI, ADX, MACD, EMAs and SMAs 10–200, Ichimoku, ATR, Bollinger, pivots |
| `bars` | OHLCV candles as CSV |
| `screener` | gainers, losers, most active, strong buy/sell, oversold, overbought, new 52-week highs and lows, across US stocks, crypto, futures, forex and more |
| `search_symbol` | name → `EXCHANGE:SYMBOL` |
| `archive_status` | what the batch puller has stored locally |
| run tools | `list_runs`, `run_report`, `leaderboard`, `inspect_genome`, … for the evolution engine in this repo |

Tickers are `EXCHANGE:SYMBOL` (`NASDAQ:AAPL`, `CME_MINI:NQ1!`,
`BINANCE:BTCUSDT`, `FX:EURUSD`, `TVC:DXY`), or a bare symbol that gets resolved
through TradingView search.

Everything is read-only. Nothing places an order.

---

## Where the data comes from

Three public TradingView endpoints, none needing an account:

- `scanner.tradingview.com/{market}/scan` — quotes, indicator values, ratings, screening
- `symbol-search.tradingview.com` — symbol lookup
- `data.tradingview.com` chart websocket — historical candles (the client here is
  ~80 lines of standard library)

Unauthenticated data carries TradingView's usual delays, about 10 minutes on CME
futures. TradingView's terms forbid automated extraction, so keep the pace
polite — the puller pauses a second between requests — and understand that using
a logged-in session ties the activity to your account.

---

## Security

A tunnel URL is the public internet. `TUNNEL-GUIDE.md` has the full list, but
the short version: anyone holding the URL can call every tool, so set a bearer
token (`evotrader serve-http --token …`), bind to `127.0.0.1`, and keep the
`cloudflared` credentials file out of the repo.
