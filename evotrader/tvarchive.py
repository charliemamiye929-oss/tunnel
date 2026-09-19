"""Batch-pull TradingView candles and keep them in a growing on-disk archive.

Without a login TradingView serves only the newest ~5,000-10,000 bars of any
one series (about 4 weeks of NQ 5m).  Two things get past that:

* every expired quarterly contract (NQH2016, NQM2016, ...) is its own series
  with its own ~5,300-bar window ending at its expiry, so pulling all of them
  recovers roughly 19 sessions of 5m per quarter back to 2015;
* re-running the pull regularly merges each new window into the stored file,
  so the continuous series (NQ1!) grows without gaps from now on.

Layout (one CSV per series and timeframe, unix-second ``ts``, never truncated):

    data/tradingview/CME_MINI_NQ1_/5m.csv
    data/tradingview/CME_MINI_NQZ2024/5m.csv
    data/tradingview/stitched/CME_MINI_NQ/5m.csv   <- one bar per timestamp

The stitched file picks one source per trading session - the series that
traded the most volume that day, which is the front month - so a day never
mixes two contracts' prices.  Different contracts trade at different levels
(the roll basis), so jumps appear between sessions from different contracts,
never inside one; the ``source`` column says which contract each bar came from.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from . import tradingview as tv

NY = ZoneInfo("America/New_York")
ARCHIVE_DIR = os.environ.get("EVOTRADER_TV_ARCHIVE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tradingview")

QUARTER_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
CONTINUOUS_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1D")
CONTRACT_TIMEFRAMES = ("1m", "5m", "15m", "1h")
FIELDS = ["ts", "open", "high", "low", "close", "volume"]

Row = Tuple[float, float, float, float, float]  # open, high, low, close, volume


def _safe(ticker: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ticker)


def series_path(ticker: str, timeframe: str, root: str = "") -> str:
    return os.path.join(root or ARCHIVE_DIR, _safe(ticker), f"{timeframe}.csv")


def read_series(path: str) -> Dict[int, Row]:
    rows: Dict[int, Row] = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["ts"])] = (float(r["open"]), float(r["high"]), float(r["low"]),
                                  float(r["close"]), float(r.get("volume") or 0))
    return rows


def write_series(path: str, rows: Dict[int, Row], extra: Optional[Dict[int, str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS + (["source"] if extra is not None else []))
        for ts in sorted(rows):
            o, h, l, c, v = rows[ts]
            line = [ts, f"{o:.10g}", f"{h:.10g}", f"{l:.10g}", f"{c:.10g}", f"{v:.0f}"]
            if extra is not None:
                line.append(extra.get(ts, ""))
            w.writerow(line)
    os.replace(tmp, path)


def pull(ticker: str, timeframe: str, root: str = "") -> Tuple[int, int]:
    """Fetch everything TradingView serves for one series and merge it in.

    Returns (bars fetched, bars now stored).  The newest stored bar may have
    been captured while still forming, so fetched bars always overwrite.
    """
    path = series_path(ticker, timeframe, root)
    rows = read_series(path)
    fetched = tv.fetch_bars(ticker, timeframe, 20_000, timeout=90)
    for t, o, h, l, c, v in fetched:
        rows[int(t)] = (float(o), float(h), float(l), float(c), float(v or 0))
    if rows:
        write_series(path, rows)
    return len(fetched), len(rows)


# --------------------------------------------------------------------------
# quarterly contracts
# --------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def contracts(root: str, since_year: int, today: Optional[date] = None,
              ahead_days: int = 200) -> List[Tuple[str, date]]:
    """Quarterly contract codes from ``since_year`` up to those expiring within
    ``ahead_days`` (further-out months barely trade), oldest first."""
    today = today or date.today()
    out = []
    for year in range(since_year, today.year + 2):
        for code, month in QUARTER_CODES.items():
            expiry = _third_friday(year, month)
            if expiry <= today + timedelta(days=ahead_days):
                out.append((f"{root}{code}{year}", expiry))
    return out


# --------------------------------------------------------------------------
# stitching
# --------------------------------------------------------------------------

def session_date(ts: int) -> date:
    """CME equity futures trade 18:00-17:00 ET; bars after 18:00 belong to the
    next day's session."""
    return (datetime.fromtimestamp(ts, NY) + timedelta(hours=6)).date()


def stitch(sources: Dict[str, Dict[int, Row]],
           prefer: Sequence[str] = ()) -> Tuple[Dict[int, Row], Dict[int, str]]:
    """One source per session: the one with the most volume that session, with
    ``prefer`` winning ties (the continuous series equals the front month)."""
    by_session: Dict[date, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for name, rows in sources.items():
        for ts in rows:
            by_session[session_date(ts)][name].append(ts)
    rank = {name: i for i, name in enumerate(prefer)}
    out: Dict[int, Row] = {}
    label: Dict[int, str] = {}
    for _, candidates in sorted(by_session.items()):
        def key(name: str) -> Tuple[float, int, int]:
            stamps = candidates[name]
            vol = sum(sources[name][t][4] for t in stamps)
            return (vol, len(stamps), -rank.get(name, len(rank)))
        best = max(candidates, key=key)
        for ts in candidates[best]:
            out[ts] = sources[best][ts]
            label[ts] = best
    return out, label


def build_stitched(exchange: str, root_symbol: str, timeframe: str,
                   archive: str = "") -> Tuple[str, int, int]:
    base = archive or ARCHIVE_DIR
    continuous = f"{exchange}:{root_symbol}1!"
    sources: Dict[str, Dict[int, Row]] = {}
    prefix = _safe(f"{exchange}:{root_symbol}")
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if not name.startswith(prefix):
                continue
            rows = read_series(os.path.join(base, name, f"{timeframe}.csv"))
            if rows:
                ticker = f"{exchange}:{name[len(_safe(exchange)) + 1:]}"
                sources[continuous if name == _safe(continuous) else ticker] = rows
    rows, label = stitch(sources, prefer=[continuous])
    path = os.path.join(base, "stitched", _safe(f"{exchange}:{root_symbol}"), f"{timeframe}.csv")
    if rows:
        write_series(path, rows, label)
    return path, len(rows), len({session_date(t) for t in rows})


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def status(archive: str = "") -> List[Dict[str, object]]:
    base = archive or ARCHIVE_DIR
    out: List[Dict[str, object]] = []
    if not os.path.isdir(base):
        return out
    for dirpath, _, files in sorted(os.walk(base)):
        for f in sorted(files):
            if not f.endswith(".csv"):
                continue
            path = os.path.join(dirpath, f)
            with open(path, newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                first = last = None
                n = 0
                for row in reader:
                    ts = int(row[0])
                    first = ts if first is None else first
                    last = ts
                    n += 1
            if not n:
                continue
            out.append({"series": os.path.relpath(dirpath, base), "timeframe": f[:-4],
                        "bars": n, "first": first, "last": last, "path": path})
    return out


def format_status(rows: List[Dict[str, object]], archive: str = "") -> str:
    if not rows:
        return f"Archive at {archive or ARCHIVE_DIR} is empty - run `evotrader tv-archive`."
    fmt = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    lines = [f"TradingView archive: {archive or ARCHIVE_DIR}",
             f"{'series':<34}{'tf':>5}{'bars':>10}  first       last"]
    stitched = [r for r in rows if str(r["series"]).startswith("stitched")]
    singles = [r for r in rows if r not in stitched]
    for r in stitched + singles:
        lines.append(f"{str(r['series']):<34}{str(r['timeframe']):>5}{r['bars']:>10,}  "
                     f"{fmt(r['first'])}  {fmt(r['last'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the batch
# --------------------------------------------------------------------------

def run_archive(exchange: str = "CME_MINI", root_symbol: str = "NQ", since_year: int = 2015,
                continuous_tfs: Sequence[str] = CONTINUOUS_TIMEFRAMES,
                contract_tfs: Sequence[str] = CONTRACT_TIMEFRAMES,
                refresh_expired: bool = False, pause: float = 1.0,
                archive: str = "", log=print) -> int:
    """Pull the continuous series and every quarterly contract, then stitch.

    Expired contracts never change, so once stored they are skipped unless
    ``refresh_expired``; live and upcoming contracts are pulled every run.
    Returns the number of failed pulls.
    """
    failures = 0
    today = date.today()
    jobs: List[Tuple[str, str]] = [(f"{exchange}:{root_symbol}1!", tf) for tf in continuous_tfs]
    for code, expiry in contracts(root_symbol, since_year, today):
        ticker = f"{exchange}:{code}"
        for tf in contract_tfs:
            settled = expiry < today - timedelta(days=3)
            if settled and not refresh_expired and os.path.exists(series_path(ticker, tf, archive)):
                continue
            jobs.append((ticker, tf))
    log(f"{len(jobs)} pulls to run -> {archive or ARCHIVE_DIR}")
    for i, (ticker, tf) in enumerate(jobs, 1):
        for attempt in range(3):
            try:
                got, stored = pull(ticker, tf, archive)
                log(f"[{i}/{len(jobs)}] {ticker:<22} {tf:>3}  fetched {got:>6,}  stored {stored:>8,}")
                break
            except (tv.TVError, OSError) as exc:
                if "invalid symbol" in str(exc) or attempt == 2:
                    log(f"[{i}/{len(jobs)}] {ticker:<22} {tf:>3}  FAILED: {exc}")
                    failures += 1
                    break
                time.sleep(5 * (attempt + 1))
        time.sleep(pause)
    for tf in sorted(set(continuous_tfs) | set(contract_tfs), key=list(tv.TIMEFRAMES).index):
        path, bars, sessions = build_stitched(exchange, root_symbol, tf, archive)
        if bars:
            log(f"stitched {tf:>3}: {bars:>8,} bars over {sessions:,} sessions -> {path}")
    return failures
