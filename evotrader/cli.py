"""Command line interface.

    evotrader run --population 100 --generations 1000
    evotrader resume run-20260101-120000-ab12
    evotrader report --html reports/run.html
    evotrader inspect <genome-id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from .config import DEFAULT_SYMBOLS, EvolutionConfig
from .data import load_universe
from .evolution import Evolution, score_genome
from .features import build_features
from .fitness import FitnessConfig
from .llm import PRICING, Claude
from .report import html_report, lineage, markdown_report, print_report
from .store import Store


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON config file; flags below override it")
    p.add_argument("--symbols", help="comma separated tickers "
                                     f"(default: {','.join(DEFAULT_SYMBOLS)})")
    p.add_argument("--start", help="first date, YYYY-MM-DD")
    p.add_argument("--end", help="last date, YYYY-MM-DD")
    p.add_argument("--population", type=int, help="agents per generation")
    p.add_argument("--generations", type=int, help="number of generations to run")
    p.add_argument("--elites", type=int, help="survivors carried over and bred from")
    p.add_argument("--survivor-reports", type=int,
                   help="how many top agents Claude reads in detail")
    p.add_argument("--breeder", choices=["hybrid", "llm", "mutation"],
                   help="who writes the offspring")
    p.add_argument("--llm-share", type=float,
                   help="fraction of each generation written by Claude (hybrid)")
    p.add_argument("--llm-every", type=int, help="call Claude every N generations")
    p.add_argument("--model", help="Claude model id for breeding")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--budget", type=float, dest="budget_usd",
                   help="stop calling Claude after this many USD")
    p.add_argument("--seed", type=int, help="RNG seed (0 = nondeterministic)")
    p.add_argument("--workers", type=int, help="backtest processes (0 = auto)")
    p.add_argument("--offline", action="store_true",
                   help="use synthetic data instead of fetching prices")
    p.add_argument("--refresh-data", action="store_true", help="re-download price data")
    p.add_argument("--db", dest="db_path", help="SQLite path (default runs/evotrader.sqlite)")
    p.add_argument("--test-frac", type=float, help="held-out tail fraction")
    p.add_argument("--note", help="free text stored with the run")
    p.add_argument("--quiet", action="store_true")


def _config_from_args(args: argparse.Namespace) -> EvolutionConfig:
    cfg = EvolutionConfig.load(args.config) if getattr(args, "config", None) else EvolutionConfig()
    for name in ("start", "end", "population", "generations", "elites", "breeder",
                 "llm_share", "llm_every", "model", "effort", "budget_usd", "seed",
                 "workers", "db_path", "test_frac", "note", "survivor_reports"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "symbols", None):
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if getattr(args, "offline", False):
        cfg.offline = True
    if getattr(args, "refresh_data", False):
        cfg.refresh_data = True
    if getattr(args, "quiet", False):
        cfg.verbose = False
    return cfg


def _estimate(cfg: EvolutionConfig) -> str:
    """A blunt up-front estimate, because 1000 generations is not free."""
    calls = cfg.generations // max(cfg.llm_every, 1) if cfg.breeder != "mutation" else 0
    price_in, price_out = PRICING.get(cfg.model, (5.0, 25.0))
    # Measured shape of one breeding call: a long briefing in, a batch of
    # genomes out.  Roughly 12k input tokens and 6k output tokens.
    per_call = (12_000 * price_in + 6_000 * price_out) / 1_000_000
    backtests = cfg.population * cfg.generations
    return (f"plan: {backtests:,} backtests, {calls:,} breeding calls "
            f"(~${calls * per_call:,.2f} at {cfg.model} list prices"
            f"{f', capped at ${cfg.budget_usd:.2f}' if cfg.budget_usd else ''})")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    cfg.validate()
    print(_estimate(cfg))
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    evolution = Evolution(cfg)
    evolution.run()
    print_report(evolution.store, evolution.run_id, limit=10)
    print(f"resume with: evotrader resume {evolution.run_id} --db {cfg.db_path}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    store = Store(args.db_path or EvolutionConfig().db_path)
    stored = store.run_config(args.run_id)
    if stored is None:
        print(f"unknown run {args.run_id!r}", file=sys.stderr)
        return 1
    cfg = EvolutionConfig.from_dict(stored)
    for name in ("generations", "breeder", "llm_share", "llm_every", "model",
                 "budget_usd", "workers"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "quiet", False):
        cfg.verbose = False
    evolution = Evolution(cfg, store=store)
    evolution.resume(args.run_id)
    evolution.run(args.generations)
    print_report(store, args.run_id, limit=10)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = Store(args.db_path or EvolutionConfig().db_path)
    run_id = args.run_id or (store.list_runs(1)[0]["id"] if store.list_runs(1) else None)
    if not run_id:
        print("no runs found", file=sys.stderr)
        return 1
    print_report(store, run_id, limit=args.limit)
    if args.html:
        os.makedirs(os.path.dirname(os.path.abspath(args.html)), exist_ok=True)
        with open(args.html, "w") as fh:
            fh.write(html_report(store, run_id, limit=args.limit))
        print(f"wrote {args.html}")
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w") as fh:
            fh.write(markdown_report(store, run_id, limit=args.limit))
        print(f"wrote {args.markdown}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = Store(args.db_path or EvolutionConfig().db_path)
    rows = store.list_runs(args.limit)
    if not rows:
        print("no runs yet")
        return 0
    for row in rows:
        cfg = json.loads(row["config"])
        print(f"{row['id']}  {row['status']:<11} gens={row['generations'] or 0:<5} "
              f"pop={cfg['population']:<4} breeder={cfg['breeder']:<8} {row['note'][:40]}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    store = Store(args.db_path or EvolutionConfig().db_path)
    genome = store.get_genome(args.genome_id)
    if genome is None:
        print(f"unknown genome {args.genome_id!r}", file=sys.stderr)
        return 1
    print(genome.describe())
    if genome.rationale:
        print(f"\nrationale: {genome.rationale}")
    rows = store.conn.execute(
        "SELECT * FROM evaluations WHERE genome_id=?", (args.genome_id,)).fetchall()
    for row in rows:
        print(f"\ngeneration {row['generation']}: score {row['score']:+.3f}")
        print("  " + json.dumps(json.loads(row["metrics"]), indent=2).replace("\n", "\n  "))
    trades = store.conn.execute(
        "SELECT * FROM trades WHERE genome_id=? ORDER BY seq LIMIT ?",
        (args.genome_id, args.trades)).fetchall()
    if trades:
        print(f"\nfirst {len(trades)} trades")
        for t in trades:
            print(f"  {t['symbol']:<5} {t['entry_date']} -> {t['exit_date']} "
                  f"({t['bars_held']:>3}b) {t['ret'] * 100:+6.1f}%  in: {t['entry_reason']}"
                  f"  out: {t['exit_reason']}")
    chain = lineage(store, args.genome_id)
    if len(chain) > 1:
        print("\nancestry (newest first)")
        for g in chain:
            print(f"  gen {g.generation:>4} {g.origin:<9} {g.name[:40]:<40} {g.id}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Re-run one stored genome over an arbitrary window."""
    store = Store(args.db_path or EvolutionConfig().db_path)
    genome = store.get_genome(args.genome_id)
    if genome is None:
        print(f"unknown genome {args.genome_id!r}", file=sys.stderr)
        return 1
    stored = store.run_config(
        store.conn.execute("SELECT run_id FROM genomes WHERE id=?",
                           (args.genome_id,)).fetchone()["run_id"]) or {}
    cfg = EvolutionConfig.from_dict(stored)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else cfg.symbols
    universe = load_universe(symbols, args.start or cfg.start, args.end or cfg.end,
                             offline=cfg.offline)
    features = build_features(universe)
    from .evolution import Context
    from .runner import buy_and_hold
    ctx = Context(train=universe, train_features=features,
                  train_benchmark=buy_and_hold(universe, features,
                                               starting_cash=cfg.starting_cash),
                  test=None, test_features=None, test_benchmark=None,
                  starting_cash=cfg.starting_cash, commission_bps=cfg.commission_bps,
                  slippage_bps=cfg.slippage_bps, fitness=cfg.fitness)
    outcome = score_genome(genome, ctx)
    if outcome.error:
        print(f"error: {outcome.error}", file=sys.stderr)
        return 1
    print(genome.describe())
    print(f"\nscore {outcome.score:+.3f} | {outcome.metrics.summary()}")
    print(json.dumps(outcome.metrics.to_dict(), indent=2))
    if outcome.journal and args.trades:
        print(f"\nfirst {args.trades} trades")
        for t in outcome.journal.trades[:args.trades]:
            print("  " + t.summary())
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in (args.symbols or ",".join(DEFAULT_SYMBOLS)).split(",")]
    universe = load_universe(symbols, args.start, args.end, refresh=True)
    a, b = universe.date_range()
    print(f"cached {len(universe.symbols)} symbols, {len(universe)} shared bars, {a}..{b}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment: data access, credentials, dependencies."""
    print(f"python            {sys.version.split()[0]}")
    try:
        import numpy
        print(f"numpy             {numpy.__version__}")
    except ImportError:
        print("numpy             MISSING (pip install numpy)")
    try:
        import anthropic
        print(f"anthropic sdk     {anthropic.__version__}")
    except ImportError:
        print("anthropic sdk     not installed — breeding falls back to mutation only")
    claude = Claude()
    print(f"claude            {'ready (' + claude.model + ')' if claude.available else claude.unavailable_reason}")
    try:
        universe = load_universe(["SPY"], "2024-01-01", "2024-12-31")
        print(f"market data       ok ({len(universe)} SPY bars in 2024)")
    except Exception as exc:  # noqa: BLE001 - this is the diagnostic
        print(f"market data       unavailable: {exc}\n"
              f"                  use --offline to run on synthetic prices")
    return 0


def cmd_serve_http(args: argparse.Namespace) -> int:
    """Serve the read-only MCP tools over Streamable HTTP."""
    from .mcp_server import serve
    if args.db_path:
        os.environ["EVOTRADER_DB"] = args.db_path
    token = args.token or os.environ.get("EVOTRADER_MCP_TOKEN", "")
    origins = tuple(o.strip() for o in (args.allow_origin or "").split(",") if o.strip())
    return serve(host=args.host, port=args.port, endpoint=args.endpoint,
                 token=token, allowed_origins=origins)


def cmd_tv_archive(args: argparse.Namespace) -> int:
    """Batch-pull TradingView candles into the on-disk archive."""
    from . import tvarchive
    if args.status:
        print(tvarchive.format_status(tvarchive.status(args.out), args.out))
        return 0
    split = lambda v: tuple(x.strip() for x in v.split(",") if x.strip())  # noqa: E731
    failures = tvarchive.run_archive(
        exchange=args.exchange, root_symbol=args.root, since_year=args.since,
        continuous_tfs=split(args.timeframes),
        contract_tfs=() if args.no_contracts else split(args.contract_timeframes),
        refresh_expired=args.refresh_expired, pause=args.pause, archive=args.out,
        log=lambda m: print(m, flush=True))
    print(tvarchive.format_status(tvarchive.status(args.out), args.out))
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evotrader",
        description="Evolve paper-trading agents; Claude breeds each generation "
                    "from the last one's winners.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start a new evolution run")
    _add_run_flags(run)
    run.add_argument("--dry-run", action="store_true",
                     help="print the resolved config and cost estimate, then stop")
    run.set_defaults(func=cmd_run)

    res = sub.add_parser("resume", help="continue a run from its last checkpoint")
    res.add_argument("run_id")
    res.add_argument("--generations", type=int, help="how many more generations")
    res.add_argument("--breeder", choices=["hybrid", "llm", "mutation"])
    res.add_argument("--llm-share", type=float)
    res.add_argument("--llm-every", type=int)
    res.add_argument("--model")
    res.add_argument("--budget", type=float, dest="budget_usd")
    res.add_argument("--workers", type=int)
    res.add_argument("--db", dest="db_path")
    res.add_argument("--quiet", action="store_true")
    res.set_defaults(func=cmd_resume)

    rep = sub.add_parser("report", help="summarise a run")
    rep.add_argument("run_id", nargs="?", help="defaults to the most recent run")
    rep.add_argument("--db", dest="db_path")
    rep.add_argument("--limit", type=int, default=10)
    rep.add_argument("--html", help="also write a standalone HTML report here")
    rep.add_argument("--markdown", help="also write a Markdown report here")
    rep.set_defaults(func=cmd_report)

    runs = sub.add_parser("runs", help="list runs in the database")
    runs.add_argument("--db", dest="db_path")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=cmd_runs)

    ins = sub.add_parser("inspect", help="show one genome, its results and ancestry")
    ins.add_argument("genome_id")
    ins.add_argument("--db", dest="db_path")
    ins.add_argument("--trades", type=int, default=20)
    ins.set_defaults(func=cmd_inspect)

    bt = sub.add_parser("backtest", help="re-run a stored genome over any window")
    bt.add_argument("genome_id")
    bt.add_argument("--db", dest="db_path")
    bt.add_argument("--symbols")
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--trades", type=int, default=10)
    bt.set_defaults(func=cmd_backtest)

    fetch = sub.add_parser("fetch", help="download and cache price data")
    fetch.add_argument("--symbols")
    fetch.add_argument("--start", default="2005-01-01")
    fetch.add_argument("--end", default="2030-01-01")
    fetch.set_defaults(func=cmd_fetch)

    srv = sub.add_parser("serve-http",
                         help="serve read-only MCP tools over Streamable HTTP")
    srv.add_argument("--host", default="127.0.0.1",
                     help="bind address; keep 127.0.0.1 and put a tunnel in front")
    srv.add_argument("--port", type=int, default=8787)
    srv.add_argument("--endpoint", default="/mcp",
                     help="request path; a random path can act as a shared secret")
    srv.add_argument("--token", default="",
                     help="require 'Authorization: Bearer <token>' (or $EVOTRADER_MCP_TOKEN)")
    srv.add_argument("--allow-origin", default="",
                     help="comma-separated Origin allowlist; empty allows any")
    srv.add_argument("--db-path", default="")
    srv.set_defaults(func=cmd_serve_http)

    arc = sub.add_parser("tv-archive",
                         help="batch-pull TradingView candles (continuous + every "
                              "quarterly contract) into a growing CSV archive")
    arc.add_argument("--exchange", default="CME_MINI")
    arc.add_argument("--root", default="NQ", help="futures root symbol")
    arc.add_argument("--since", type=int, default=2015, help="first contract year")
    arc.add_argument("--timeframes", default="1m,5m,15m,30m,1h,4h,1D",
                     help="timeframes for the continuous series")
    arc.add_argument("--contract-timeframes", default="1m,5m,15m,1h",
                     help="timeframes for each quarterly contract")
    arc.add_argument("--no-contracts", action="store_true",
                     help="only refresh the continuous series (for the daily job)")
    arc.add_argument("--refresh-expired", action="store_true",
                     help="re-pull contracts that are already stored and expired")
    arc.add_argument("--pause", type=float, default=1.0, help="seconds between pulls")
    arc.add_argument("--out", default="", help="archive directory")
    arc.add_argument("--status", action="store_true", help="show what is stored and stop")
    arc.set_defaults(func=cmd_tv_archive)

    doc = sub.add_parser("doctor", help="check data access, credentials and deps")
    doc.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
