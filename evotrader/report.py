"""Reporting: what happened in a run, in text, Markdown or a standalone page."""
from __future__ import annotations

import html
import json
import math
from typing import Any, Dict, List, Optional, Sequence

from .genome import Genome
from .store import Store

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float]) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return "".join(SPARK[min(int((v - lo) / span * (len(SPARK) - 1)), len(SPARK) - 1)]
                   for v in vals)


def run_summary(store: Store, run_id: str) -> Dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run {run_id!r}")
    history = store.generation_history(run_id)
    board = store.leaderboard(run_id, limit=15)
    board_oos = store.leaderboard(run_id, limit=15, by_test=True)
    return {
        "run_id": run_id,
        "status": run["status"],
        "note": run["note"],
        "config": json.loads(run["config"]),
        "generations": len(history),
        "history": history,
        "leaderboard": board,
        "leaderboard_oos": board_oos,
        "reflections": store.generation_reflections(run_id, limit=3),
        "cost_usd": sum(float(store.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM generations WHERE run_id=?",
            (run_id,)).fetchone()[0] or 0.0) for _ in [0]),
    }


def _fmt_metrics(raw: Optional[str]) -> str:
    if not raw:
        return "-"
    m = json.loads(raw)
    return (f"ret {m['total_return'] * 100:+.1f}% (bh {m['benchmark_return'] * 100:+.1f}%) "
            f"sharpe {m['sharpe']:.2f} mdd {m['max_drawdown'] * 100:.1f}% "
            f"trades {m['trades']} win {m['win_rate'] * 100:.0f}%")


def print_report(store: Store, run_id: str, *, limit: int = 10) -> None:
    s = run_summary(store, run_id)
    cfg = s["config"]
    print(f"\nrun {run_id}  [{s['status']}]  {s['note']}")
    print(f"  {cfg['population']} agents x {s['generations']} generations completed "
          f"| breeder={cfg['breeder']} model={cfg['model']} "
          f"| symbols {','.join(cfg['symbols'][:8])}"
          f"{'...' if len(cfg['symbols']) > 8 else ''}")
    print(f"  window {cfg['start']}..{cfg['end']} (held out last {cfg['test_frac']:.0%})"
          f" | LLM spend ${s['cost_usd']:.2f}")

    best = [h["best_score"] for h in s["history"]]
    mean = [h["mean_score"] for h in s["history"]]
    if best:
        print(f"\n  best score  {sparkline(best)}  {best[0]:+.2f} -> {best[-1]:+.2f}")
        print(f"  mean score  {sparkline(mean)}  {mean[0]:+.2f} -> {mean[-1]:+.2f}")

    print(f"\n  top agents by training fitness")
    for row in s["leaderboard"][:limit]:
        print(f"    {row['score']:+.3f}  gen {row['gen']:>4}  {row['name'][:30]:<30} "
              f"{_fmt_metrics(row['metrics'])}")
    if any(r.get("test_score") is not None for r in s["leaderboard_oos"]):
        print(f"\n  same agents on the held-out window (never used for selection)")
        for row in s["leaderboard_oos"][:limit]:
            if row.get("test_score") is None:
                continue
            print(f"    {row['test_score']:+.3f}  gen {row['gen']:>4}  {row['name'][:30]:<30} "
                  f"{_fmt_metrics(row['test_metrics'])}")

    champion = s["leaderboard"][0] if s["leaderboard"] else None
    if champion:
        genome = store.get_genome(champion["genome_id"])
        if genome:
            print("\n  champion genome")
            for line in genome.describe().split("\n"):
                print(f"    {line}")
            if genome.rationale:
                print(f"    rationale: {genome.rationale}")

    for ref in s["reflections"]:
        if not ref["analysis"]:
            continue
        print(f"\n  Claude's read after generation {ref['generation']}:")
        for line in ref["analysis"].strip().split("\n"):
            print(f"    {line}")
        lessons = json.loads(ref["lessons"] or "[]")
        for lesson in lessons:
            print(f"      - {lesson}")
        break
    print()


def lineage(store: Store, genome_id: str, *, max_depth: int = 40) -> List[Genome]:
    """Walk a genome's ancestry back towards generation 0."""
    chain: List[Genome] = []
    seen = set()
    current = store.get_genome(genome_id)
    while current is not None and current.id not in seen and len(chain) < max_depth:
        chain.append(current)
        seen.add(current.id)
        current = store.get_genome(current.parents[0]) if current.parents else None
    return chain


def markdown_report(store: Store, run_id: str, *, limit: int = 15) -> str:
    s = run_summary(store, run_id)
    cfg = s["config"]
    out = [f"# evotrader run `{run_id}`", "",
           f"* status: **{s['status']}**",
           f"* {cfg['population']} agents x {s['generations']} generations "
           f"(breeder `{cfg['breeder']}`, model `{cfg['model']}`)",
           f"* symbols: {', '.join(cfg['symbols'])}",
           f"* window: {cfg['start']} to {cfg['end']}, last {cfg['test_frac']:.0%} held out",
           f"* LLM spend: ${s['cost_usd']:.2f}", ""]
    out += ["## Best score by generation", "",
            "| generation | best | mean | champion |", "|---:|---:|---:|---|"]
    for h in s["history"][-40:]:
        out.append(f"| {h['generation']} | {h['best_score']:+.3f} | "
                   f"{h['mean_score']:+.3f} | {h.get('best_name', '')} |")
    out += ["", "## Leaderboard (training window)", "",
            "| score | gen | agent | result |", "|---:|---:|---|---|"]
    for row in s["leaderboard"][:limit]:
        out.append(f"| {row['score']:+.3f} | {row['gen']} | {row['name']} | "
                   f"{_fmt_metrics(row['metrics'])} |")
    if s["leaderboard"]:
        genome = store.get_genome(s["leaderboard"][0]["genome_id"])
        if genome:
            out += ["", "## Champion genome", "", "```", genome.describe(), "```"]
    for ref in s["reflections"]:
        if ref["analysis"]:
            out += ["", f"## Claude's analysis after generation {ref['generation']}", "",
                    ref["analysis"]]
            lessons = json.loads(ref["lessons"] or "[]")
            if lessons:
                out += [""] + [f"* {l}" for l in lessons]
            break
    return "\n".join(out) + "\n"


def _svg_chart(history: Sequence[Dict[str, Any]], width: int = 720,
               height: int = 240) -> str:
    """Best/mean fitness by generation as an inline SVG."""
    pts = [(h["generation"], h["best_score"], h["mean_score"]) for h in history
           if h["best_score"] is not None and math.isfinite(h["best_score"])]
    if len(pts) < 2:
        return "<p>Not enough generations to chart yet.</p>"
    xs = [p[0] for p in pts]
    ys = [v for p in pts for v in (p[1], p[2]) if v is not None and math.isfinite(v)]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 36

    def sx(x: float) -> float:
        return pad + (x - x0) / max(x1 - x0, 1e-9) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - y0) / max(y1 - y0, 1e-9) * (height - 2 * pad)

    def path(idx: int) -> str:
        vals = [(p[0], p[idx]) for p in pts
                if p[idx] is not None and math.isfinite(p[idx])]
        return " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                        for i, (x, y) in enumerate(vals))

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
  <rect x="0" y="0" width="{width}" height="{height}" fill="none"/>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#999"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#999"/>
  <path d="{path(1)}" fill="none" stroke="#1a7f37" stroke-width="2"/>
  <path d="{path(2)}" fill="none" stroke="#8250df" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="{pad}" y="{pad - 12}" font-size="12" fill="#1a7f37">best fitness</text>
  <text x="{pad + 90}" y="{pad - 12}" font-size="12" fill="#8250df">population mean</text>
  <text x="{pad}" y="{height - 10}" font-size="11" fill="#666">gen {x0}</text>
  <text x="{width - pad - 40}" y="{height - 10}" font-size="11" fill="#666">gen {x1}</text>
  <text x="4" y="{pad}" font-size="11" fill="#666">{y1:+.2f}</text>
  <text x="4" y="{height - pad}" font-size="11" fill="#666">{y0:+.2f}</text>
</svg>"""


def html_report(store: Store, run_id: str, *, limit: int = 20) -> str:
    s = run_summary(store, run_id)
    cfg = s["config"]
    e = html.escape
    rows = "".join(
        f"<tr><td class=n>{r['score']:+.3f}</td><td class=n>{r['gen']}</td>"
        f"<td>{e(r['name'])}</td><td>{e(r['origin'])}</td>"
        f"<td>{e(_fmt_metrics(r['metrics']))}</td>"
        f"<td>{e(_fmt_metrics(r['test_metrics']))}</td></tr>"
        for r in s["leaderboard"][:limit])
    champion_block = ""
    if s["leaderboard"]:
        genome = store.get_genome(s["leaderboard"][0]["genome_id"])
        if genome:
            champion_block = f"<h2>Champion genome</h2><pre>{e(genome.describe())}</pre>"
            if genome.rationale:
                champion_block += f"<p><em>{e(genome.rationale)}</em></p>"
    reflection = ""
    for ref in s["reflections"]:
        if ref["analysis"]:
            lessons = "".join(f"<li>{e(l)}</li>" for l in json.loads(ref["lessons"] or "[]"))
            reflection = (f"<h2>Claude's analysis after generation {ref['generation']}</h2>"
                          f"<pre class=analysis>{e(ref['analysis'])}</pre>"
                          f"<ul>{lessons}</ul>")
            break
    return f"""<!doctype html>
<meta charset="utf-8"><title>evotrader {e(run_id)}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:52rem;color:#1f2328}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border-bottom:1px solid #d0d7de;padding:.35rem .5rem;text-align:left}}
 td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
 pre{{background:#f6f8fa;padding:.75rem;border-radius:6px;overflow-x:auto;white-space:pre-wrap}}
 .meta{{color:#57606a}}
</style>
<h1>evotrader run {e(run_id)}</h1>
<p class=meta>{cfg['population']} agents &times; {s['generations']} generations &middot;
breeder {e(cfg['breeder'])} &middot; model {e(cfg['model'])} &middot;
{e(', '.join(cfg['symbols']))} &middot; {e(cfg['start'])} to {e(cfg['end'])}
(last {cfg['test_frac']:.0%} held out) &middot; LLM spend ${s['cost_usd']:.2f}</p>
{_svg_chart(s['history'])}
<h2>Leaderboard</h2>
<table><tr><th class=n>fitness</th><th class=n>gen</th><th>agent</th><th>origin</th>
<th>training window</th><th>held-out window</th></tr>{rows}</table>
{champion_block}
{reflection}
"""
