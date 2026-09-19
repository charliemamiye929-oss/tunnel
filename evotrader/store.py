"""SQLite persistence: every genome, score, reflection and checkpoint.

A thousand generations of a hundred agents is a hundred thousand backtests; the
value is in the record of them.  Everything needed to resume a run, or to trace
a champion's ancestry back to generation 0, lives in this database.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .fitness import Evaluation
from .genome import Genome

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL,
    updated_at REAL,
    status TEXT,
    config TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS generations (
    run_id TEXT, generation INTEGER,
    best_score REAL, mean_score REAL, median_score REAL,
    best_genome_id TEXT, analysis TEXT, lessons TEXT,
    cost_usd REAL, elapsed_s REAL, created_at REAL,
    PRIMARY KEY (run_id, generation)
);
CREATE TABLE IF NOT EXISTS genomes (
    id TEXT PRIMARY KEY, run_id TEXT, generation INTEGER,
    name TEXT, origin TEXT, fingerprint TEXT, parents TEXT, body TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    run_id TEXT, generation INTEGER, genome_id TEXT,
    score REAL, test_score REAL, metrics TEXT, test_metrics TEXT, error TEXT,
    PRIMARY KEY (run_id, generation, genome_id)
);
CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT, generation INTEGER, genome_id TEXT, seq INTEGER,
    symbol TEXT, entry_date TEXT, exit_date TEXT, ret REAL, pnl REAL,
    bars_held INTEGER, entry_reason TEXT, exit_reason TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT, generation INTEGER, population TEXT, lessons TEXT,
    rng_state TEXT, usage TEXT, created_at REAL,
    PRIMARY KEY (run_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_genomes_run ON genomes(run_id, generation);
CREATE INDEX IF NOT EXISTS idx_evals_run ON evaluations(run_id, generation, score);
CREATE INDEX IF NOT EXISTS idx_trades_genome ON trades(run_id, genome_id);
"""


class Store:
    """Thin SQLite wrapper.  Safe to open the same file from several readers."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- runs
    def create_run(self, run_id: str, config: Dict[str, Any], note: str = "") -> None:
        now = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (id, created_at, updated_at, status, config, note)"
            " VALUES (?, COALESCE((SELECT created_at FROM runs WHERE id=?), ?), ?, ?, ?, ?)",
            (run_id, run_id, now, now, "running", json.dumps(config), note))
        self.conn.commit()

    def set_status(self, run_id: str, status: str) -> None:
        self.conn.execute("UPDATE runs SET status=?, updated_at=? WHERE id=?",
                          (status, time.time(), run_id))
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,))
        return cur.fetchone()

    def list_runs(self, limit: int = 25) -> List[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM generations g WHERE g.run_id=r.id)"
            " AS generations FROM runs r ORDER BY created_at DESC LIMIT ?", (limit,))
        return cur.fetchall()

    # ------------------------------------------------------------- genomes
    def save_genomes(self, run_id: str, genomes: Sequence[Genome]) -> None:
        rows = [(g.id, run_id, g.generation, g.name, g.origin, g.fingerprint(),
                 json.dumps(g.parents), json.dumps(g.to_dict())) for g in genomes]
        self.conn.executemany(
            "INSERT OR REPLACE INTO genomes (id, run_id, generation, name, origin,"
            " fingerprint, parents, body) VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def get_genome(self, genome_id: str) -> Optional[Genome]:
        cur = self.conn.execute("SELECT body FROM genomes WHERE id=?", (genome_id,))
        row = cur.fetchone()
        return Genome.from_dict(json.loads(row["body"])) if row else None

    # --------------------------------------------------------- evaluations
    def save_evaluations(self, run_id: str, generation: int,
                         evals: Sequence[Evaluation]) -> None:
        rows = [(run_id, generation, e.genome_id, e.score,
                 e.test_score, json.dumps(e.metrics.to_dict()),
                 json.dumps(e.test_metrics.to_dict()) if e.test_metrics else None,
                 e.error) for e in evals]
        self.conn.executemany(
            "INSERT OR REPLACE INTO evaluations (run_id, generation, genome_id, score,"
            " test_score, metrics, test_metrics, error) VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def save_trades(self, run_id: str, generation: int, genome_id: str,
                    trades: Iterable) -> None:
        rows = [(run_id, generation, genome_id, i, t.symbol, t.entry_date, t.exit_date,
                 t.ret, t.pnl, t.bars_held, t.entry_reason, t.exit_reason)
                for i, t in enumerate(trades)]
        if rows:
            self.conn.executemany(
                "INSERT INTO trades (run_id, generation, genome_id, seq, symbol,"
                " entry_date, exit_date, ret, pnl, bars_held, entry_reason, exit_reason)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    # -------------------------------------------------------- generations
    def save_generation(self, run_id: str, generation: int, *, best_score: float,
                        mean_score: float, median_score: float, best_genome_id: str,
                        analysis: str = "", lessons: Sequence[str] = (),
                        cost_usd: float = 0.0, elapsed_s: float = 0.0) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO generations (run_id, generation, best_score,"
            " mean_score, median_score, best_genome_id, analysis, lessons, cost_usd,"
            " elapsed_s, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, generation, best_score, mean_score, median_score, best_genome_id,
             analysis, json.dumps(list(lessons)), cost_usd, elapsed_s, time.time()))
        self.conn.execute("UPDATE runs SET updated_at=? WHERE id=?", (time.time(), run_id))
        self.conn.commit()

    def generation_history(self, run_id: str, limit: int = 0) -> List[Dict[str, Any]]:
        sql = ("SELECT g.generation, g.best_score, g.mean_score, g.best_genome_id,"
               " (SELECT name FROM genomes n WHERE n.id=g.best_genome_id) AS best_name"
               " FROM generations g WHERE run_id=? ORDER BY generation")
        rows = self.conn.execute(sql, (run_id,)).fetchall()
        out = [dict(r) for r in rows]
        return out[-limit:] if limit else out

    def leaderboard(self, run_id: str, limit: int = 20, *,
                    by_test: bool = False) -> List[Dict[str, Any]]:
        order = "e.test_score" if by_test else "e.score"
        # One row per distinct strategy: an elite carried across generations is
        # the same agent with a new id, and should not fill the board.
        sql = (f"SELECT e.*, n.name, n.origin, MIN(n.generation) AS gen FROM evaluations e"
               f" JOIN genomes n ON n.id = e.genome_id WHERE e.run_id=? AND e.error=''"
               f" AND {order} IS NOT NULL GROUP BY n.fingerprint"
               f" ORDER BY {order} DESC LIMIT ?")
        return [dict(r) for r in self.conn.execute(sql, (run_id, limit)).fetchall()]

    def generation_reflections(self, run_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        sql = ("SELECT generation, analysis, lessons, cost_usd FROM generations"
               " WHERE run_id=? AND analysis != '' ORDER BY generation DESC LIMIT ?")
        return [dict(r) for r in self.conn.execute(sql, (run_id, limit)).fetchall()]

    # -------------------------------------------------------- checkpoints
    def save_checkpoint(self, run_id: str, generation: int, population: Sequence[Genome],
                        lessons: Sequence[str], rng_state: Any,
                        usage: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints (run_id, generation, population, lessons,"
            " rng_state, usage, created_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, generation, json.dumps([g.to_dict() for g in population]),
             json.dumps(list(lessons)), json.dumps(rng_state), json.dumps(usage or {}),
             time.time()))
        self.conn.execute("DELETE FROM checkpoints WHERE run_id=? AND generation < ?",
                          (run_id, generation - 3))
        self.conn.commit()

    def latest_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY generation DESC LIMIT 1",
            (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "generation": row["generation"],
            "population": [Genome.from_dict(d) for d in json.loads(row["population"])],
            "lessons": json.loads(row["lessons"]),
            "rng_state": json.loads(row["rng_state"]),
            "usage": json.loads(row["usage"] or "{}"),
        }

    def run_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.get_run(run_id)
        return json.loads(row["config"]) if row else None
