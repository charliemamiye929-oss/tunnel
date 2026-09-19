import json
import random

from evotrader.config import EvolutionConfig
from evotrader.fitness import Evaluation, Metrics
from evotrader.journal import Trade
from evotrader.population import seed_population
from evotrader.report import html_report, lineage, markdown_report, sparkline
from evotrader.store import Store


def _store(tmp_path):
    store = Store(str(tmp_path / "s.sqlite"))
    store.create_run("r1", EvolutionConfig(population=8).to_dict(), "note")
    return store


def test_genomes_and_evaluations_round_trip(tmp_path):
    store = _store(tmp_path)
    pop = seed_population(4, random.Random(0))
    store.save_genomes("r1", pop)
    evals = [Evaluation(g.id, g.name, 0, 1.0 - i, Metrics(sharpe=1.0, trades=20))
             for i, g in enumerate(pop)]
    store.save_evaluations("r1", 0, evals)
    board = store.leaderboard("r1", limit=10)
    assert board[0]["genome_id"] == pop[0].id
    again = store.get_genome(pop[0].id)
    assert again.fingerprint() == pop[0].fingerprint()


def test_checkpoint_round_trip_and_pruning(tmp_path):
    store = _store(tmp_path)
    pop = seed_population(3, random.Random(1))
    rng = random.Random(5)
    for gen in range(6):
        store.save_checkpoint("r1", gen, pop, ["lesson"], rng.getstate())
    latest = store.latest_checkpoint("r1")
    assert latest["generation"] == 5
    assert len(latest["population"]) == 3
    assert latest["lessons"] == ["lesson"]
    kept = store.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE run_id='r1'").fetchone()[0]
    assert kept <= 4          # old checkpoints are pruned


def test_trades_are_stored_for_elites(tmp_path):
    store = _store(tmp_path)
    pop = seed_population(1, random.Random(2))
    store.save_genomes("r1", pop)
    store.save_trades("r1", 0, pop[0].id, [
        Trade("SPY", "d0", "d1", 100.0, 110.0, 5.0, 50.0, 0.1, 12, "in", "out")])
    rows = store.conn.execute("SELECT * FROM trades WHERE genome_id=?", (pop[0].id,)).fetchall()
    assert rows[0]["symbol"] == "SPY" and abs(rows[0]["ret"] - 0.1) < 1e-9


def test_lineage_walks_back_through_parents(tmp_path):
    store = _store(tmp_path)
    root = seed_population(1, random.Random(3))[0]
    child = root.copy(generation=1, parents=[root.id])
    grandchild = child.copy(generation=2, parents=[child.id])
    store.save_genomes("r1", [root, child, grandchild])
    chain = lineage(store, grandchild.id)
    assert [g.id for g in chain] == [grandchild.id, child.id, root.id]


def test_reports_render(tmp_path):
    store = _store(tmp_path)
    pop = seed_population(3, random.Random(4))
    store.save_genomes("r1", pop)
    store.save_evaluations("r1", 0, [
        Evaluation(pop[0].id, pop[0].name, 0, 1.5, Metrics(sharpe=1.2, trades=30),
                   test_metrics=Metrics(sharpe=0.8, trades=9), test_score=0.6)])
    store.save_generation("r1", 0, best_score=1.5, mean_score=0.2, median_score=0.1,
                          best_genome_id=pop[0].id, analysis="it bought dips",
                          lessons=["dips work"], cost_usd=0.05, elapsed_s=1.0)
    store.save_generation("r1", 1, best_score=1.9, mean_score=0.4, median_score=0.3,
                          best_genome_id=pop[0].id)
    html = html_report(store, "r1")
    assert "evotrader run r1" in html and "<svg" in html and "it bought dips" in html
    md = markdown_report(store, "r1")
    assert "## Champion genome" in md and "dips work" in md


def test_sparkline_shapes_values():
    assert sparkline([1, 2, 3])[0] == "▁"
    assert sparkline([1, 2, 3])[-1] == "█"
    assert sparkline([1]) == ""
