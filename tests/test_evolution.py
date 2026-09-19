"""End-to-end: the loop, persistence, resume, and the LLM breeding path."""
import os
import random

import pytest

from evotrader.config import EvolutionConfig
from evotrader.evolution import Evolution
from evotrader.genome import is_valid
from evotrader.llm import Response, Usage
from evotrader.store import Store
from tests.test_breeder import _payload


class FakeClaude:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.available = True
        self.unavailable_reason = ""
        self.usage = Usage()
        self.model = "fake"

    def budget_left(self):
        return float("inf")

    def complete(self, prompt, *, system="", schema=None, **kwargs):
        self.calls += 1
        return Response(text="", data=self.payload, model="fake", cost_usd=0.02)


def _cfg(tmp_path, **over):
    base = dict(symbols=["AAA", "BBB", "CCC"], start="2015-01-01", end="2025-12-31",
                offline=True, population=12, generations=3, elites=3,
                survivor_reports=3, breeder="mutation", seed=11, verbose=False,
                workers=1, validate_top=3, db_path=str(tmp_path / "run.sqlite"))
    base.update(over)
    return EvolutionConfig(**base)


def test_full_run_persists_everything(tmp_path):
    cfg = _cfg(tmp_path)
    evolution = Evolution(cfg)
    reports = evolution.run()
    assert len(reports) == 3
    assert all(r.best is not None for r in reports)

    store = evolution.store
    history = store.generation_history(evolution.run_id)
    assert [h["generation"] for h in history] == [0, 1, 2]
    assert store.get_run(evolution.run_id)["status"] == "finished"
    board = store.leaderboard(evolution.run_id, limit=5)
    assert board and board[0]["score"] >= board[-1]["score"]
    assert store.get_genome(board[0]["genome_id"]) is not None


def test_elitism_never_loses_the_best_score(tmp_path):
    evolution = Evolution(_cfg(tmp_path, generations=4, population=16))
    reports = evolution.run()
    bests = [r.best.score for r in reports]
    assert all(b >= a - 1e-9 for a, b in zip(bests, bests[1:])), bests


def test_population_size_is_stable_and_valid(tmp_path):
    evolution = Evolution(_cfg(tmp_path, generations=3, population=20))
    evolution.run()
    assert len(evolution.population) == 20
    assert all(is_valid(g) for g in evolution.population)


def test_held_out_scores_are_recorded_but_do_not_drive_selection(tmp_path):
    evolution = Evolution(_cfg(tmp_path, generations=2))
    reports = evolution.run()
    scored = [e for e in reports[-1].evaluations if e.test_metrics is not None]
    assert scored, "expected the leaders to be scored on the held-out window"
    ranked_by_train = [e.genome_id for e in reports[-1].evaluations]
    assert ranked_by_train[0] == reports[-1].best.genome_id


def test_resume_continues_from_the_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, generations=2)
    first = Evolution(cfg)
    first.run()
    run_id = first.run_id
    first.store.close()

    store = Store(cfg.db_path)
    second = Evolution(_cfg(tmp_path, generations=2, run_id=run_id), store=store)
    second.resume(run_id)
    assert second.generation == 2
    second.run(2)
    generations = [h["generation"] for h in store.generation_history(run_id)]
    assert generations == [0, 1, 2, 3]


def test_llm_breeding_path_end_to_end(tmp_path):
    claude = FakeClaude(_payload(6))
    cfg = _cfg(tmp_path, breeder="hybrid", llm_share=0.5, generations=2)
    evolution = Evolution(cfg, claude=claude)
    reports = evolution.run()
    assert claude.calls == 2
    assert reports[0].analysis.startswith("The winners")
    stored = evolution.store.generation_reflections(evolution.run_id)
    assert stored and stored[0]["analysis"]
    origins = {g.origin for g in evolution.population}
    assert "llm" in origins


def test_a_genome_that_cannot_run_is_recorded_not_fatal(tmp_path):
    from evotrader.genome import Genome
    evolution = Evolution(_cfg(tmp_path, generations=1))
    evolution.prepare()
    evolution.start()
    broken = Genome.from_dict({"name": "broken", "entry_rules": [{"when": "bogus > 1"}],
                               "exit_rules": ["rsi14 > 70"]})
    evolution.population[0] = broken
    report = evolution.step()
    errors = [e for e in report.evaluations if e.error]
    assert len(errors) == 1 and "invalid genome" in errors[0].error


def test_config_validation_rejects_nonsense():
    with pytest.raises(ValueError):
        EvolutionConfig(population=2).validate()
    with pytest.raises(ValueError):
        EvolutionConfig(population=10, elites=10).validate()
    with pytest.raises(ValueError):
        EvolutionConfig(breeder="telepathy").validate()


def test_config_survives_a_json_roundtrip(tmp_path):
    cfg = EvolutionConfig(symbols=["SPY"], population=8)
    cfg.fitness.drawdown_limit = 0.11
    path = str(tmp_path / "cfg.json")
    cfg.save(path)
    again = EvolutionConfig.load(path)
    assert again.symbols == ["SPY"] and again.population == 8
    assert again.fitness.drawdown_limit == 0.11
