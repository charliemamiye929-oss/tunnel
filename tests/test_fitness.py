import numpy as np

from evotrader.fitness import (Evaluation, FitnessConfig, Metrics, compute_metrics,
                               fitness_score, rank)
from evotrader.journal import Trade


def _trade(ret, pnl, bars=10):
    return Trade("AAA", "d0", "d1", 100.0, 100.0 * (1 + ret), 10.0, pnl, ret, bars,
                 "in", "out")


def test_metrics_on_a_simple_curve():
    equity = list(np.linspace(100_000, 150_000, 252))
    m = compute_metrics(equity, [_trade(0.1, 100.0), _trade(-0.05, -50.0)],
                        benchmark=[100_000, 120_000], turnover=4.0)
    assert abs(m.total_return - 0.5) < 1e-9
    assert m.trades == 2 and abs(m.win_rate - 0.5) < 1e-9
    assert abs(m.profit_factor - 2.0) < 1e-9
    assert abs(m.excess_return - 0.3) < 1e-9
    assert m.max_drawdown == 0.0
    assert m.turnover == 4.0     # one year of bars, so annualised == raw


def test_turnover_is_annualised():
    equity = list(np.linspace(100.0, 110.0, 504))    # two years
    m = compute_metrics(equity, [], turnover=10.0)
    assert abs(m.turnover - 5.0) < 1e-6


def test_drawdown_is_measured_from_the_peak():
    m = compute_metrics([100.0, 200.0, 100.0], [])
    assert abs(m.max_drawdown + 0.5) < 1e-9


def test_fitness_penalises_inactivity_and_deep_drawdowns():
    cfg = FitnessConfig()
    active = Metrics(sharpe=1.0, trades=50, years=1.0)
    idle = Metrics(sharpe=1.0, trades=0, years=1.0)
    drawn = Metrics(sharpe=1.0, trades=50, max_drawdown=-0.50, years=1.0)
    assert fitness_score(active, cfg) > fitness_score(idle, cfg)
    assert fitness_score(active, cfg) > fitness_score(drawn, cfg)


def test_fitness_rewards_beating_buy_and_hold():
    cfg = FitnessConfig()
    beat = Metrics(sharpe=1.0, trades=50, excess_return=0.30, years=1.0)
    lag = Metrics(sharpe=1.0, trades=50, excess_return=-0.30, years=1.0)
    assert fitness_score(beat, cfg) - fitness_score(lag, cfg) > 0.5


def test_fitness_ignores_nan_and_inf():
    assert fitness_score(Metrics(sharpe=float("nan"), trades=50, years=1.0)) == 0.0
    assert fitness_score(Metrics(sharpe=float("inf"), trades=50, years=1.0)) == 0.0


def test_rank_sorts_best_first_and_failures_last():
    good = Evaluation("a", "good", 0, 1.0, Metrics())
    bad = Evaluation("b", "bad", 0, 5.0, Metrics(), error="boom")
    mid = Evaluation("c", "mid", 0, 0.5, Metrics())
    assert [e.genome_id for e in rank([mid, bad, good])] == ["a", "c", "b"]
