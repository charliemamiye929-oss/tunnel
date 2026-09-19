import numpy as np

from evotrader.data import Bars, Universe
from evotrader.features import build_features
from evotrader.genome import Genome, compile_genome
from evotrader.runner import backtest_genome, buy_and_hold


def _ramp_universe(n=400):
    """A deterministic series with a known shape, for exact assertions."""
    close = np.linspace(100.0, 200.0, n)
    dates = [f"2020-{1 + i // 28 % 12:02d}-{1 + i % 28:02d}-{i:04d}" for i in range(n)]
    dates = sorted(set(dates))[:n]
    bars = Bars("AAA", dates, open=close * 0.99, high=close * 1.01,
                low=close * 0.98, close=close, volume=np.full(n, 1e6))
    return Universe({"AAA": bars}, list(dates))


def test_orders_fill_at_the_next_bar_open_not_the_signal_bar_close():
    universe = _ramp_universe()
    features = build_features(universe)
    genome = Genome.from_dict({
        "name": "always in",
        "entry_rules": [{"when": "close > 0", "weight": 1.0}],
        "exit_rules": ["close < 0"],
        "risk": {"max_position_pct": 1.0, "max_positions": 1, "stop_loss_pct": 0.0},
    })
    result = backtest_genome(genome, universe, features)
    trade = result.journal.trades[0]
    fill_bar = universe.bars["AAA"].dates.index(trade.entry_date)
    expected_open = universe.bars["AAA"].open[fill_bar]
    # Entry price is the next bar's open plus slippage, never the signal close.
    assert expected_open <= trade.entry_price <= expected_open * 1.001
    assert trade.entry_date > universe.bars["AAA"].dates[features.warmup]


def test_stop_loss_closes_a_losing_position():
    n = 400
    close = np.concatenate([np.linspace(100, 130, n // 2), np.linspace(130, 60, n - n // 2)])
    dates = [f"d{i:04d}" for i in range(n)]
    bars = Bars("AAA", dates, close * 0.99, close * 1.01, close * 0.98, close,
                np.full(n, 1e6))
    universe = Universe({"AAA": bars}, dates)
    features = build_features(universe)
    genome = Genome.from_dict({
        "name": "stopped out",
        "entry_rules": [{"when": "close > 0", "weight": 1.0}],
        "exit_rules": ["close < 0"],
        "risk": {"max_position_pct": 1.0, "stop_loss_pct": 0.05, "cooldown_bars": 500},
    })
    result = backtest_genome(genome, universe, features)
    assert any("stop loss" in t.exit_reason for t in result.journal.trades)


def test_a_never_firing_genome_makes_no_trades_and_keeps_its_cash():
    universe = _ramp_universe()
    features = build_features(universe)
    genome = Genome.from_dict({
        "name": "never",
        "entry_rules": [{"when": "rsi14 < 1", "weight": 0.5}],
        "exit_rules": ["rsi14 > 99"],
    })
    result = backtest_genome(genome, universe, features)
    assert result.journal.trades == []
    assert abs(result.final_equity - result.starting_cash) < 1e-6


def test_max_positions_limits_concurrent_holdings():
    universe = _ramp_universe()
    extra = {}
    for name in ("AAA", "BBB", "CCC"):
        base = universe.bars["AAA"]
        extra[name] = Bars(name, base.dates, base.open, base.high, base.low,
                           base.close, base.volume)
    universe = Universe(extra, universe.calendar)
    features = build_features(universe)
    genome = Genome.from_dict({
        "name": "greedy",
        "entry_rules": [{"when": "close > 0", "weight": 0.5}],
        "exit_rules": ["close < 0"],
        "risk": {"max_position_pct": 0.5, "max_positions": 2},
    })
    compiled = compile_genome(genome)
    from evotrader.runner import run_backtest
    result = run_backtest(compiled, universe, features)
    symbols = {t.symbol for t in result.journal.trades}
    assert len(symbols) <= 2
    assert result.journal.rejected_entries > 0


def test_buy_and_hold_benchmark_grows_with_the_market():
    universe = _ramp_universe()
    features = build_features(universe)
    curve = buy_and_hold(universe, features, starting_cash=1000.0)
    assert curve[-1] > curve[0] > 0
