import numpy as np

from evotrader import indicators as ind


def test_sma_matches_manual_mean():
    x = np.arange(1.0, 11.0)
    out = ind.sma(x, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == 2.0
    assert out[-1] == 9.0


def test_rsi_bounds_and_extremes():
    up = np.arange(1.0, 60.0)
    r = ind.rsi(up, 14)
    assert np.nanmax(r) <= 100.0 and np.nanmin(r) >= 0.0
    assert r[-1] > 95.0            # a pure uptrend pins RSI high
    down = up[::-1].copy()
    assert ind.rsi(down, 14)[-1] < 5.0


def test_atr_is_positive_and_warms_up():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 200))
    a = ind.atr(close + 1, close - 1, close, 14)
    assert np.isnan(a[:14]).all()
    assert (a[14:] > 0).all()


def test_bollinger_percent_b_is_bounded_for_normal_data():
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    _, _, _, pct = ind.bollinger(close, 20, 2.0)
    valid = pct[~np.isnan(pct)]
    assert valid.min() > -1.0 and valid.max() < 2.0


def test_returns_and_drawdown():
    x = np.array([100.0, 110.0, 99.0])
    r = ind.returns(x, 1)
    assert np.isnan(r[0]) and abs(r[1] - 0.1) < 1e-12
    dd = ind.drawdown_series(x)
    assert dd[-1] < -0.09 and dd[0] == 0.0
