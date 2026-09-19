"""Vectorised technical indicators over numpy price arrays.

Every function takes 1-D float arrays and returns a 1-D float array of the same
length, left-padded with NaN where the indicator is not yet defined.  Keeping
them vectorised matters: the evolution loop backtests thousands of genomes, and
features are computed once per symbol and reused by every genome.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sma", "ema", "rsi", "macd", "atr", "bollinger", "rolling_std",
    "rolling_max", "rolling_min", "zscore", "returns", "drawdown_series",
]


def _as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(values, window: int) -> np.ndarray:
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    if window <= 0 or window > v.size:
        return out
    csum = np.cumsum(np.insert(v, 0, 0.0))
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def ema(values, window: int) -> np.ndarray:
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    if window <= 0 or v.size == 0:
        return out
    alpha = 2.0 / (window + 1.0)
    # Seed with the simple average of the first `window` points so the series is
    # stable regardless of how much history precedes it.
    if v.size < window:
        return out
    acc = float(np.mean(v[:window]))
    out[window - 1] = acc
    for i in range(window, v.size):
        acc = alpha * v[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def rolling_std(values, window: int) -> np.ndarray:
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    if window <= 1 or window > v.size:
        return out
    c1 = np.cumsum(np.insert(v, 0, 0.0))
    c2 = np.cumsum(np.insert(v * v, 0, 0.0))
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    var = np.maximum((s2 - s1 * s1 / window) / (window - 1), 0.0)
    out[window - 1:] = np.sqrt(var)
    return out


def rolling_max(values, window: int) -> np.ndarray:
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    for i in range(window - 1, v.size):
        out[i] = np.max(v[i - window + 1:i + 1])
    return out


def rolling_min(values, window: int) -> np.ndarray:
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    for i in range(window - 1, v.size):
        out[i] = np.min(v[i - window + 1:i + 1])
    return out


def returns(values, periods: int = 1) -> np.ndarray:
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    if periods <= 0 or periods >= v.size:
        return out
    prev = v[:-periods]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[periods:] = np.where(prev != 0, v[periods:] / prev - 1.0, np.nan)
    return out


def rsi(values, window: int = 14) -> np.ndarray:
    """Wilder's RSI."""
    v = _as_float(values)
    out = np.full(v.shape, np.nan)
    if v.size <= window:
        return out
    delta = np.diff(v)
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)
    avg_gain = float(np.mean(gain[:window]))
    avg_loss = float(np.mean(loss[:window]))
    def _rsi(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else 50.0
        rs = g / l
        return 100.0 - 100.0 / (1.0 + rs)
    out[window] = _rsi(avg_gain, avg_loss)
    for i in range(window + 1, v.size):
        avg_gain = (avg_gain * (window - 1) + gain[i - 1]) / window
        avg_loss = (avg_loss * (window - 1) + loss[i - 1]) / window
        out[i] = _rsi(avg_gain, avg_loss)
    return out


def macd(values, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    v = _as_float(values)
    line = ema(v, fast) - ema(v, slow)
    valid = ~np.isnan(line)
    sig = np.full(v.shape, np.nan)
    if valid.any():
        idx = int(np.argmax(valid))
        sig_tail = ema(line[idx:], signal)
        sig[idx:] = sig_tail
    return line, sig, line - sig


def atr(high, low, close, window: int = 14) -> np.ndarray:
    """Average true range (Wilder smoothing)."""
    h, l, c = _as_float(high), _as_float(low), _as_float(close)
    out = np.full(c.shape, np.nan)
    if c.size <= window:
        return out
    prev_close = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    tr[0] = h[0] - l[0]
    acc = float(np.mean(tr[1:window + 1]))
    out[window] = acc
    for i in range(window + 1, c.size):
        acc = (acc * (window - 1) + tr[i]) / window
        out[i] = acc
    return out


def bollinger(values, window: int = 20, num_std: float = 2.0):
    """Return (upper, middle, lower, percent_b)."""
    v = _as_float(values)
    mid = sma(v, window)
    sd = rolling_std(v, window)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(width > 0, (v - lower) / width, 0.5)
    return upper, mid, lower, pct_b


def zscore(values, window: int = 20) -> np.ndarray:
    v = _as_float(values)
    mean = sma(v, window)
    sd = rolling_std(v, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (v - mean) / sd, 0.0)


def drawdown_series(equity) -> np.ndarray:
    e = _as_float(equity)
    peak = np.maximum.accumulate(np.where(np.isnan(e), -np.inf, e))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(peak > 0, e / peak - 1.0, 0.0)
