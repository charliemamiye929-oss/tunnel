"""Performance metrics and the fitness function that drives selection.

Fitness is deliberately *not* raw return.  Selecting on return alone breeds
agents that lever into one lucky regime; the composite below rewards
risk-adjusted return and excess return over buy-and-hold, then charges for
drawdown, churn, and statistically meaningless trade counts.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TRADING_DAYS = 252.0


@dataclass
class Metrics:
    total_return: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0      # negative, e.g. -0.23
    calmar: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_bars_held: float = 0.0
    turnover: float = 0.0      # gross traded / starting capital, per year
    exposure: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0     # total return minus buy-and-hold
    years: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}

    def summary(self) -> str:
        return (f"ret {self.total_return * 100:+.1f}% (bh {self.benchmark_return * 100:+.1f}%) "
                f"sharpe {self.sharpe:.2f} mdd {self.max_drawdown * 100:.1f}% "
                f"trades {self.trades} win {self.win_rate * 100:.0f}%")


def _equity_returns(equity: Sequence[float]) -> np.ndarray:
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        return np.zeros(0)
    prev = e[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(prev > 0, e[1:] / prev - 1.0, 0.0)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def compute_metrics(equity: Sequence[float], trades: Sequence, *,
                    benchmark: Optional[Sequence[float]] = None,
                    turnover: float = 0.0, exposure: float = 0.0,
                    bars_per_year: float = TRADING_DAYS) -> Metrics:
    """Summarise one backtest's equity curve and trade list."""
    e = np.asarray(equity, dtype=float)
    m = Metrics(exposure=exposure)
    if e.size < 2 or e[0] <= 0:
        m.turnover = turnover
        return m

    m.total_return = float(e[-1] / e[0] - 1.0)
    m.years = max(e.size / bars_per_year, 1e-9)
    # Turnover is annualised so that windows of different lengths are comparable.
    m.turnover = turnover / m.years
    if e[-1] > 0:
        m.cagr = float((e[-1] / e[0]) ** (1.0 / m.years) - 1.0)
    else:
        m.cagr = -1.0

    r = _equity_returns(e)
    if r.size > 1:
        sd = float(np.std(r, ddof=1))
        mean = float(np.mean(r))
        m.volatility = sd * math.sqrt(bars_per_year)
        m.sharpe = (mean / sd) * math.sqrt(bars_per_year) if sd > 0 else 0.0
        downside = r[r < 0]
        dsd = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
        m.sortino = (mean / dsd) * math.sqrt(bars_per_year) if dsd > 0 else m.sharpe

    peak = np.maximum.accumulate(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, e / peak - 1.0, 0.0)
    m.max_drawdown = float(np.min(dd)) if dd.size else 0.0
    m.calmar = m.cagr / abs(m.max_drawdown) if m.max_drawdown < -1e-9 else 0.0

    m.trades = len(trades)
    if trades:
        rets = np.asarray([t.ret for t in trades], dtype=float)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        m.win_rate = float(wins.size / rets.size)
        gross_win = float(np.sum([t.pnl for t in trades if t.pnl > 0]))
        gross_loss = float(-np.sum([t.pnl for t in trades if t.pnl <= 0]))
        m.profit_factor = gross_win / gross_loss if gross_loss > 1e-9 else (
            float("inf") if gross_win > 0 else 0.0)
        m.avg_trade_return = float(np.mean(rets))
        m.avg_bars_held = float(np.mean([t.bars_held for t in trades]))

    if benchmark is not None and len(benchmark) >= 2 and benchmark[0] > 0:
        m.benchmark_return = float(benchmark[-1] / benchmark[0] - 1.0)
        m.excess_return = m.total_return - m.benchmark_return
    return m


@dataclass
class FitnessConfig:
    """Weights for the composite score.  Tune these to change what evolves."""

    sharpe_weight: float = 1.0
    excess_weight: float = 1.5          # annualised excess return over buy-and-hold
    drawdown_limit: float = 0.20        # drawdown deeper than this is charged for
    drawdown_penalty: float = 3.0
    turnover_limit: float = 6.0         # annual gross traded / starting capital
    turnover_penalty: float = 0.10
    min_trades: int = 10                # below this the result is mostly noise
    inactivity_penalty: float = 1.5
    max_trades: int = 2000
    overtrading_penalty: float = 0.002
    ruin_threshold: float = -0.60       # a drawdown past this is treated as failure
    ruin_penalty: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fitness_score(m: Metrics, cfg: Optional[FitnessConfig] = None) -> float:
    """Single scalar used for ranking and selection.  Higher is better."""
    cfg = cfg or FitnessConfig()
    excess_annual = m.excess_return / max(m.years, 1e-9)
    score = cfg.sharpe_weight * _finite(m.sharpe) + cfg.excess_weight * _finite(excess_annual)

    over_dd = max(0.0, abs(m.max_drawdown) - cfg.drawdown_limit)
    score -= cfg.drawdown_penalty * over_dd
    if m.max_drawdown <= cfg.ruin_threshold:
        score -= cfg.ruin_penalty

    score -= cfg.turnover_penalty * max(0.0, m.turnover - cfg.turnover_limit)
    score -= cfg.overtrading_penalty * max(0, m.trades - cfg.max_trades)

    if m.trades < cfg.min_trades:
        shortfall = (cfg.min_trades - m.trades) / max(cfg.min_trades, 1)
        score -= cfg.inactivity_penalty * shortfall
    return float(score)


def _finite(x: float) -> float:
    if x is None or math.isnan(x) or math.isinf(x):
        return 0.0
    return float(x)


@dataclass
class Evaluation:
    """One genome's scored result on one window."""

    genome_id: str
    name: str
    generation: int
    score: float
    metrics: Metrics
    test_metrics: Optional[Metrics] = None      # held-out window, reporting only
    test_score: Optional[float] = None
    error: str = ""
    journal_ref: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id, "name": self.name,
            "generation": self.generation, "score": round(self.score, 6),
            "metrics": self.metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict() if self.test_metrics else None,
            "test_score": round(self.test_score, 6) if self.test_score is not None else None,
            "error": self.error,
        }


def rank(evaluations: List[Evaluation]) -> List[Evaluation]:
    """Best first; failed genomes always sort last."""
    return sorted(evaluations, key=lambda e: (e.error != "", -e.score))
