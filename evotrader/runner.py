"""The backtest engine: run one compiled genome over one universe window.

Sequencing is strict so that results are honest:

    bar t close  ->  features observed  ->  rules evaluated  ->  orders queued
    bar t+1 open ->  orders fill (sells first, then buys)

Nothing a rule can see at bar ``t`` comes from bar ``t+1`` or later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .broker import PaperBroker
from .data import Universe
from .features import FeatureSet
from .genome import CompiledGenome, Genome, compile_genome
from .journal import Journal, Thought

NEVER = 9999
MAX_THOUGHTS = 300


@dataclass
class BacktestResult:
    genome_id: str
    journal: Journal
    final_equity: float
    starting_cash: float
    bars: int
    symbols: List[str]
    start_date: str
    end_date: str
    turnover: float          # gross traded / starting cash
    total_costs: float
    exposure: float          # average fraction of equity invested


def _portfolio_context(broker: PaperBroker, prices: Dict[str, float],
                       equity: float) -> Dict[str, float]:
    invested = broker.invested(prices)
    return {
        "cash_pct": broker.cash / equity if equity else 1.0,
        "gross_exposure": invested / equity if equity else 0.0,
        "position_count": float(len(broker.positions)),
        "portfolio_return": equity / broker.starting_cash - 1.0 if broker.starting_cash else 0.0,
        "portfolio_drawdown": broker.drawdown(prices),
    }


def _position_context(broker: PaperBroker, symbol: str, price: float, bar: int,
                      equity: float, last_exit_bar: Dict[str, int]) -> Dict[str, float]:
    pos = broker.positions.get(symbol)
    if pos is None:
        since = last_exit_bar.get(symbol)
        return {
            "in_position": 0.0, "bars_held": 0.0, "position_return": 0.0,
            "position_drawdown": 0.0, "position_weight": 0.0,
            "bars_since_exit": float(NEVER if since is None else bar - since),
        }
    return {
        "in_position": 1.0,
        "bars_held": float(bar - pos.entry_bar),
        "position_return": pos.unrealised_return(price),
        "position_drawdown": pos.drawdown(price),
        "position_weight": pos.value(price) / equity if equity else 0.0,
        "bars_since_exit": float(NEVER),
    }


def _risk_exit(pos, price: float, bar: int, risk) -> Optional[str]:
    """Hard guardrails, checked before the genome's own exit rules."""
    ret = pos.unrealised_return(price)
    if risk.stop_loss_pct > 0 and ret <= -risk.stop_loss_pct:
        return f"stop loss hit ({ret * 100:.1f}% <= -{risk.stop_loss_pct * 100:.1f}%)"
    if risk.take_profit_pct > 0 and ret >= risk.take_profit_pct:
        return f"take profit hit ({ret * 100:.1f}% >= {risk.take_profit_pct * 100:.1f}%)"
    if risk.trailing_stop_pct > 0 and pos.drawdown(price) <= -risk.trailing_stop_pct:
        return f"trailing stop hit ({pos.drawdown(price) * 100:.1f}%)"
    if risk.max_hold_bars > 0 and (bar - pos.entry_bar) >= risk.max_hold_bars:
        return f"max hold reached ({risk.max_hold_bars} bars)"
    return None


def run_backtest(compiled: CompiledGenome, universe: Universe, features: FeatureSet, *,
                 starting_cash: float = 100_000.0, commission_bps: float = 1.0,
                 slippage_bps: float = 5.0, record_thoughts: bool = True,
                 start_bar: Optional[int] = None) -> BacktestResult:
    """Simulate one genome and return its journal plus summary statistics."""
    genome = compiled.genome
    risk = compiled.risk
    symbols = features.symbols
    dates = features.dates
    n = len(dates)
    journal = Journal()
    broker = PaperBroker(starting_cash, commission_bps=commission_bps,
                         slippage_bps=slippage_bps, journal=journal)
    first = features.warmup if start_bar is None else max(start_bar, 1)
    if n - first < 5:
        first = max(n - 5, 1)

    last_exit_bar: Dict[str, int] = {}
    prev_snap: Dict[str, Dict[str, float]] = {}
    exposure_sum = 0.0
    exposure_n = 0

    for i in range(first, n - 1):
        date = dates[i]
        prices = {s: float(universe.bars[s].close[i]) for s in symbols}
        equity = broker.mark(date, prices)
        if equity <= 0:  # wiped out; nothing left to trade
            break
        exposure_sum += broker.invested(prices) / equity
        exposure_n += 1
        pf = _portfolio_context(broker, prices, equity)

        snaps: Dict[str, Dict[str, float]] = {}
        for sym in symbols:
            snap = features.snapshot(sym, i)
            snap.update(pf)
            snap.update(_position_context(broker, sym, prices[sym], i, equity, last_exit_bar))
            snaps[sym] = snap

        sells: List[Tuple[str, str]] = []
        buys: List[Tuple[str, float, str]] = []

        # ---- exits first: they free capital for the same bar's entries
        for sym in symbols:
            pos = broker.positions.get(sym)
            if pos is None:
                continue
            cur = snaps[sym]
            prev = prev_snap.get(sym, cur)
            reason = _risk_exit(pos, prices[sym], i, risk)
            if reason is None and (i - pos.entry_bar) >= risk.min_hold_bars:
                for rule in compiled.exits:
                    if rule(cur, prev):
                        reason = f"exit rule: {rule.source}"
                        journal.record_hit(rule.source)
                        break
            if reason:
                sells.append((sym, reason))

        # ---- entries: first matching rule wins, subject to the risk budget
        open_after_sells = len(broker.positions) - len(sells)
        invested_after = broker.invested(prices) - sum(
            broker.positions[s].value(prices[s]) for s, _ in sells)
        for sym in symbols:
            if sym in broker.positions and sym not in {s for s, _ in sells}:
                continue
            if sym in {s for s, _ in sells}:
                continue  # never re-enter on the same bar we exit
            if risk.cooldown_bars > 0:
                last = last_exit_bar.get(sym)
                if last is not None and (i - last) < risk.cooldown_bars:
                    continue
            cur = snaps[sym]
            prev = prev_snap.get(sym, cur)
            for rule, weight in compiled.entries:
                if not rule(cur, prev):
                    continue
                journal.record_hit(rule.source)
                if open_after_sells >= risk.max_positions:
                    journal.rejected_entries += 1
                    break
                target = min(weight, risk.max_position_pct) * equity
                headroom = risk.max_gross_exposure * equity - invested_after
                target = min(target, headroom)
                if target < broker.min_trade_value:
                    journal.rejected_entries += 1
                    break
                buys.append((sym, target, f"entry rule: {rule.source}"))
                open_after_sells += 1
                invested_after += target
                break

        # ---- execute at the next bar's open
        j = i + 1
        fill_date = dates[j]
        for sym, reason in sells:
            price = float(universe.bars[sym].open[j])
            trade = broker.sell(sym, price, fill_date, j, reason, context=snaps[sym])
            if trade and record_thoughts and len(journal.thoughts) < MAX_THOUGHTS:
                journal.thoughts.append(Thought(
                    date=fill_date, kind="exit", symbol=sym,
                    text=f"{reason}; realised {trade.ret * 100:+.1f}% over {trade.bars_held} bars",
                    context={k: round(snaps[sym].get(k, 0.0), 4)
                             for k in ("rsi14", "dist_sma200", "position_return", "atr_pct")},
                ))
            last_exit_bar[sym] = j
        for sym, notional, reason in buys:
            price = float(universe.bars[sym].open[j])
            if broker.buy(sym, notional, price, fill_date, j, reason, context=snaps[sym]):
                if record_thoughts and len(journal.thoughts) < MAX_THOUGHTS:
                    journal.thoughts.append(Thought(
                        date=fill_date, kind="entry", symbol=sym,
                        text=f"{reason}; sized {notional / equity:.0%} of equity",
                        context={k: round(snaps[sym].get(k, 0.0), 4)
                                 for k in ("rsi14", "dist_sma200", "ret20", "atr_pct",
                                           "mkt_above_sma200")},
                    ))
            else:
                journal.rejected_entries += 1

        prev_snap = snaps

    # ---- close out at the final bar
    last = n - 1
    final_prices = {s: float(universe.bars[s].close[last]) for s in symbols}
    broker.liquidate(final_prices, dates[last], last)
    final_equity = broker.mark(dates[last], final_prices)

    return BacktestResult(
        genome_id=genome.id,
        journal=journal,
        final_equity=final_equity,
        starting_cash=starting_cash,
        bars=max(n - first, 0),
        symbols=list(symbols),
        start_date=dates[first] if first < n else "",
        end_date=dates[last] if n else "",
        turnover=broker.gross_traded / starting_cash if starting_cash else 0.0,
        total_costs=broker.total_costs,
        exposure=exposure_sum / exposure_n if exposure_n else 0.0,
    )


def backtest_genome(genome: Genome, universe: Universe, features: FeatureSet,
                    **kwargs) -> BacktestResult:
    """Compile then backtest — convenience wrapper."""
    return run_backtest(compile_genome(genome), universe, features, **kwargs)


def buy_and_hold(universe: Universe, features: FeatureSet, *,
                 starting_cash: float = 100_000.0) -> List[float]:
    """Equal-weight buy-and-hold equity curve over the same window, as the
    benchmark every genome is scored against."""
    symbols = features.symbols
    n = len(features.dates)
    start = features.warmup
    if start >= n - 1:
        return [starting_cash] * max(n - start, 1)
    per = starting_cash / len(symbols)
    shares = {s: per / float(universe.bars[s].close[start]) for s in symbols}
    return [sum(shares[s] * float(universe.bars[s].close[i]) for s in symbols)
            for i in range(start, n)]
