"""A paper broker: cash, positions, costs, and fills.

Deliberately conservative so that evolved edges are not artefacts of a
frictionless simulator:

* orders decided on bar *t* fill at bar *t+1*'s open (no look-ahead),
* every fill pays commission and crosses the spread via a slippage charge,
* positions are long-only, one lot per symbol, and cannot exceed available cash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .journal import Journal, Trade


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float          # fill price including costs
    entry_date: str
    entry_bar: int
    entry_reason: str
    peak_price: float
    entry_context: Dict[str, float] = field(default_factory=dict)

    def value(self, price: float) -> float:
        return self.shares * price

    def unrealised_return(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return price / self.entry_price - 1.0

    def drawdown(self, price: float) -> float:
        if self.peak_price <= 0:
            return 0.0
        return min(price / self.peak_price - 1.0, 0.0)


class PaperBroker:
    """Cash-and-positions account with explicit trading costs."""

    def __init__(self, starting_cash: float = 100_000.0, *,
                 commission_bps: float = 1.0, slippage_bps: float = 5.0,
                 min_trade_value: float = 100.0, journal: Optional[Journal] = None):
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.min_trade_value = float(min_trade_value)
        self.positions: Dict[str, Position] = {}
        self.journal = journal if journal is not None else Journal()
        self.gross_traded = 0.0       # sum of notional traded, for turnover
        self.total_costs = 0.0
        self.peak_equity = float(starting_cash)

    # ------------------------------------------------------------- accounting
    def equity(self, prices: Mapping[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            total += pos.value(prices.get(sym, pos.entry_price))
        return total

    def invested(self, prices: Mapping[str, float]) -> float:
        return sum(p.value(prices.get(s, p.entry_price)) for s, p in self.positions.items())

    def mark(self, date: str, prices: Mapping[str, float]) -> float:
        """Record the equity curve point and refresh position peaks."""
        eq = self.equity(prices)
        self.peak_equity = max(self.peak_equity, eq)
        for sym, pos in self.positions.items():
            price = prices.get(sym)
            if price:
                pos.peak_price = max(pos.peak_price, price)
        self.journal.equity_dates.append(date)
        self.journal.equity.append(eq)
        return eq

    def drawdown(self, prices: Mapping[str, float]) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return min(self.equity(prices) / self.peak_equity - 1.0, 0.0)

    # ----------------------------------------------------------------- orders
    def _buy_price(self, px: float) -> float:
        return px * (1.0 + self.slippage_bps / 10_000.0)

    def _sell_price(self, px: float) -> float:
        return px * (1.0 - self.slippage_bps / 10_000.0)

    def buy(self, symbol: str, notional: float, price: float, date: str, bar: int,
            reason: str, context: Optional[Dict[str, float]] = None) -> bool:
        """Open a long position worth roughly ``notional``.  Returns True on fill."""
        if symbol in self.positions or price <= 0 or notional <= 0:
            return False
        fill = self._buy_price(price)
        comm_rate = self.commission_bps / 10_000.0
        # Clip the requested notional to what the cash balance can actually
        # cover once commission is added, rather than rejecting the order.
        budget = min(float(notional), self.cash / (1.0 + comm_rate))
        if budget < self.min_trade_value:
            return False
        shares = budget / fill
        cost = shares * fill
        commission = cost * comm_rate
        self.cash -= cost + commission
        self.total_costs += commission + shares * (fill - price)
        self.gross_traded += cost
        self.positions[symbol] = Position(
            symbol=symbol, shares=shares, entry_price=fill, entry_date=date,
            entry_bar=bar, entry_reason=reason, peak_price=fill,
            entry_context=dict(context or {}),
        )
        return True

    def sell(self, symbol: str, price: float, date: str, bar: int, reason: str,
             context: Optional[Dict[str, float]] = None) -> Optional[Trade]:
        """Close a position entirely and record the round trip."""
        pos = self.positions.pop(symbol, None)
        if pos is None or price <= 0:
            return None
        fill = self._sell_price(price)
        proceeds = pos.shares * fill
        commission = proceeds * self.commission_bps / 10_000.0
        self.cash += proceeds - commission
        self.total_costs += commission + pos.shares * (price - fill)
        self.gross_traded += proceeds
        cost_basis = pos.shares * pos.entry_price
        pnl = proceeds - commission - cost_basis
        trade = Trade(
            symbol=symbol, entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=fill, shares=pos.shares,
            pnl=pnl, ret=(pnl / cost_basis if cost_basis else 0.0),
            bars_held=max(bar - pos.entry_bar, 0),
            entry_reason=pos.entry_reason, exit_reason=reason,
            entry_context=pos.entry_context, exit_context=dict(context or {}),
        )
        self.journal.trades.append(trade)
        return trade

    def liquidate(self, prices: Mapping[str, float], date: str, bar: int,
                  reason: str = "end of backtest") -> List[Trade]:
        out: List[Trade] = []
        for symbol in list(self.positions):
            price = prices.get(symbol)
            if price:
                trade = self.sell(symbol, price, date, bar, reason)
                if trade:
                    out.append(trade)
        return out
