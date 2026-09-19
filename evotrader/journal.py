"""Trade and thought records — the material the breeder actually reasons over."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Trade:
    """One completed round trip."""

    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    ret: float                 # return on the capital committed to this trade
    bars_held: int
    entry_reason: str          # the rule that fired, verbatim
    exit_reason: str
    entry_context: Dict[str, float] = field(default_factory=dict)
    exit_context: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (f"{self.symbol} {self.entry_date}->{self.exit_date} "
                f"({self.bars_held}b) {self.ret * 100:+.1f}% "
                f"| in: {self.entry_reason} | out: {self.exit_reason}")


@dataclass
class Thought:
    """A dated note explaining what the agent was thinking.

    Deterministic genomes emit these when a rule fires; the LLM-in-the-loop mode
    stores Claude's own reasoning here.  Either way the breeder reads them.
    """

    date: str
    kind: str            # entry | exit | skip | commentary
    symbol: str
    text: str
    context: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Journal:
    """Everything one agent did over one backtest window."""

    trades: List[Trade] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    equity_dates: List[str] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    rejected_entries: int = 0        # signals blocked by risk limits
    rule_hits: Dict[str, int] = field(default_factory=dict)

    def record_hit(self, rule: str) -> None:
        self.rule_hits[rule] = self.rule_hits.get(rule, 0) + 1

    def best_trades(self, n: int = 5) -> List[Trade]:
        return sorted(self.trades, key=lambda t: t.ret, reverse=True)[:n]

    def worst_trades(self, n: int = 5) -> List[Trade]:
        return sorted(self.trades, key=lambda t: t.ret)[:n]

    def to_dict(self, *, max_trades: int = 0) -> Dict[str, Any]:
        trades = self.trades if max_trades <= 0 else self.trades[:max_trades]
        return {
            "trades": [t.to_dict() for t in trades],
            "thoughts": [t.to_dict() for t in self.thoughts],
            "equity_dates": self.equity_dates,
            "equity": self.equity,
            "rejected_entries": self.rejected_entries,
            "rule_hits": self.rule_hits,
        }
