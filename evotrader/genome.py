"""The genome: a readable, mutable description of one trading agent.

A genome is deliberately small and human-legible — a thesis in plain English
plus entry/exit rules and risk limits.  That is what makes the evolutionary loop
work: Claude can read a genome, read how it actually traded, and write a better
one.  Anything that is not expressible here is not heritable.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Sequence

from .dsl import DslError, Rule, compile_rule
from .features import FEATURE_SET

MAX_ENTRY_RULES = 6
MAX_EXIT_RULES = 6


class GenomeError(ValueError):
    """Raised when a genome is structurally invalid or uncompilable."""


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


@dataclass
class RiskParams:
    """Position sizing and exit guardrails, applied on top of the exit rules."""

    max_position_pct: float = 0.25      # cap on one position as a share of equity
    max_positions: int = 5              # concurrent open positions
    max_gross_exposure: float = 1.0     # total invested share of equity (1.0 = no leverage)
    stop_loss_pct: float = 0.10         # hard stop on unrealised loss (0 disables)
    take_profit_pct: float = 0.0        # hard target on unrealised gain (0 disables)
    trailing_stop_pct: float = 0.0      # give-back from the position's peak (0 disables)
    max_hold_bars: int = 0              # forced exit after N bars (0 disables)
    min_hold_bars: int = 0              # block rule exits before N bars
    cooldown_bars: int = 0              # bars to wait before re-entering a symbol

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "RiskParams":
        d = d or {}
        return RiskParams(
            max_position_pct=_clamp(d.get("max_position_pct"), 0.01, 1.0, 0.25),
            max_positions=int(_clamp(d.get("max_positions"), 1, 20, 5)),
            max_gross_exposure=_clamp(d.get("max_gross_exposure"), 0.05, 1.0, 1.0),
            stop_loss_pct=_clamp(d.get("stop_loss_pct"), 0.0, 0.9, 0.10),
            take_profit_pct=_clamp(d.get("take_profit_pct"), 0.0, 5.0, 0.0),
            trailing_stop_pct=_clamp(d.get("trailing_stop_pct"), 0.0, 0.9, 0.0),
            max_hold_bars=int(_clamp(d.get("max_hold_bars"), 0, 2000, 0)),
            min_hold_bars=int(_clamp(d.get("min_hold_bars"), 0, 250, 0)),
            cooldown_bars=int(_clamp(d.get("cooldown_bars"), 0, 250, 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntryRule:
    when: str
    weight: float = 0.2   # target position size as a fraction of equity
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"when": self.when, "weight": round(self.weight, 4), "note": self.note}


@dataclass
class ExitRule:
    when: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"when": self.when, "note": self.note}


@dataclass
class Genome:
    """One agent.  ``id`` is stable for the lifetime of the genome."""

    name: str
    thesis: str = ""
    entry_rules: List[EntryRule] = field(default_factory=list)
    exit_rules: List[ExitRule] = field(default_factory=list)
    risk: RiskParams = field(default_factory=RiskParams)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    parents: List[str] = field(default_factory=list)
    origin: str = "seed"          # seed | llm | mutation | crossover | elite
    rationale: str = ""           # why the breeder created this genome
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------- lifecycle
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "thesis": self.thesis,
            "entry_rules": [r.to_dict() for r in self.entry_rules],
            "exit_rules": [r.to_dict() for r in self.exit_rules],
            "risk": self.risk.to_dict(),
            "generation": self.generation,
            "parents": list(self.parents),
            "origin": self.origin,
            "rationale": self.rationale,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Genome":
        if not isinstance(d, dict):
            raise GenomeError(f"genome must be an object, got {type(d).__name__}")
        name = str(d.get("name") or "unnamed").strip()[:80]
        entries: List[EntryRule] = []
        for raw in (d.get("entry_rules") or [])[:MAX_ENTRY_RULES]:
            if isinstance(raw, str):
                raw = {"when": raw}
            if not isinstance(raw, dict) or not raw.get("when"):
                continue
            entries.append(EntryRule(
                when=str(raw["when"]).strip(),
                weight=_clamp(raw.get("weight"), 0.01, 1.0, 0.2),
                note=str(raw.get("note") or "")[:200],
            ))
        exits: List[ExitRule] = []
        for raw in (d.get("exit_rules") or [])[:MAX_EXIT_RULES]:
            if isinstance(raw, str):
                raw = {"when": raw}
            if not isinstance(raw, dict) or not raw.get("when"):
                continue
            exits.append(ExitRule(when=str(raw["when"]).strip(),
                                  note=str(raw.get("note") or "")[:200]))
        g = Genome(
            name=name,
            thesis=str(d.get("thesis") or "")[:1200],
            entry_rules=entries,
            exit_rules=exits,
            risk=RiskParams.from_dict(d.get("risk")),
            generation=int(d.get("generation") or 0),
            parents=[str(p)[:32] for p in (d.get("parents") or [])][:4],
            origin=str(d.get("origin") or "seed")[:20],
            rationale=str(d.get("rationale") or "")[:1200],
        )
        if d.get("id"):
            g.id = str(d["id"])[:32]
        if d.get("created_at"):
            g.created_at = float(d["created_at"])
        return g

    def copy(self, **changes: Any) -> "Genome":
        clone = replace(
            self,
            entry_rules=[EntryRule(r.when, r.weight, r.note) for r in self.entry_rules],
            exit_rules=[ExitRule(r.when, r.note) for r in self.exit_rules],
            risk=RiskParams(**self.risk.to_dict()),
        )
        clone.id = uuid.uuid4().hex[:12]
        clone.created_at = time.time()
        for k, v in changes.items():
            setattr(clone, k, v)
        return clone

    # ------------------------------------------------------------ validation
    def fingerprint(self) -> str:
        """Hash of the behaviour-determining fields — used to spot clones."""
        payload = json.dumps({
            "entries": sorted((r.when, round(r.weight, 3)) for r in self.entry_rules),
            "exits": sorted(r.when for r in self.exit_rules),
            "risk": self.risk.to_dict(),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def describe(self) -> str:
        """Compact text form — this is what the breeder reads."""
        lines = [f"{self.name} (id={self.id}, gen={self.generation}, origin={self.origin})"]
        if self.thesis:
            lines.append(f"  thesis: {self.thesis}")
        for r in self.entry_rules:
            lines.append(f"  BUY  {r.weight:.0%} when: {r.when}" + (f"   # {r.note}" if r.note else ""))
        for r in self.exit_rules:
            lines.append(f"  SELL when: {r.when}" + (f"   # {r.note}" if r.note else ""))
        rk = self.risk
        lines.append(
            f"  risk: max_pos={rk.max_position_pct:.0%} max_open={rk.max_positions} "
            f"gross<={rk.max_gross_exposure:.0%} stop={rk.stop_loss_pct:.0%} "
            f"target={rk.take_profit_pct:.0%} trail={rk.trailing_stop_pct:.0%} "
            f"hold<={rk.max_hold_bars or '-'} hold>={rk.min_hold_bars} "
            f"cooldown={rk.cooldown_bars}"
        )
        return "\n".join(lines)


@dataclass
class CompiledGenome:
    """A genome with its rules parsed and validated, ready to backtest."""

    genome: Genome
    entries: List[tuple]        # (Rule, weight)
    exits: List[Rule]

    @property
    def risk(self) -> RiskParams:
        return self.genome.risk


def compile_genome(genome: Genome, allowed: Sequence[str] | frozenset = FEATURE_SET) -> CompiledGenome:
    """Compile every rule, raising :class:`GenomeError` on the first problem."""
    allowed = frozenset(allowed)
    if not genome.entry_rules:
        raise GenomeError(f"genome {genome.name!r} has no entry rules")
    if not genome.exit_rules:
        raise GenomeError(f"genome {genome.name!r} has no exit rules")
    entries = []
    for r in genome.entry_rules:
        try:
            entries.append((compile_rule(r.when, allowed), float(r.weight)))
        except DslError as exc:
            raise GenomeError(f"{genome.name}: bad entry rule: {exc}") from exc
    exits = []
    for r in genome.exit_rules:
        try:
            exits.append(compile_rule(r.when, allowed))
        except DslError as exc:
            raise GenomeError(f"{genome.name}: bad exit rule: {exc}") from exc
    return CompiledGenome(genome, entries, exits)


def is_valid(genome: Genome) -> bool:
    try:
        compile_genome(genome)
        return True
    except GenomeError:
        return False
