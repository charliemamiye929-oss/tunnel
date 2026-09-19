"""Run configuration: one JSON file describes an entire evolution run."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from .fitness import FitnessConfig
from .llm import DEFAULT_MODEL

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "XLE", "XLF", "XLK", "XLV"]


@dataclass
class EvolutionConfig:
    # --- what to trade
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    offline: bool = False              # use synthetic data instead of fetching
    refresh_data: bool = False
    test_frac: float = 0.25            # held-out tail, never used for selection

    # --- the loop
    population: int = 100
    generations: int = 1000
    elites: int = 5                    # survivors that breed and carry over
    survivor_reports: int = 5          # how many elite journals Claude reads
    seed: int = 0
    starting_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    # --- breeding
    breeder: str = "hybrid"            # hybrid | llm | mutation
    llm_share: float = 0.5             # share of each generation written by Claude
    llm_every: int = 1                 # call Claude every N generations
    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 16_000
    budget_usd: float = 0.0            # 0 = no ceiling
    immigrant_rate: float = 0.10
    crossover_rate: float = 0.35

    # --- fitness
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    # --- plumbing
    run_id: str = ""
    db_path: str = os.path.join("runs", "evotrader.sqlite")
    workers: int = 0                   # 0 = auto
    checkpoint_every: int = 1
    validate_top: int = 10             # genomes scored on the held-out window
    verbose: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fitness"] = self.fitness.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvolutionConfig":
        known = {f.name for f in fields(EvolutionConfig)}
        clean = {k: v for k, v in (d or {}).items() if k in known}
        fit = clean.pop("fitness", None)
        cfg = EvolutionConfig(**clean)
        if isinstance(fit, dict):
            valid = {f.name for f in fields(FitnessConfig)}
            cfg.fitness = FitnessConfig(**{k: v for k, v in fit.items() if k in valid})
        return cfg

    @staticmethod
    def load(path: str) -> "EvolutionConfig":
        with open(path) as fh:
            return EvolutionConfig.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def validate(self) -> None:
        if self.population < 4:
            raise ValueError("population must be at least 4")
        if self.elites < 1 or self.elites >= self.population:
            raise ValueError("elites must be between 1 and population-1")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.breeder not in ("hybrid", "llm", "mutation"):
            raise ValueError("breeder must be hybrid, llm or mutation")
        if not self.symbols:
            raise ValueError("at least one symbol is required")
