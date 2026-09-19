"""evotrader — evolving paper-trading agents, bred by Claude.

Quick start::

    from evotrader import EvolutionConfig, Evolution

    cfg = EvolutionConfig(symbols=["SPY", "QQQ"], population=40, generations=10)
    Evolution(cfg).run()
"""
from .config import EvolutionConfig
from .evolution import Evolution, score_genome
from .fitness import FitnessConfig, Metrics, compute_metrics, fitness_score
from .genome import Genome, compile_genome
from .llm import Claude
from .store import Store

__version__ = "0.1.0"
__all__ = [
    "EvolutionConfig", "Evolution", "score_genome", "FitnessConfig", "Metrics",
    "compute_metrics", "fitness_score", "Genome", "compile_genome", "Claude", "Store",
]
