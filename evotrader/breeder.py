"""Breeding: turn one generation's results into the next generation.

Two breeders, used together:

* :class:`LLMBreeder` — Claude reads the elites' genomes and journals, writes an
  explicit analysis of what worked and why, then authors new genomes.  This is
  the part that makes the loop more than a random search: it can transfer a
  *reason* from one generation to the next, not just a parameter.
* :class:`MutationBreeder` — cheap, offline genetic operators.  They supply
  diversity every generation and take over entirely when the API is
  unavailable or the run's cost budget is spent.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fitness import Evaluation
from .genome import Genome, GenomeError, compile_genome
from .journal import Journal
from .llm import BudgetExceeded, Claude, LLMError
from .population import crossover, mutate, random_genome
from .prompts import GENOME_JSON_SCHEMA, SYSTEM_PROMPT, build_breeding_prompt

#: (genome, evaluation, journal) for one elite.
Elite = Tuple[Genome, Evaluation, Optional[Journal]]


@dataclass
class BreedResult:
    genomes: List[Genome] = field(default_factory=list)
    analysis: str = ""
    lessons: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    rejected: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"analysis": self.analysis, "lessons": self.lessons,
                "cost_usd": round(self.cost_usd, 4), "rejected": self.rejected,
                "error": self.error, "offspring": len(self.genomes)}


class MutationBreeder:
    """Elitist mutation + crossover + random immigrants.  No API needed."""

    def __init__(self, rng: Optional[random.Random] = None, *,
                 crossover_rate: float = 0.35, immigrant_rate: float = 0.10):
        self.rng = rng or random.Random()
        self.crossover_rate = crossover_rate
        self.immigrant_rate = immigrant_rate

    def breed(self, elites: Sequence[Elite], count: int, *, generation: int,
              parent_pool: Optional[Sequence[Genome]] = None) -> BreedResult:
        parents = [g for g, _, _ in elites] or list(parent_pool or [])
        out: List[Genome] = []
        if not parents:
            return BreedResult(genomes=[random_genome(self.rng, generation=generation)
                                        for _ in range(count)])
        pool = list(parent_pool or parents)
        for _ in range(count):
            roll = self.rng.random()
            if roll < self.immigrant_rate:
                out.append(random_genome(self.rng, generation=generation))
            elif roll < self.immigrant_rate + self.crossover_rate and len(pool) > 1:
                a, b = self.rng.sample(pool, 2)
                out.append(crossover(a, b, self.rng, generation=generation))
            else:
                parent = _tournament(pool, self.rng)
                out.append(mutate(parent, self.rng, generation=generation))
        return BreedResult(genomes=out)


def _tournament(pool: Sequence[Genome], rng: random.Random, k: int = 3) -> Genome:
    """Pool is assumed ranked best-first, so the lowest index wins."""
    picks = [rng.randrange(len(pool)) for _ in range(min(k, len(pool)))]
    return pool[min(picks)]


class LLMBreeder:
    """Claude-authored offspring, with validation and graceful degradation."""

    def __init__(self, claude: Claude, *, rng: Optional[random.Random] = None,
                 verbose: bool = True):
        self.claude = claude
        self.rng = rng or random.Random()
        self.verbose = verbose
        self.lessons: List[str] = []
        self.last_analysis: str = ""

    @property
    def available(self) -> bool:
        return self.claude.available and self.claude.budget_left() > 0

    def breed(self, elites: Sequence[Elite], count: int, *, generation: int,
              evals: Sequence[Evaluation] = (), history: Sequence[Dict[str, Any]] = (),
              window: str = "", extra: str = "") -> BreedResult:
        if not self.available:
            return BreedResult(error=self.claude.unavailable_reason or "LLM budget exhausted")

        prompt = build_breeding_prompt(
            generation=generation, elites=elites, evals=evals, history=history,
            n_offspring=count, lessons=self.lessons, window=window, extra=extra,
        )
        try:
            response = self.claude.complete(prompt, system=SYSTEM_PROMPT,
                                            schema=GENOME_JSON_SCHEMA)
        except BudgetExceeded as exc:
            return BreedResult(error=str(exc))
        except LLMError as exc:
            return BreedResult(error=str(exc))

        payload = response.data if isinstance(response.data, dict) else {}
        analysis = str(payload.get("analysis") or "")
        lessons = [str(x) for x in (payload.get("lessons") or []) if str(x).strip()][:10]
        genomes, rejected = self._materialise(payload.get("genomes") or [], generation)

        self.last_analysis = analysis
        for lesson in lessons:
            if lesson not in self.lessons:
                self.lessons.append(lesson)
        self.lessons = self.lessons[-24:]

        return BreedResult(genomes=genomes, analysis=analysis, lessons=lessons,
                           cost_usd=response.cost_usd, rejected=rejected)

    def _materialise(self, raw: Sequence[Any], generation: int) -> Tuple[List[Genome], int]:
        """Convert JSON objects into validated genomes, dropping what will not
        compile.  A malformed offspring is a wasted slot, never a crashed run."""
        out: List[Genome] = []
        rejected = 0
        for item in raw:
            try:
                genome = Genome.from_dict({**(item if isinstance(item, dict) else {}),
                                           "generation": generation, "origin": "llm"})
                # Identity is assigned here, never taken from the model's output.
                genome.id = uuid.uuid4().hex[:12]
                compile_genome(genome)
            except (GenomeError, TypeError, ValueError) as exc:
                rejected += 1
                if self.verbose:
                    name = item.get("name", "?") if isinstance(item, dict) else "?"
                    print(f"  [breeder] rejected offspring {name!r}: {exc}")
                continue
            out.append(genome)
        return out, rejected


class HybridBreeder:
    """The default: a share of Claude-authored offspring, the rest mutated.

    ``llm_share`` of each generation comes from Claude (when available and in
    budget); the remainder comes from mutation and crossover, which keeps the
    population diverse and the cost bounded.  Anything Claude fails to deliver
    is backfilled by mutation so the population size never drops.
    """

    def __init__(self, claude: Optional[Claude] = None, *,
                 rng: Optional[random.Random] = None, llm_share: float = 0.5,
                 llm_every: int = 1, verbose: bool = True):
        self.rng = rng or random.Random()
        self.mutation = MutationBreeder(self.rng)
        self.llm = LLMBreeder(claude, rng=self.rng, verbose=verbose) if claude else None
        self.llm_share = max(0.0, min(1.0, llm_share))
        self.llm_every = max(1, llm_every)
        self.verbose = verbose

    @property
    def lessons(self) -> List[str]:
        return self.llm.lessons if self.llm else []

    def breed(self, elites: Sequence[Elite], count: int, *, generation: int,
              evals: Sequence[Evaluation] = (), history: Sequence[Dict[str, Any]] = (),
              window: str = "", parent_pool: Optional[Sequence[Genome]] = None) -> BreedResult:
        use_llm = (self.llm is not None and self.llm.available
                   and self.llm_share > 0 and generation % self.llm_every == 0)
        result = BreedResult()
        if use_llm:
            want = max(1, int(round(count * self.llm_share)))
            llm_result = self.llm.breed(elites, want, generation=generation, evals=evals,
                                        history=history, window=window)
            result.genomes.extend(llm_result.genomes)
            result.analysis = llm_result.analysis
            result.lessons = llm_result.lessons
            result.cost_usd = llm_result.cost_usd
            result.rejected = llm_result.rejected
            result.error = llm_result.error
            if llm_result.error and self.verbose:
                print(f"  [breeder] Claude unavailable this generation: {llm_result.error}")

        shortfall = count - len(result.genomes)
        if shortfall > 0:
            filler = self.mutation.breed(elites, shortfall, generation=generation,
                                         parent_pool=parent_pool)
            result.genomes.extend(filler.genomes)
        return result
