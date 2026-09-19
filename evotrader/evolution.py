"""The evolution loop.

    seed a population
    for each generation:
        backtest every agent on the training window   (free, deterministic)
        rank by fitness, keep the elites
        show the elites' genomes and journals to Claude, which explains what
            worked and writes the next generation                 (one API call)
        backfill with mutation and crossover, carry the elites over
        checkpoint everything to SQLite

The expensive part is bounded on purpose: one breeding call per generation, not
one call per trade.  A run of 1000 generations x 100 agents is 100k backtests
and about 1000 API calls.
"""
from __future__ import annotations

import math
import os
import random
import statistics
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .breeder import BreedResult, Elite, HybridBreeder, LLMBreeder, MutationBreeder
from .config import EvolutionConfig
from .data import Universe, holdout_split, load_universe
from .features import FeatureSet, build_features
from .fitness import Evaluation, FitnessConfig, Metrics, compute_metrics, fitness_score, rank
from .genome import Genome, GenomeError, compile_genome
from .journal import Journal
from .llm import Claude
from .population import seed_population
from .runner import buy_and_hold, run_backtest
from .store import Store


@dataclass
class Context:
    """Everything a worker process needs to score a genome."""

    train: Universe
    train_features: FeatureSet
    train_benchmark: List[float]
    test: Optional[Universe]
    test_features: Optional[FeatureSet]
    test_benchmark: Optional[List[float]]
    starting_cash: float
    commission_bps: float
    slippage_bps: float
    fitness: FitnessConfig


@dataclass
class Outcome:
    """One genome's evaluation plus the journal the breeder will read."""

    genome_id: str
    name: str
    generation: int
    score: float
    metrics: Metrics
    journal: Optional[Journal] = None
    error: str = ""


_CTX: Optional[Context] = None


def _init_worker(ctx: Context) -> None:  # pragma: no cover - process boundary
    global _CTX
    _CTX = ctx


def _score_in_worker(payload: Dict[str, Any]) -> Outcome:  # pragma: no cover
    assert _CTX is not None
    return score_genome(Genome.from_dict(payload), _CTX)


def score_genome(genome: Genome, ctx: Context, *, window: str = "train") -> Outcome:
    """Backtest one genome and turn the result into a fitness score."""
    universe = ctx.train if window == "train" else ctx.test
    features = ctx.train_features if window == "train" else ctx.test_features
    benchmark = ctx.train_benchmark if window == "train" else ctx.test_benchmark
    if universe is None or features is None:
        return Outcome(genome.id, genome.name, genome.generation, float("-inf"),
                       Metrics(), None, "no data for window")
    try:
        compiled = compile_genome(genome)
        result = run_backtest(
            compiled, universe, features, starting_cash=ctx.starting_cash,
            commission_bps=ctx.commission_bps, slippage_bps=ctx.slippage_bps)
    except GenomeError as exc:
        return Outcome(genome.id, genome.name, genome.generation, float("-inf"),
                       Metrics(), None, f"invalid genome: {exc}")
    except Exception as exc:  # noqa: BLE001 - one bad genome must not stop a run
        return Outcome(genome.id, genome.name, genome.generation, float("-inf"),
                       Metrics(), None, f"backtest failed: {type(exc).__name__}: {exc}")

    metrics = compute_metrics(result.journal.equity, result.journal.trades,
                              benchmark=benchmark, turnover=result.turnover,
                              exposure=result.exposure)
    journal = result.journal
    journal.equity = []          # the curve is large and only metrics need it
    journal.equity_dates = []
    return Outcome(genome.id, genome.name, genome.generation,
                   fitness_score(metrics, ctx.fitness), metrics, journal)


@dataclass
class GenerationReport:
    generation: int
    best: Optional[Evaluation]
    evaluations: List[Evaluation]
    elapsed_s: float
    cost_usd: float
    analysis: str = ""
    lessons: List[str] = field(default_factory=list)
    breeding_error: str = ""

    @property
    def scores(self) -> List[float]:
        return [e.score for e in self.evaluations if not e.error and math.isfinite(e.score)]


class Evolution:
    """Owns one run: data, population, breeder, persistence."""

    def __init__(self, cfg: EvolutionConfig, *, store: Optional[Store] = None,
                 claude: Optional[Claude] = None):
        cfg.validate()
        self.cfg = cfg
        self.run_id = cfg.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        cfg.run_id = self.run_id
        self.rng = random.Random(cfg.seed or None)
        self.store = store if store is not None else Store(cfg.db_path)
        self.claude = claude if claude is not None else Claude(
            model=cfg.model, max_tokens=cfg.max_tokens, effort=cfg.effort,
            budget_usd=cfg.budget_usd, verbose=cfg.verbose)
        self.breeder = self._make_breeder()
        self.ctx: Optional[Context] = None
        self.population: List[Genome] = []
        self.history: List[Dict[str, Any]] = []
        self.generation = 0
        self._cache: Dict[str, Outcome] = {}
        self._pool: Optional[ProcessPoolExecutor] = None
        self.window_label = ""

    # ------------------------------------------------------------- breeder
    def _make_breeder(self):
        if self.cfg.breeder == "mutation":
            m = MutationBreeder(self.rng, crossover_rate=self.cfg.crossover_rate,
                                immigrant_rate=self.cfg.immigrant_rate)
            return _MutationOnly(m)
        share = 1.0 if self.cfg.breeder == "llm" else self.cfg.llm_share
        breeder = HybridBreeder(self.claude, rng=self.rng, llm_share=share,
                                llm_every=self.cfg.llm_every, verbose=self.cfg.verbose)
        breeder.mutation.crossover_rate = self.cfg.crossover_rate
        breeder.mutation.immigrant_rate = self.cfg.immigrant_rate
        return breeder

    # ---------------------------------------------------------------- data
    def prepare(self) -> None:
        """Load market data, build features, split off the held-out window."""
        cfg = self.cfg
        self._log(f"loading {len(cfg.symbols)} symbols {cfg.start}..{cfg.end}"
                  f"{' (offline/synthetic)' if cfg.offline else ''}")
        universe = load_universe(cfg.symbols, cfg.start, cfg.end,
                                 offline=cfg.offline, refresh=cfg.refresh_data)
        train, test = holdout_split(universe, cfg.test_frac)
        train_features = build_features(train)
        test_features = build_features(test) if len(test) > 60 else None
        self.ctx = Context(
            train=train, train_features=train_features,
            train_benchmark=buy_and_hold(train, train_features,
                                         starting_cash=cfg.starting_cash),
            test=test if test_features else None, test_features=test_features,
            test_benchmark=(buy_and_hold(test, test_features,
                                         starting_cash=cfg.starting_cash)
                            if test_features else None),
            starting_cash=cfg.starting_cash, commission_bps=cfg.commission_bps,
            slippage_bps=cfg.slippage_bps, fitness=cfg.fitness,
        )
        a, b = train.date_range()
        self.window_label = f"train {a}..{b} ({len(train)} bars, {len(train.symbols)} symbols)"
        if self.ctx.test is not None:
            ta, tb = self.ctx.test.date_range()
            self.window_label += f"; held-out {ta}..{tb} ({len(self.ctx.test)} bars)"
        self._log(self.window_label)
        bh = self.ctx.train_benchmark
        if bh:
            self._log(f"buy-and-hold over the training window: "
                      f"{(bh[-1] / bh[0] - 1) * 100:+.1f}%")

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.ctx is None:
            self.prepare()
        self.store.create_run(self.run_id, self.cfg.to_dict(), self.cfg.note)
        self.population = seed_population(self.cfg.population, self.rng, generation=0)
        self.generation = 0
        self._log(f"run {self.run_id}: seeded {len(self.population)} agents")
        if not self.claude.available and self.cfg.breeder != "mutation":
            self._log(f"note: {self.claude.unavailable_reason}")

    def resume(self, run_id: str) -> None:
        """Continue a run from its most recent checkpoint."""
        stored = self.store.run_config(run_id)
        if stored is None:
            raise ValueError(f"unknown run {run_id!r}")
        checkpoint = self.store.latest_checkpoint(run_id)
        if checkpoint is None:
            raise ValueError(f"run {run_id!r} has no checkpoint to resume from")
        self.run_id = run_id
        self.cfg.run_id = run_id
        if self.ctx is None:
            self.prepare()
        self.population = checkpoint["population"]
        self.generation = int(checkpoint["generation"]) + 1
        self.history = self.store.generation_history(run_id)
        if isinstance(self.breeder, HybridBreeder) and self.breeder.llm:
            self.breeder.llm.lessons = list(checkpoint["lessons"])
        try:
            self.rng.setstate(_restore_state(checkpoint["rng_state"]))
        except (TypeError, ValueError):
            pass  # a fresh stream is acceptable; the population is what matters
        self.store.set_status(run_id, "running")
        self._log(f"resumed {run_id} at generation {self.generation} "
                  f"with {len(self.population)} agents")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # ---------------------------------------------------------------- loop
    def run(self, generations: Optional[int] = None) -> List[GenerationReport]:
        if not self.population:
            self.start()
        total = generations if generations is not None else self.cfg.generations
        reports: List[GenerationReport] = []
        try:
            for _ in range(total):
                reports.append(self.step())
        except KeyboardInterrupt:
            self._log("interrupted; the last checkpoint is intact "
                      f"(resume with: evotrader resume {self.run_id})")
            self.store.set_status(self.run_id, "interrupted")
            return reports
        finally:
            self.close()
        self.store.set_status(self.run_id, "finished")
        return reports

    def step(self) -> GenerationReport:
        """Evaluate the current population, then breed the next one."""
        assert self.ctx is not None, "call prepare() or start() first"
        started = time.time()
        gen = self.generation

        outcomes = self._evaluate(self.population)
        by_id = {o.genome_id: o for o in outcomes}
        genome_by_id = {g.id: g for g in self.population}

        evals = [Evaluation(genome_id=o.genome_id, name=o.name, generation=gen,
                            score=o.score, metrics=o.metrics, error=o.error)
                 for o in outcomes]
        ranked = rank(evals)
        self._validate_top(ranked, genome_by_id)

        elites: List[Elite] = []
        for ev in ranked[:self.cfg.survivor_reports]:
            if ev.error:
                continue
            genome = genome_by_id.get(ev.genome_id)
            if genome is not None:
                elites.append((genome, ev, by_id[ev.genome_id].journal))

        self.store.save_genomes(self.run_id, self.population)
        self.store.save_evaluations(self.run_id, gen, ranked)
        for genome, ev, journal in elites[:self.cfg.elites]:
            if journal is not None:
                self.store.save_trades(self.run_id, gen, genome.id, journal.trades)

        best = ranked[0] if ranked and not ranked[0].error else None
        report = GenerationReport(generation=gen, best=best, evaluations=ranked,
                                  elapsed_s=time.time() - started, cost_usd=0.0)

        # ---- breed the next generation
        survivors = [g for g, _, _ in elites][:self.cfg.elites]
        n_offspring = self.cfg.population - len(survivors)
        breed = self.breeder.breed(
            elites[:self.cfg.survivor_reports], n_offspring, generation=gen,
            evals=ranked, history=self.history, window=self.window_label,
            parent_pool=survivors or self.population,
        )
        report.cost_usd = breed.cost_usd
        report.analysis = breed.analysis
        report.lessons = list(breed.lessons)
        report.breeding_error = breed.error

        children = breed.genomes[:n_offspring]
        while len(children) < n_offspring:   # never let the population shrink
            filler = self.breeder.breed(elites[:1], n_offspring - len(children),
                                        generation=gen, parent_pool=survivors or self.population)
            if not filler.genomes:
                break
            children.extend(filler.genomes)
        next_population = [g.copy(generation=gen + 1, origin="elite",
                                  parents=[g.id], rationale="carried over as elite")
                           for g in survivors] + children[:n_offspring]

        scores = report.scores
        self.store.save_generation(
            self.run_id, gen,
            best_score=best.score if best else float("nan"),
            mean_score=statistics.fmean(scores) if scores else float("nan"),
            median_score=statistics.median(scores) if scores else float("nan"),
            best_genome_id=best.genome_id if best else "",
            analysis=breed.analysis, lessons=breed.lessons,
            cost_usd=breed.cost_usd, elapsed_s=report.elapsed_s)
        self.history.append({
            "generation": gen,
            "best_score": best.score if best else float("nan"),
            "mean_score": statistics.fmean(scores) if scores else float("nan"),
            "best_name": best.name if best else "?",
            "best_genome_id": best.genome_id if best else "",
        })

        if self.cfg.checkpoint_every and gen % self.cfg.checkpoint_every == 0:
            self.store.save_checkpoint(self.run_id, gen, next_population,
                                       self.breeder.lessons if hasattr(self.breeder, "lessons") else [],
                                       self.rng.getstate(), self.claude.usage.to_dict())

        self._log_generation(report, breed)
        self.population = next_population
        self.generation = gen + 1
        return report

    # ---------------------------------------------------------- evaluation
    def _evaluate(self, population: Sequence[Genome]) -> List[Outcome]:
        """Score every genome, reusing cached results for unchanged agents."""
        assert self.ctx is not None
        todo: List[Genome] = []
        outcomes: List[Outcome] = []
        for genome in population:
            cached = self._cache.get(genome.fingerprint())
            if cached is not None:
                outcomes.append(Outcome(genome.id, genome.name, genome.generation,
                                        cached.score, cached.metrics, cached.journal,
                                        cached.error))
            else:
                todo.append(genome)

        if todo:
            workers = self._worker_count(len(todo))
            if workers > 1:
                pool = self._get_pool(workers)
                payloads = [g.to_dict() for g in todo]
                fresh = list(pool.map(_score_in_worker, payloads, chunksize=4))
            else:
                fresh = [score_genome(g, self.ctx) for g in todo]
            for genome, outcome in zip(todo, fresh):
                if len(self._cache) < 50_000:
                    self._cache[genome.fingerprint()] = outcome
                outcomes.append(outcome)
        return outcomes

    def _validate_top(self, ranked: Sequence[Evaluation],
                      genome_by_id: Dict[str, Genome]) -> None:
        """Score the leaders on the held-out window — reporting only, never
        selection, so the out-of-sample result stays honest."""
        assert self.ctx is not None
        if self.ctx.test is None or self.cfg.validate_top <= 0:
            return
        for ev in list(ranked)[:self.cfg.validate_top]:
            if ev.error:
                continue
            genome = genome_by_id.get(ev.genome_id)
            if genome is None:
                continue
            out = score_genome(genome, self.ctx, window="test")
            if not out.error:
                ev.test_metrics = out.metrics
                ev.test_score = out.score

    def _worker_count(self, jobs: int) -> int:
        if self.cfg.workers > 0:
            return min(self.cfg.workers, jobs)
        if self.cfg.workers == 0 and jobs >= 16:
            return min(max((os.cpu_count() or 2) - 1, 1), jobs, 16)
        return 1

    def _get_pool(self, workers: int) -> ProcessPoolExecutor:
        if self._pool is None or getattr(self._pool, "_max_workers", 0) != workers:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
            self._pool = ProcessPoolExecutor(max_workers=workers,
                                             initializer=_init_worker,
                                             initargs=(self.ctx,))
        return self._pool

    # ------------------------------------------------------------ logging
    def _log(self, message: str) -> None:
        if self.cfg.verbose:
            print(message, flush=True)

    def _log_generation(self, report: GenerationReport, breed: BreedResult) -> None:
        if not self.cfg.verbose:
            return
        scores = report.scores
        best = report.best
        head = (f"gen {report.generation:>4} | best {best.score:+.3f} "
                f"| mean {statistics.fmean(scores):+.3f} " if best and scores
                else f"gen {report.generation:>4} | no scored agents ")
        detail = f"| {best.name[:28]:<28} {best.metrics.summary()}" if best else ""
        timing = f" | {report.elapsed_s:.1f}s"
        if breed.cost_usd:
            timing += f" | ${breed.cost_usd:.3f} (run ${self.claude.usage.cost_usd:.2f})"
        print(head + detail + timing, flush=True)
        if best is not None and best.test_metrics is not None:
            print(f"        held-out: {best.test_metrics.summary()}", flush=True)
        if breed.analysis:
            first = breed.analysis.strip().split("\n")[0]
            print(f"        claude: {first[:150]}", flush=True)


class _MutationOnly:
    """Adapter so the offline breeder matches the hybrid breeder's signature."""

    def __init__(self, mutation: MutationBreeder):
        self.mutation = mutation
        self.lessons: List[str] = []

    def breed(self, elites: Sequence[Elite], count: int, *, generation: int,
              evals: Sequence[Evaluation] = (), history: Sequence[Dict[str, Any]] = (),
              window: str = "", parent_pool: Optional[Sequence[Genome]] = None) -> BreedResult:
        return self.mutation.breed(elites, count, generation=generation,
                                   parent_pool=parent_pool)


def _restore_state(state: Any) -> tuple:
    """JSON turns the RNG state's tuples into lists; turn them back."""
    if isinstance(state, list):
        return tuple(_restore_state(x) for x in state)
    return state
