"""Batch evaluation of a population over the evaluator, with objective caching.

This is the bridge between :mod:`medus.evolutionary.nsga2`, which knows nothing
about models, and :class:`medus_class.evaluation.evaluator.ClassEvaluator`,
which evaluates exactly one
chromosome. It adds the three things a search needs and a single evaluation does
not: caching, per-individual logging, and a failure policy.

Caching is keyed by the **canonical decoded strategy**, not by the genome.
Latent genes (everything behind ``b_i = 0``, and the operator gene when the
intensity is ``0``) mean many distinct genomes decode to the same executable
strategy, and re-running the evaluator for those would be pure waste. The identity
chromosome alone is generated repeatedly by mutation.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medus_class.search import Chromosome, decode
from medus_class.evaluation.evaluator import (
    PENALTY_OBJECTIVES,
    ClassEvaluator,
    ClassResult,
)


@dataclass
class EvaluationRecord:
    """One individual's evaluation, whether it ran or was served from cache."""

    generation: int
    individual: int
    canonical_key: str
    cached: bool
    status: str
    obj1: float
    obj2: float
    obj3: float
    runtime_seconds: float
    n_actions: int
    operators: str
    chromosome: list[int]
    active_groups: str = ""
    error: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "individual": self.individual,
            "canonical_key": self.canonical_key,
            "cached": int(self.cached),
            "status": self.status,
            "obj1_js": round(self.obj1, 6),
            "obj2_retain_loss": round(self.obj2, 6),
            "obj3_edit_cost": round(self.obj3, 6),
            "runtime_seconds": round(self.runtime_seconds, 3),
            "n_actions": self.n_actions,
            "operators": self.operators,
            "active_groups": self.active_groups,
            "chromosome": " ".join(str(g) for g in self.chromosome),
            "error": self.error,
        }


@dataclass
class PopulationEvaluator:
    """Evaluate whole generations through one evaluator instance.

    Attributes
    ----------
    evaluator:
        The evaluator. Reused across the whole run -- it holds the original
        weights and the loaders in memory, and rebuilding it per individual
        would dominate the runtime.
    cache_objectives:
        Cache by canonical decoded strategy. Off only for timing experiments.
    """

    evaluator: ClassEvaluator
    cache_objectives: bool = True

    generation: int = 0
    records: list[EvaluationRecord] = field(default_factory=list)
    _cache: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    _cached_status: dict[str, str] = field(default_factory=dict)

    #: Counters for the run report.
    evaluated: int = 0
    cache_hits: int = 0
    failures: int = 0

    def __call__(self, population: list[Chromosome]) -> list[tuple[float, float, float]]:
        """Evaluate one generation. Signature matches ``nsga2.EvaluateFn``."""
        objectives: list[tuple[float, float, float]] = []

        for index, chromosome in enumerate(population):
            # Decoded once and reused: the strategy supplies the cache key, the
            # action count AND the operator list. Cached rows must carry the
            # same operator information as evaluated ones, or any frequency
            # analysis over the history silently under-counts whichever
            # operators happen to recur.
            strategy = decode(chromosome, self.evaluator.registry.names)
            key = strategy.canonical_key()
            operators = "|".join(a.operator_name for a in strategy.actions) or "(none)"

            if self.cache_objectives and key in self._cache:
                values = self._cache[key]
                self.cache_hits += 1
                self.records.append(
                    EvaluationRecord(
                        generation=self.generation,
                        individual=index,
                        canonical_key=key,
                        cached=True,
                        status=self._cached_status[key],
                        obj1=values[0], obj2=values[1], obj3=values[2],
                        runtime_seconds=0.0,
                        n_actions=len(strategy.actions),
                        operators=operators,
                        active_groups="|".join(strategy.active_groups),
                        chromosome=chromosome.to_vector().tolist(),
                    )
                )
                objectives.append(values)
                continue

            started = time.perf_counter()
            result: ClassResult = self.evaluator.evaluate(chromosome)
            elapsed = time.perf_counter() - started

            # Read through the `objectives` property rather than by field
            # name: it is the one place that defines NSGA-II's (f1, f2, f3)
            # ordering, so a rename cannot silently desync the two again.
            values = tuple(float(v) for v in result.objectives)
            self.evaluated += 1
            if result.status != "ok":
                # A failed evaluation is scored at the penalty vector rather
                # than dropped: NSGA-II needs an objective value for every
                # individual, and the worst possible one both keeps the
                # population size correct and drives selection away from it.
                self.failures += 1
                # Mode-aware: loss-based objectives are unbounded, so the
                # bounded (1,1,1) penalty would rank a crash ABOVE a genuinely
                # bad candidate. See PENALTY_OBJECTIVES.
                values = PENALTY_OBJECTIVES

            if self.cache_objectives:
                self._cache[key] = values
                self._cached_status[key] = result.status

            self.records.append(
                EvaluationRecord(
                    generation=self.generation,
                    individual=index,
                    canonical_key=key,
                    cached=False,
                    status=result.status,
                    obj1=values[0], obj2=values[1], obj3=values[2],
                    runtime_seconds=elapsed,
                    n_actions=len(result.decoded_actions),
                    operators=operators,
                    active_groups="|".join(strategy.active_groups),
                    chromosome=chromosome.to_vector().tolist(),
                    error=(result.error or "").splitlines()[0] if result.error else "",
                )
            )
            objectives.append(values)

        self.generation += 1
        return objectives

    # -- reporting ----------------------------------------------------------

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def write_history(self, path: str | Path) -> Path:
        """Every individual ever evaluated, in order, as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [record.to_row() for record in self.records]
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def runtime_summary(self) -> dict[str, float]:
        """Per-individual runtime statistics, over evaluations that actually ran."""
        times = [r.runtime_seconds for r in self.records if not r.cached]
        if not times:
            return {"n": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "n": len(times),
            "total": round(sum(times), 2),
            "mean": round(sum(times) / len(times), 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
        }

    # -- checkpointing ------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Everything a resume needs to continue without losing history.

        The records matter as much as the cache: without them the final
        evaluation_history.csv would contain only the post-resume portion,
        silently corrupting every runtime statistic computed from it.
        """
        from dataclasses import asdict

        return {
            "records": [asdict(r) for r in self.records],
            "cache": dict(self._cache),
            "cached_status": dict(self._cached_status),
            "counters": {
                "generation": self.generation,
                "evaluated": self.evaluated,
                "cache_hits": self.cache_hits,
                "failures": self.failures,
            },
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`export_state`."""
        self.records = [EvaluationRecord(**r) for r in state.get("records", [])]
        self._cache = dict(state.get("cache", {}))
        self._cached_status = dict(state.get("cached_status", {}))
        counters = state.get("counters", {})
        self.generation = counters.get("generation", 0)
        self.evaluated = counters.get("evaluated", 0)
        self.cache_hits = counters.get("cache_hits", 0)
        self.failures = counters.get("failures", 0)
