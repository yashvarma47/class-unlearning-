"""NSGA-II for the integer-coded MED-US genome.

Written from scratch (Deb et al., 2002) rather than taken from a library,
because the genome is not a real vector: it is five integer gene blocks with
different bounds, latent genes behind a ``b`` mask, and an expensive,
cache-able objective function. The standard SBX / polynomial-mutation operators
assume continuous variables and do not apply.

The three pieces of NSGA-II proper are here and are pure functions of an
objective matrix, so they can be tested against hand-worked examples without
touching a GPU:

* :func:`fast_non_dominated_sort` -- rank individuals into Pareto fronts.
* :func:`crowding_distance` -- density estimate used to break rank ties.
* :func:`binary_tournament` -- selection by (rank, then crowding).

:class:`NSGA2` wires them together with variation and an injected ``evaluate``
callback. The algorithm never imports the SEC: it is handed a function from a
list of chromosomes to a list of objective tuples, which keeps the search
testable with a cheap synthetic objective.

All three MED-US objectives are **minimised**, so domination is "no worse in
every objective and strictly better in at least one".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from medus_class.search.genome import GENE_NAMES, Chromosome, ChromosomeBounds
from medus_class.evaluation.metrics import normalise_objectives

#: A chromosome's objective vector. Minimised on every axis.
Objectives = tuple[float, float, float]
#: Evaluate a whole generation at once, so callers can batch or cache.
EvaluateFn = Callable[[list[Chromosome]], list[Objectives]]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Does ``a`` Pareto-dominate ``b``? (minimisation on every objective)"""
    no_worse = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return no_worse and strictly_better


def fast_non_dominated_sort(objectives: Sequence[Sequence[float]]) -> list[list[int]]:
    """Partition indices into Pareto fronts, best first.

    Returns a list of fronts; ``fronts[0]`` is the non-dominated set. This is
    Deb's O(MN^2) formulation: count how many individuals dominate each one, and
    peel off the zero-count layer repeatedly.

    Duplicate objective vectors never dominate each other (domination requires a
    strict improvement somewhere), so identical individuals share a front. That
    matters here because latent genes make distinct genomes decode to the same
    strategy and therefore to identical objectives.
    """
    n = len(objectives)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
                domination_count[q] += 1
            elif dominates(objectives[q], objectives[p]):
                dominated_by[q].append(p)
                domination_count[p] += 1

    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        following: list[int] = []
        for p in fronts[current]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    following.append(q)
        current += 1
        fronts.append(following)

    fronts.pop()  # the loop always appends one empty front before stopping
    return fronts


def crowding_distance(objectives: Sequence[Sequence[float]]) -> list[float]:
    """Deb's crowding distance over one front.

    Boundary solutions get ``inf`` so the extremes of the front are always
    preserved. An objective with zero range across the front contributes
    nothing -- dividing by that range would be a division by zero, and the
    objective genuinely carries no discriminating information here.

    This is exactly the case MED-US hits early: ``obj2`` is 0 for every
    chromosome that has not yet damaged utility, so without the guard a whole
    generation would come out ``nan`` and selection would collapse.
    """
    n = len(objectives)
    if n == 0:
        return []
    if n <= 2:
        return [math.inf] * n

    n_objectives = len(objectives[0])
    distance = [0.0] * n

    for m in range(n_objectives):
        order = sorted(range(n), key=lambda i: objectives[i][m])
        lowest = objectives[order[0]][m]
        highest = objectives[order[-1]][m]

        distance[order[0]] = math.inf
        distance[order[-1]] = math.inf

        spread = highest - lowest
        if spread <= 0.0:
            continue

        for position in range(1, n - 1):
            index = order[position]
            if distance[index] == math.inf:
                continue
            previous = objectives[order[position - 1]][m]
            following = objectives[order[position + 1]][m]
            distance[index] += (following - previous) / spread

    return distance


def crowded_comparison(
    a: int, b: int, rank: Sequence[int], distance: Sequence[float]
) -> int:
    """The NSGA-II partial order: lower rank wins, then larger crowding distance.

    Returns ``-1`` if ``a`` is preferred, ``1`` if ``b`` is, ``0`` if neither.
    """
    if rank[a] != rank[b]:
        return -1 if rank[a] < rank[b] else 1
    if distance[a] != distance[b]:
        return -1 if distance[a] > distance[b] else 1
    return 0


def binary_tournament(
    rng: np.random.Generator,
    count: int,
    rank: Sequence[int],
    distance: Sequence[float],
) -> list[int]:
    """Pick ``count`` parents by binary tournament on the crowded comparison.

    Ties are broken by coin flip rather than by index, so a population with many
    equal individuals -- common before the operators start separating them --
    does not systematically favour whoever happens to sit earlier in the list.
    """
    n = len(rank)
    chosen: list[int] = []
    for _ in range(count):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        verdict = crowded_comparison(a, b, rank, distance)
        if verdict < 0:
            chosen.append(a)
        elif verdict > 0:
            chosen.append(b)
        else:
            chosen.append(a if rng.random() < 0.5 else b)
    return chosen


def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: np.random.Generator,
    probability: float,
) -> tuple[Chromosome, Chromosome]:
    """Swap gene positions independently with probability 0.5.

    Uniform rather than one-point: the genome is gene-major
    ``[b | g | s | d_g | d_s]``, so a single cut point would almost always split
    between gene *types* rather than mixing strategies, and children would
    inherit whole channels intact. Uniform crossover mixes at the level of
    individual (gene, group) pairs, which is what a layer-wise strategy is made
    of.
    """
    if rng.random() >= probability:
        return parent_a.copy(), parent_b.copy()

    a = parent_a.to_vector()
    b = parent_b.to_vector()
    mask = rng.random(a.size) < 0.5

    child_a = np.where(mask, b, a)
    child_b = np.where(mask, a, b)
    return (
        Chromosome.from_vector(child_a, parent_a.bounds),
        Chromosome.from_vector(child_b, parent_a.bounds),
    )


def random_reset_mutation(
    chromosome: Chromosome, rng: np.random.Generator, probability: float
) -> Chromosome:
    """Reset each gene to a uniform draw from *its own* bounds, independently.

    Per-gene bounds matter: ``b`` is binary, ``g``/``s`` index the selectable
    operator tuples, and ``d_g``/``d_s`` run 0-5. A single shared range would
    generate invalid genomes, and clipping them afterwards would bias the
    distribution towards the bounds.

    Drawing from the full range (rather than nudging) is the right move for an
    unordered categorical gene: operator ID 2 is not "between" 1 and 3 in any
    meaningful sense, so a local step has no meaning on the ``g``/``s`` genes.
    """
    genes = {}
    for name in GENE_NAMES:
        vector = getattr(chromosome, name).copy()
        low, high = chromosome.bounds.gene_bounds(name)
        for index in range(vector.size):
            if rng.random() < probability:
                vector[index] = int(rng.integers(low, high + 1))
        genes[name] = vector
    return Chromosome(bounds=chromosome.bounds, **genes)


@dataclass
class GenerationRecord:
    """What happened in one generation, for the history log."""

    generation: int
    evaluated: int
    cache_hits: int
    failures: int
    wall_time: float
    front_sizes: list[int]
    best_per_objective: dict[str, float]
    worst_per_objective: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "evaluated": self.evaluated,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "wall_time": round(self.wall_time, 3),
            "n_fronts": len(self.front_sizes),
            "front_sizes": self.front_sizes,
            **{f"best_{k}": v for k, v in self.best_per_objective.items()},
            **{f"worst_{k}": v for k, v in self.worst_per_objective.items()},
        }


@dataclass
class NSGA2Config:
    population_size: int = 8
    #: Apply per-generation min-max normalisation to the objectives before
    #: selection.
    #:
    #: Note this CANNOT change the Pareto fronts: min-max is a strictly
    #: increasing per-objective transform and dominance is invariant under
    #: such transforms. Its purpose is to put objectives of very different
    #: scale on a common footing -- f2 is an unbounded loss while f1 is capped
    #: at ln 2 -- which matters for crowding distance on boundary cases and for
    #: reading the numbers. Raw values are what get recorded.
    normalise_objectives: bool = False
    generations: int = 2
    crossover_probability: float = 0.9
    #: ``None`` -> 1 / n_genes, the standard default.
    mutation_probability: float | None = None
    p_active: float = 0.5
    seed: int = 42


@dataclass
class NSGA2Result:
    population: list[Chromosome]
    objectives: list[Objectives]
    rank: list[int]
    distance: list[float]
    history: list[GenerationRecord] = field(default_factory=list)

    def pareto_front(self) -> list[int]:
        """Indices of the final non-dominated set."""
        return [i for i, r in enumerate(self.rank) if r == 0]


OBJECTIVE_NAMES = ("obj1_js", "obj2_retain_loss", "obj3_edit_cost")


def record_from_dict(payload: dict[str, Any]) -> "GenerationRecord":
    """Rebuild a GenerationRecord from its saved dict form (for resume)."""
    sizes = payload.get("front_sizes")
    if isinstance(sizes, str):
        sizes = [int(x) for x in sizes.split("|") if x]
    return GenerationRecord(
        generation=payload["generation"],
        evaluated=payload.get("evaluated", 0),
        cache_hits=payload.get("cache_hits", 0),
        failures=payload.get("failures", 0),
        wall_time=payload.get("wall_time", 0.0),
        front_sizes=list(sizes or []),
        best_per_objective={
            k[len("best_"):]: v for k, v in payload.items() if k.startswith("best_")
        },
        worst_per_objective={
            k[len("worst_"):]: v for k, v in payload.items() if k.startswith("worst_")
        },
    )


class NSGA2:
    """NSGA-II over :class:`Chromosome`, with an injected objective function.

    The ``evaluate`` callback receives a list of chromosomes and returns a list
    of objective tuples in the same order. Batching the whole generation into
    one call lets the caller cache, log and report per-individual timings
    without the algorithm knowing anything about the SEC.
    """

    def __init__(
        self,
        bounds: ChromosomeBounds,
        config: NSGA2Config,
        evaluate: EvaluateFn,
    ) -> None:
        self.bounds = bounds
        self.config = config
        self.evaluate = evaluate
        self.rng = np.random.default_rng(config.seed)
        self.mutation_probability = (
            config.mutation_probability
            if config.mutation_probability is not None
            else 1.0 / bounds.n_genes
        )

    # -- population ---------------------------------------------------------

    def initial_population(self) -> list[Chromosome]:
        """Random population, with the identity chromosome forced in as seed 0.

        Including the identity is deliberate: it is the only individual whose
        objectives are known in advance (``obj1 = 1``, ``obj2 = 0``), so if the
        search ever reports something else for it, the plumbing is broken rather
        than the search being interesting.
        """
        population = [Chromosome.identity(self.bounds)]
        while len(population) < self.config.population_size:
            population.append(
                Chromosome.random(self.bounds, self.rng, p_active=self.config.p_active)
            )
        return population

    def make_offspring(
        self, population: list[Chromosome], rank: list[int], distance: list[float]
    ) -> list[Chromosome]:
        parents = binary_tournament(
            self.rng, self.config.population_size, rank, distance
        )
        offspring: list[Chromosome] = []
        for i in range(0, self.config.population_size, 2):
            a = population[parents[i]]
            b = population[parents[(i + 1) % self.config.population_size]]
            child_a, child_b = uniform_crossover(
                a, b, self.rng, self.config.crossover_probability
            )
            for child in (child_a, child_b):
                if len(offspring) < self.config.population_size:
                    offspring.append(
                        random_reset_mutation(
                            child, self.rng, self.mutation_probability
                        )
                    )
        return offspring

    # -- survival -----------------------------------------------------------

    def _for_selection(self, objectives: list[Objectives]) -> list[Objectives]:
        """Objectives as selection should see them.

        Normalisation is applied here rather than at evaluation time so the RAW
        values are what reach the result files. Per-generation normalised values
        are not comparable across generations -- the range is recomputed each
        time -- so recording them would make convergence unreadable.
        """
        if not self.config.normalise_objectives:
            return objectives
        return [tuple(row) for row in normalise_objectives(objectives)]

    def select_survivors(
        self, objectives: list[Objectives]
    ) -> tuple[list[int], list[int], list[float]]:
        """Elitist (mu + lambda) truncation to ``population_size``.

        Fronts are admitted whole while they fit; the first front that overflows
        is sorted by crowding distance and truncated. Returns the surviving
        indices together with their rank and distance, so the caller does not
        have to recompute either.
        """
        scored = self._for_selection(objectives)
        fronts = fast_non_dominated_sort(scored)

        survivors: list[int] = []
        rank_of: dict[int, int] = {}
        distance_of: dict[int, float] = {}

        for rank, front in enumerate(fronts):
            front_distance = crowding_distance([scored[i] for i in front])
            for index, member in enumerate(front):
                rank_of[member] = rank
                distance_of[member] = front_distance[index]

            room = self.config.population_size - len(survivors)
            if room <= 0:
                break
            if len(front) <= room:
                survivors.extend(front)
                continue

            ordered = sorted(front, key=lambda i: distance_of[i], reverse=True)
            survivors.extend(ordered[:room])
            break

        return (
            survivors,
            [rank_of[i] for i in survivors],
            [distance_of[i] for i in survivors],
        )

    # -- the loop -----------------------------------------------------------

    def run(
        self,
        on_generation: Callable[[int, list, list], None] | None = None,
        resume_from: "tuple[int, list, list, list] | None" = None,
    ):
        """Run the search, optionally continuing from a saved state.

        ``resume_from`` is ``(generation, population, objectives, history)``
        from a checkpoint. The caller is responsible for having restored the RNG
        state onto ``self.rng`` first -- without that the resumed run draws a
        different stream and is no longer the same experiment.
        """
        import time

        if resume_from is None:
            population = self.initial_population()
            objectives = self.evaluate(population)

            survivors, rank, distance = self.select_survivors(objectives)
            population = [population[i] for i in survivors]
            objectives = [objectives[i] for i in survivors]

            history: list[GenerationRecord] = []
            history.append(self._record(0, len(population), objectives, 0.0))
            # Exposed so a checkpointing callback can persist progress so far;
            # the callback fires before run() returns, so it cannot reach the
            # local otherwise.
            self.history = history
            if on_generation:
                on_generation(0, population, objectives)
            first_generation = 1
        else:
            done, population, objectives, history = resume_from
            self.history = history
            # rank and distance are pure functions of the objectives, so they
            # are recomputed rather than stored -- one less thing to keep in
            # sync with the checkpoint format.
            survivors, rank, distance = self.select_survivors(objectives)
            population = [population[i] for i in survivors]
            objectives = [objectives[i] for i in survivors]
            first_generation = done + 1

        for generation in range(first_generation, self.config.generations + 1):
            started = time.perf_counter()

            offspring = self.make_offspring(population, rank, distance)
            offspring_objectives = self.evaluate(offspring)

            combined = population + offspring
            combined_objectives = objectives + offspring_objectives

            survivors, rank, distance = self.select_survivors(combined_objectives)
            population = [combined[i] for i in survivors]
            objectives = [combined_objectives[i] for i in survivors]

            history.append(
                self._record(
                    generation, len(offspring), objectives,
                    time.perf_counter() - started,
                )
            )
            if on_generation:
                on_generation(generation, population, objectives)

        return NSGA2Result(
            population=population,
            objectives=objectives,
            rank=rank,
            distance=distance,
            history=history,
        )

    @staticmethod
    def _record(
        generation: int, evaluated: int, objectives: list[Objectives], wall: float
    ) -> GenerationRecord:
        fronts = fast_non_dominated_sort(objectives)
        columns = list(zip(*objectives)) if objectives else [()] * 3
        return GenerationRecord(
            generation=generation,
            evaluated=evaluated,
            cache_hits=0,
            failures=0,
            wall_time=wall,
            front_sizes=[len(f) for f in fronts],
            best_per_objective={
                name: round(min(column), 6)
                for name, column in zip(OBJECTIVE_NAMES, columns)
            },
            worst_per_objective={
                name: round(max(column), 6)
                for name, column in zip(OBJECTIVE_NAMES, columns)
            },
        )


def objective_matrix(results: Iterable[Any]) -> list[Objectives]:
    """Pull ``(obj1, obj2, obj3)`` out of a sequence of SEC results."""
    return [(float(r.obj1), float(r.obj2), float(r.obj3)) for r in results]
