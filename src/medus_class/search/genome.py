"""Chromosome encoding for MED-US.

A candidate unlearning strategy is

.. math::

    x = (b, g, s, d_g, d_s)

where every component is an **integer vector of length L**, the number of layer
groups (``L = 6`` for both ResNet-18 and VGG19 in the first implementation).

======  ==========================================================  ============
Gene    Meaning                                                     Range
======  ==========================================================  ============
``b``   is layer group ``i`` active?                                ``{0, 1}``
``g``   which editor operator to apply to group ``i``             ``0..G-1``
``s``   which smoother operator to apply to group ``i``            ``0..S-1``
``d_g`` ordinal intensity of the editor operator                  ``0..5``
``d_s`` ordinal intensity of the smoother operator                 ``0..5``
======  ==========================================================  ============

Intensities are **ordinal levels**, not continuous values:
``0 = OFF, 1 = VERY_LOW, 2 = LOW, 3 = MEDIUM, 4 = HIGH, 5 = VERY_HIGH``.

Latent genes
------------
When ``b_i = 0`` the other four genes of group ``i`` are ignored by the decoder
but **kept in the genome**. They are latent genetic material: a later mutation
can flip ``b_i`` back on and recover a strategy the population already
discovered. The cost is that distinct genomes can decode to the same strategy,
which is why the decoder also produces a canonical form for caching (see
:mod:`medus.chromosome.decoder`).

The genome is integer-coded throughout, so NSGA-II uses uniform crossover and
random-reset mutation rather than SBX / polynomial mutation, which assume real
variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from medus_class.operators.registry import (
    MAX_LEVEL,
    n_operators,
    operator_spec,
    selectable_operator_ids,
)

#: The five gene vectors, in canonical order.
GENE_NAMES = ("b", "g", "s", "d_g", "d_s")


@dataclass(frozen=True)
class ChromosomeBounds:
    """Valid ranges for every gene, and hence the search space.

    Attributes
    ----------
    n_groups:
        ``L`` -- the number of layer groups, taken from the model's registry.
    editor_operator_ids:
        Lookup-table IDs selectable by the ``g`` gene, in order.
    smoother_operator_ids:
        Lookup-table IDs selectable by the ``s`` gene, in order. May be empty
        while no smoother operator is implemented, in which case the smoothing
        channel is disabled and ``d_s`` is forced to ``0``.
    max_level:
        Highest intensity level, ``5``.

    Notes
    -----
    Genes index **into** these ID tuples rather than holding lookup-table IDs
    directly. This indirection matters because selectable operators are not
    contiguous: after the Phase 6e atomicity cleanup the selectable gradient IDs
    are ``(0, 1, 3, 6, 7)`` and the selectable smoothing IDs are
    ``(1, 2, 4, 5, 6)``, with the gaps occupied by composite or unported entries
    kept for reference only. Storing a count instead of the IDs would let gene
    value ``0`` resolve to a non-selectable operator. The indirection also keeps
    lookup-table IDs stable as the library changes, so decoded strategies
    recorded in past results stay interpretable.
    """

    n_groups: int
    editor_operator_ids: tuple[int, ...]
    smoother_operator_ids: tuple[int, ...] = ()
    max_level: int = MAX_LEVEL

    def __post_init__(self) -> None:
        # Normalise to tuples so the dataclass stays hashable and immutable.
        object.__setattr__(self, "editor_operator_ids", tuple(self.editor_operator_ids))
        object.__setattr__(self, "smoother_operator_ids", tuple(self.smoother_operator_ids))

        if self.n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {self.n_groups}")
        if not self.editor_operator_ids:
            raise ValueError(
                "at least one editor operator must be available; none are "
                "implemented, so no chromosome could ever be executed"
            )

    @property
    def n_editor_operators(self) -> int:
        return len(self.editor_operator_ids)

    @property
    def n_smoother_operators(self) -> int:
        return len(self.smoother_operator_ids)

    @property
    def smoother_enabled(self) -> bool:
        """``False`` while no smoother operator is implemented."""
        return bool(self.smoother_operator_ids)

    def editor_operator_id(self, gene_value: int) -> int:
        """Map a ``g`` gene value to its lookup-table operator ID."""
        return self.editor_operator_ids[int(gene_value)]

    def smoother_operator_id(self, gene_value: int) -> int:
        """Map an ``s`` gene value to its lookup-table operator ID."""
        if not self.smoother_operator_ids:
            raise IndexError(
                "no smoother operator is implemented, so the s gene is not "
                "selectable; d_s should be pinned to 0"
            )
        return self.smoother_operator_ids[int(gene_value)]

    @property
    def n_genes(self) -> int:
        """Total genome length as a flat vector: ``5 * L``."""
        return 5 * self.n_groups

    @classmethod
    def from_registry(
        cls,
        n_groups: int,
        implemented_only: bool = True,
        allow: Sequence[str] | None = None,
        max_level: int = MAX_LEVEL,
    ) -> "ChromosomeBounds":
        """Build bounds from the operator lookup table.

        Parameters
        ----------
        implemented_only:
            If ``True`` (default), the search space is restricted to the
            operators marked ``selectable: true`` in the lookup table -- the
            atomic MED-US library. That excludes both declared-but-unported
            entries (which would fail partway through an expensive SEC
            evaluation) and the composite baseline references ``RL`` and
            ``L1_SPARSE``, which are runnable but are *methods*, not atomic
            operators, and so must never appear inside a chromosome.
        allow:
            Operator NAMES to keep, as a further restriction on top of
            ``selectable``. Used to narrow a search without demoting anything in
            the lookup table -- an experiment that excludes the most destructive
            operators is a property of that run, not of the library.

            Names not present in the selectable set are ignored rather than
            raising: an allowlist naming an operator from the other channel is a
            normal way to express "only these two here".
        max_level:
            Highest intensity the chromosome may express. Lowering it confines a
            search to the gentle end of every ladder, which is where S is
            measurable at all -- a candidate pinned at the forget-loss ceiling
            has no headroom left to trade.
        """
        if implemented_only:
            gradient = list(selectable_operator_ids("editor"))
            smoothing = list(selectable_operator_ids("smoother"))
        else:
            gradient = list(range(n_operators("editor")))
            smoothing = list(range(n_operators("smoother")))

        if allow is not None:
            wanted = set(allow)
            gradient = [
                i for i in gradient if operator_spec("editor", i)["name"] in wanted
            ]
            smoothing = [
                i for i in smoothing if operator_spec("smoother", i)["name"] in wanted
            ]
            if not gradient:
                raise ValueError(
                    "the operator allowlist leaves the gradient channel empty; "
                    "at least one operator must remain selectable there. "
                    f"allow={sorted(wanted)}"
                )

        return cls(
            n_groups=n_groups,
            editor_operator_ids=tuple(gradient),
            smoother_operator_ids=tuple(smoothing),
            max_level=max_level,
        )

    def gene_bounds(self, gene: str) -> tuple[int, int]:
        """Inclusive ``(low, high)`` range for one gene vector."""
        if gene == "b":
            return 0, 1
        if gene == "g":
            return 0, self.n_editor_operators - 1
        if gene == "s":
            return 0, max(self.n_smoother_operators - 1, 0)
        if gene in ("d_g", "d_s"):
            if gene == "d_s" and not self.smoother_enabled:
                return 0, 0
            return 0, self.max_level
        raise KeyError(f"unknown gene '{gene}'; expected one of {GENE_NAMES}")

    def summary(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "editor_operator_ids": list(self.editor_operator_ids),
            "smoother_operator_ids": list(self.smoother_operator_ids),
            "n_editor_operators": self.n_editor_operators,
            "n_smoother_operators": self.n_smoother_operators,
            "max_level": self.max_level,
            "smoother_enabled": self.smoother_enabled,
            "n_genes": self.n_genes,
            "search_space_size": self.search_space_size(),
        }

    def search_space_size(self) -> int:
        """Number of distinct genomes -- the raw size of the search space.

        Counts genomes, not strategies: latent genes mean the number of distinct
        *decoded strategies* is smaller.
        """
        per_group = 2 * self.n_editor_operators * (self.max_level + 1)
        if self.smoother_enabled:
            per_group *= self.n_smoother_operators * (self.max_level + 1)
        return per_group**self.n_groups


@dataclass
class Chromosome:
    """One candidate layer-wise unlearning strategy.

    All five vectors have length ``L``. Genes are plain Python ints held in
    ``numpy`` integer arrays so the genome can be handed to NSGA-II operators
    without conversion.
    """

    b: np.ndarray
    g: np.ndarray
    s: np.ndarray
    d_g: np.ndarray
    d_s: np.ndarray
    bounds: ChromosomeBounds

    def __post_init__(self) -> None:
        for name in GENE_NAMES:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.int64))
        self.validate()

    # -- construction ------------------------------------------------------

    @classmethod
    def identity(cls, bounds: ChromosomeBounds) -> "Chromosome":
        """The no-op chromosome: every group inactive, so nothing is applied.

        Decodes to zero actions, so the SEC must return exactly the original
        model's objectives. This is the sanity check that the SEC's plumbing --
        cloning, decoding, evaluation -- is correct before any operator runs.
        """
        zeros = np.zeros(bounds.n_groups, dtype=np.int64)
        return cls(
            b=zeros.copy(),
            g=zeros.copy(),
            s=zeros.copy(),
            d_g=zeros.copy(),
            d_s=zeros.copy(),
            bounds=bounds,
        )

    @classmethod
    def random(
        cls,
        bounds: ChromosomeBounds,
        rng: np.random.Generator,
        p_active: float = 0.5,
    ) -> "Chromosome":
        """Sample a random chromosome.

        Parameters
        ----------
        p_active:
            Probability that a given layer group is active (``b_i = 1``).
            Lower values bias the initial population towards sparse strategies
            that touch few groups, which are cheaper to evaluate and easier to
            interpret.

        Notes
        -----
        Latent genes are sampled for *every* group, including inactive ones, so
        that switching a group on later reveals a meaningful strategy rather
        than a default one.
        """
        n = bounds.n_groups
        return cls(
            b=(rng.random(n) < p_active).astype(np.int64),
            g=rng.integers(0, bounds.n_editor_operators, size=n),
            s=(
                rng.integers(0, bounds.n_smoother_operators, size=n)
                if bounds.smoother_enabled
                else np.zeros(n, dtype=np.int64)
            ),
            d_g=rng.integers(0, bounds.max_level + 1, size=n),
            d_s=(
                rng.integers(0, bounds.max_level + 1, size=n)
                if bounds.smoother_enabled
                else np.zeros(n, dtype=np.int64)
            ),
            bounds=bounds,
        )

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Check every gene lies within its bounds.

        Raises
        ------
        ValueError
            On a wrong-length vector or an out-of-range gene value.
        """
        for name in GENE_NAMES:
            vector = getattr(self, name)
            if vector.shape != (self.bounds.n_groups,):
                raise ValueError(
                    f"gene '{name}' has shape {vector.shape}, expected "
                    f"({self.bounds.n_groups},)"
                )
            low, high = self.bounds.gene_bounds(name)
            if vector.min() < low or vector.max() > high:
                raise ValueError(
                    f"gene '{name}' has values outside [{low}, {high}]: "
                    f"{vector.tolist()}"
                )

    def clip(self) -> "Chromosome":
        """Return a copy with every gene clipped into range.

        Used after variation operators, which can otherwise step outside the
        bounds.
        """
        clipped = {}
        for name in GENE_NAMES:
            low, high = self.bounds.gene_bounds(name)
            clipped[name] = np.clip(getattr(self, name), low, high)
        return Chromosome(bounds=self.bounds, **clipped)

    # -- flat-vector interface for NSGA-II ---------------------------------

    def to_vector(self) -> np.ndarray:
        """Flatten to a single integer vector of length ``5L``.

        Layout is ``[b | g | s | d_g | d_s]``, i.e. gene-major. Crossover
        therefore mixes whole gene types across groups; a group-major layout
        would instead favour keeping a group's five genes together. Gene-major
        is the more standard choice for integer-coded NSGA-II and keeps the
        ``b`` mask contiguous.
        """
        return np.concatenate([getattr(self, name) for name in GENE_NAMES])

    @classmethod
    def from_vector(
        cls, vector: Sequence[int] | np.ndarray, bounds: ChromosomeBounds
    ) -> "Chromosome":
        """Inverse of :meth:`to_vector`."""
        array = np.asarray(vector, dtype=np.int64)
        if array.size != bounds.n_genes:
            raise ValueError(
                f"expected a vector of length {bounds.n_genes} (5 x L), "
                f"got {array.size}"
            )
        chunks = np.split(array, 5)
        return cls(bounds=bounds, **dict(zip(GENE_NAMES, chunks)))

    def lower_bounds(self) -> np.ndarray:
        """Per-gene lower bounds as a flat vector, for variation operators."""
        return np.concatenate(
            [
                np.full(self.bounds.n_groups, self.bounds.gene_bounds(name)[0])
                for name in GENE_NAMES
            ]
        )

    def upper_bounds(self) -> np.ndarray:
        """Per-gene upper bounds as a flat vector, for variation operators."""
        return np.concatenate(
            [
                np.full(self.bounds.n_groups, self.bounds.gene_bounds(name)[1])
                for name in GENE_NAMES
            ]
        )

    # -- inspection --------------------------------------------------------

    @property
    def n_active_groups(self) -> int:
        """How many groups have ``b_i = 1``."""
        return int(self.b.sum())

    def is_no_op(self) -> bool:
        """``True`` if the chromosome decodes to no actions at all.

        A group contributes nothing when it is inactive, or when both intensity
        genes are OFF. Such chromosomes are legal and useful -- they express
        "leave this model alone" -- but the SEC can skip all execution for them.
        """
        return not any(
            self.b[i] == 1 and (self.d_g[i] > 0 or self.d_s[i] > 0)
            for i in range(self.bounds.n_groups)
        )

    def genes_for(self, index: int) -> dict[str, int]:
        """The five gene values of one layer group."""
        return {name: int(getattr(self, name)[index]) for name in GENE_NAMES}

    def __iter__(self) -> Iterator[dict[str, int]]:
        """Iterate over per-group gene dictionaries, in layer order."""
        for index in range(self.bounds.n_groups):
            yield self.genes_for(index)

    def copy(self) -> "Chromosome":
        return Chromosome(
            b=self.b.copy(),
            g=self.g.copy(),
            s=self.s.copy(),
            d_g=self.d_g.copy(),
            d_s=self.d_s.copy(),
            bounds=self.bounds,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, written into every SEC result file."""
        return {name: getattr(self, name).tolist() for name in GENE_NAMES}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in GENE_NAMES
        )

    def __repr__(self) -> str:
        return (
            f"Chromosome(b={self.b.tolist()}, g={self.g.tolist()}, "
            f"s={self.s.tolist()}, d_g={self.d_g.tolist()}, "
            f"d_s={self.d_s.tolist()})"
        )
