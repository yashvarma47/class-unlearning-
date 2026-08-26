"""Decode a chromosome into an executable, layer-wise unlearning strategy.

The decoder is the contract between the genome and the operators. It applies
exactly the rules agreed for MED-US, for each layer group ``i`` in **forward
order** (``stem``, ``layer1``, ..., ``fc``):

* ``b_i = 0`` -> the group is frozen/inactive. ``g_i``, ``s_i``, ``d_g,i`` and
  ``d_s,i`` are ignored entirely (they stay in the genome as latent material).
* ``b_i = 1`` and ``d_g,i > 0`` -> apply editor operator ``g_i`` at intensity
  ``d_g,i``.
* ``b_i = 1`` and ``d_s,i > 0`` -> apply smoother operator ``s_i`` at intensity
  ``d_s,i``.
* Execution order is fixed: **gradient first, smoothing second**, within each
  group.
* ``d_g,i = 0`` skips the gradient step; ``d_s,i = 0`` skips the smoothing step.

Ordering note
-------------
Actions are emitted **group-major**: all of group 0's actions, then all of
group 1's, and so on. The alternative (all gradient actions across every group,
then all smoothing actions) would give different results, because operators are
applied sequentially to a model that each one changes. Group-major follows the
"for each layer group ``i``" phrasing of the specification and keeps a group's
two actions adjacent, so a group is fully processed before the next begins.

Canonical form
--------------
Latent genes mean many genomes decode to the same strategy: if ``b_i = 0``, the
values of ``g_i``/``d_g,i`` are irrelevant, and if ``d_g,i = 0`` then ``g_i`` is
irrelevant too. :meth:`DecodedStrategy.canonical_key` collapses those
irrelevant genes so that equivalent genomes share one cache entry. With ~600 SEC
evaluations at minutes apiece, that deduplication is a real saving, not a
micro-optimisation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from medus_class.search.genome import Chromosome
from medus_class.operators.registry import LEVEL_OFF, describe, load_lookup, resolve_hparams


@dataclass(frozen=True)
class Action:
    """One operator application to one layer group.

    Attributes
    ----------
    group_index:
        Position in the chromosome, i.e. which gene produced this action.
    group_name:
        Layer-group name, e.g. ``"layer3"``.
    family:
        ``"editor"`` or ``"smoother"``.
    operator_id:
        Gene value selecting the operator within its family.
    operator_name:
        Resolved operator name, e.g. ``"MASK"``.
    level:
        Ordinal intensity, ``1..5`` (never ``0`` -- level 0 produces no action).
    level_name:
        Human-readable level, e.g. ``"MEDIUM"``.
    hparams:
        Concrete hyperparameters resolved from the lookup table.
    """

    group_index: int
    group_name: str
    family: str
    operator_id: int
    operator_name: str
    level: int
    level_name: str
    hparams: dict[str, Any]

    def describe(self) -> str:
        rendered = ", ".join(f"{k}={v}" for k, v in self.hparams.items())
        return (
            f"{self.group_name}: {self.operator_name} "
            f"[{self.level_name}] ({rendered})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "group_name": self.group_name,
            "family": self.family,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "level": self.level,
            "level_name": self.level_name,
            "hparams": self.hparams,
        }


@dataclass
class DecodedStrategy:
    """An ordered, executable plan produced from one chromosome."""

    actions: list[Action]
    active_groups: list[str]
    chromosome: Chromosome = field(repr=False)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    @property
    def is_no_op(self) -> bool:
        """``True`` when the strategy leaves the model untouched."""
        return not self.actions

    @property
    def n_editor_actions(self) -> int:
        return sum(1 for a in self.actions if a.family == "editor")

    @property
    def n_smoother_actions(self) -> int:
        return sum(1 for a in self.actions if a.family == "smoother")

    def canonical_form(self) -> list[tuple[int, str, int, int]]:
        """Minimal representation of what will actually be executed.

        Each entry is ``(group_index, family, operator_id, level)``. Genes that
        the decoder ignores never appear, so two genomes that behave identically
        produce identical canonical forms.
        """
        return [
            (a.group_index, a.family, a.operator_id, a.level) for a in self.actions
        ]

    def canonical_key(self) -> str:
        """Stable hash of the canonical form, used as the objective-cache key.

        SHA-256 of the JSON-encoded canonical form. Python's builtin ``hash`` is
        salted per process and would not survive a restart, which the cache must.
        """
        payload = json.dumps(self.canonical_form(), separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def estimated_epochs(self) -> float:
        """Total operator epochs -- a cheap proxy for evaluation cost.

        Useful for reporting how expensive the Pareto front's strategies are,
        and for spotting a population drifting towards costly chromosomes.
        """
        return float(sum(a.hparams.get("epochs", 1) for a in self.actions))

    def describe(self) -> str:
        """Multi-line human-readable plan, logged with every SEC evaluation."""
        if self.is_no_op:
            return "no-op strategy (no active group has a non-zero intensity)"
        lines = [
            f"strategy {self.canonical_key()} -- {len(self.actions)} action(s) "
            f"across {len(self.active_groups)} active group(s):"
        ]
        lines += [f"  {i + 1}. {a.describe()}" for i, a in enumerate(self.actions)]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_key": self.canonical_key(),
            "n_actions": len(self.actions),
            "active_groups": self.active_groups,
            "estimated_epochs": self.estimated_epochs(),
            "actions": [a.to_dict() for a in self.actions],
            "chromosome": self.chromosome.to_dict(),
        }


def decode(chromosome: Chromosome, group_names: Sequence[str]) -> DecodedStrategy:
    """Turn a chromosome into an ordered list of executable actions.

    Parameters
    ----------
    chromosome:
        The genome to decode.
    group_names:
        Layer-group names in forward order, from the model's
        :class:`~medus_class.models.layer_groups.LayerGroupRegistry`. Its length must
        equal ``L``.

    Raises
    ------
    ValueError
        If ``group_names`` does not have length ``L``.
    """
    if len(group_names) != chromosome.bounds.n_groups:
        raise ValueError(
            f"expected {chromosome.bounds.n_groups} group names (L), "
            f"got {len(group_names)}"
        )

    level_names = load_lookup()["intensity_levels"]
    actions: list[Action] = []
    active_groups: list[str] = []

    for index in range(chromosome.bounds.n_groups):
        if chromosome.b[index] == 0:
            # Inactive group: every other gene here is latent and ignored.
            continue

        group_name = group_names[index]
        active_groups.append(group_name)

        # Fixed order: gradient first, then smoothing.
        bounds = chromosome.bounds
        for family, operator_gene, level_gene, to_operator_id in (
            ("editor", chromosome.g, chromosome.d_g, bounds.editor_operator_id),
            ("smoother", chromosome.s, chromosome.d_s, bounds.smoother_operator_id),
        ):
            level = int(level_gene[index])
            if level == LEVEL_OFF:
                continue

            # The gene indexes into the selectable-ID tuple; it is not itself a
            # lookup-table ID (implemented operators are not contiguous).
            operator_id = to_operator_id(int(operator_gene[index]))
            hparams = resolve_hparams(family, operator_id, level)
            operator_name = describe(family, operator_id, level).split("[")[0]

            actions.append(
                Action(
                    group_index=index,
                    group_name=group_name,
                    family=family,
                    operator_id=operator_id,
                    operator_name=operator_name,
                    level=level,
                    level_name=level_names[level],
                    hparams=hparams,
                )
            )

    return DecodedStrategy(
        actions=actions, active_groups=active_groups, chromosome=chromosome
    )
