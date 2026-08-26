"""Operator registry: gene value + intensity level -> executable operator.

Adapted from the predecessor project. Two families remain, renamed to say what
they do rather than how they were once implemented:

============  ==========================================  ====================
family        operators                                   chromosome channel
============  ==========================================  ====================
``editor``    MASK, PRUNE, RANDOM_PRUNE                   ``g``
``smoother``  DAMP, NOISE, CLIP, QUANTIZE, RESET          ``s``
============  ==========================================  ====================

The old names were ``gradient`` and ``smoothing``, which became misleading once
the library went gradient-free: every operator here is a direct weight edit and
none of them computes a gradient.

Intensity semantics
-------------------
``0`` is ``OFF`` and means *skip this operator entirely*; it never indexes the
lookup table. Levels ``1..5`` map onto ``levels[0..4]``, so the off-by-one is
handled here, once.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from medus_class.operators.base import UnlearningOperator
from medus_class.operators.gradient_free import EDITOR_OPERATORS, SMOOTHER_OPERATORS
from medus_class.utils.config import load_config

#: Intensity level meaning ``skip``. Never used as a lookup index.
LEVEL_OFF = 0
#: Highest valid intensity level.
MAX_LEVEL = 5

#: family -> (implementations by name, lookup-table section)
FAMILIES: dict[str, tuple[dict[str, Any], str]] = {
    "editor": (EDITOR_OPERATORS, "editor_operators"),
    "smoother": (SMOOTHER_OPERATORS, "smoother_operators"),
}


def _family(family: str) -> tuple[dict[str, Any], str]:
    if family not in FAMILIES:
        raise KeyError(
            f"unknown operator family '{family}'; expected one of {sorted(FAMILIES)}"
        )
    return FAMILIES[family]


@lru_cache(maxsize=1)
def load_lookup(path: str = "operators/lookup.yaml") -> dict[str, Any]:
    """Load and cache the operator lookup table."""
    return load_config(path)


def operator_spec(family: str, operator_id: int) -> dict[str, Any]:
    """The lookup-table entry for one operator.

    Raises
    ------
    KeyError
        If ``operator_id`` is not a valid gene value for this family. Operator
        ids are contiguous from 0 precisely so a gene can be validated this way.
    """
    _, key = _family(family)
    table = load_lookup()[key]
    if operator_id not in table:
        raise KeyError(
            f"operator_id {operator_id} is not valid for family '{family}'; "
            f"valid ids are {sorted(table)}"
        )
    return table[operator_id]


def n_operators(family: str) -> int:
    """How many operators the family defines -- the gene's cardinality."""
    _, key = _family(family)
    return len(load_lookup()[key])


def resolve_hparams(family: str, operator_id: int, level: int) -> dict[str, Any]:
    """Concrete hyperparameters for one (operator, intensity level) pair.

    Raises
    ------
    ValueError
        If ``level`` is ``LEVEL_OFF`` -- callers must check for OFF and skip the
        operator rather than asking for its hyperparameters -- or out of range.
    """
    if level == LEVEL_OFF:
        raise ValueError(
            "level 0 means OFF; the caller must skip the operator rather than "
            "resolve hyperparameters for it"
        )
    if not 1 <= level <= MAX_LEVEL:
        raise ValueError(f"intensity level must be in 0..{MAX_LEVEL}, got {level}")

    spec = operator_spec(family, operator_id)
    levels = spec["levels"]
    if level > len(levels):
        raise ValueError(
            f"operator {spec['name']} defines {len(levels)} levels but level "
            f"{level} was requested"
        )
    return dict(levels[level - 1])


def is_selectable(family: str, operator_id: int) -> bool:
    """Whether the search may choose this operator."""
    return bool(operator_spec(family, operator_id).get("selectable", True))


def build_operator(family: str, operator_id: int) -> UnlearningOperator:
    """Instantiate the operator class for a gene value."""
    implementations, _ = _family(family)
    spec = operator_spec(family, operator_id)
    name = spec["name"]

    if name not in implementations:
        raise KeyError(
            f"lookup table names operator '{name}' for family '{family}' but no "
            f"implementation is registered; available: {sorted(implementations)}"
        )
    return implementations[name]()


def describe(family: str, operator_id: int, level: int) -> str:
    """Human-readable ``NAME@LEVEL(hparams)``, for logs and audit trails."""
    spec = operator_spec(family, operator_id)
    if level == LEVEL_OFF:
        return f"{spec['name']}@OFF"
    hparams = resolve_hparams(family, operator_id, level)
    rendered = ", ".join(f"{k}={v}" for k, v in hparams.items())
    return f"{spec['name']}@{level}({rendered})"


def operator_ids(family: str) -> list[int]:
    """Every gene value the family defines."""
    _, key = _family(family)
    return sorted(load_lookup()[key])


def selectable_operator_ids(family: str) -> list[int]:
    """Gene values the search may use.

    Every operator in this library is selectable -- the unsafe ones were left
    out of the table entirely rather than disabled. The distinction is kept
    because the chromosome bounds are derived from it, and a future addition
    might legitimately want to be present-but-disabled.
    """
    return [i for i in operator_ids(family) if is_selectable(family, i)]


def operator_names(family: str) -> list[str]:
    """Operator names in gene order."""
    return [operator_spec(family, i)["name"] for i in operator_ids(family)]


def selectable_operator_names(family: str) -> list[str]:
    return [operator_spec(family, i)["name"] for i in selectable_operator_ids(family)]
