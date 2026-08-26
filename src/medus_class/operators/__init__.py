"""The safe gradient-free operator library and the class-activation selector.

Eight operators across two chromosome channels. Every one is a direct weight
edit: nothing here computes a gradient, which is why the families are named
``editor`` and ``smoother`` rather than the predecessor's ``gradient`` and
``smoothing``.

REINIT and SIGN_FLIP are deliberately not part of this library. See
``configs/operators/lookup.yaml`` for why.
"""

from medus_class.operators.base import (
    OperatorContext,
    OperatorResult,
    UnlearningOperator,
    restrict_to_group,
)
from medus_class.operators.gradient_free import (
    EDITOR_OPERATORS,
    SMOOTHER_OPERATORS,
    MaskedWriteOperator,
)
from medus_class.operators.registry import (
    FAMILIES,
    LEVEL_OFF,
    MAX_LEVEL,
    build_operator,
    describe,
    is_selectable,
    load_lookup,
    n_operators,
    operator_ids,
    operator_names,
    operator_spec,
    resolve_hparams,
    selectable_operator_ids,
    selectable_operator_names,
)
from medus_class.operators.selection import ConnectionMask, select_connections

__all__ = [
    "OperatorContext", "OperatorResult", "UnlearningOperator", "restrict_to_group",
    "EDITOR_OPERATORS", "SMOOTHER_OPERATORS", "MaskedWriteOperator",
    "FAMILIES", "LEVEL_OFF", "MAX_LEVEL",
    "build_operator", "describe", "is_selectable", "load_lookup", "n_operators",
    "operator_ids", "operator_names", "operator_spec", "resolve_hparams",
    "selectable_operator_ids", "selectable_operator_names",
    "ConnectionMask", "select_connections",
]
