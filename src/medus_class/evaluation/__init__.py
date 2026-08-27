"""Objectives, measurement, and the class-unlearning evaluator."""

from medus_class.evaluation.anchor import (
    AnchorMetrics,
    AnchorMiaResult,
    anchor_composite,
    anchor_metrics_from_accuracies,
    anchor_mia,
)

from medus_class.evaluation.evaluator import (
    PENALTY_OBJECTIVES,
    ClassEvaluator,
    ClassResult,
)
from medus_class.evaluation.metrics import EvalOutput, evaluate, normalise_objectives
from medus_class.evaluation.objectives import (
    JS_MAX_NATS,
    js_to_reference,
    kl_to_reference,
    reference_logits,
    relative_parameter_delta,
    selectivity,
)

__all__ = [
    "AnchorMetrics", "AnchorMiaResult", "anchor_composite",
    "anchor_metrics_from_accuracies", "anchor_mia",
    "PENALTY_OBJECTIVES", "ClassEvaluator", "ClassResult",
    "EvalOutput", "evaluate", "normalise_objectives",
    "JS_MAX_NATS", "js_to_reference", "kl_to_reference", "reference_logits",
    "relative_parameter_delta", "selectivity",
]
