"""The class-unlearning evaluator: one chromosome in, three objectives out.

Replaces the predecessor project's SEC. Structurally similar -- build once,
evaluate many times, restore the model between candidates -- because that design
was sound and expensive to rebuild. Everything about *what is measured* is new.

What changed, and why
---------------------
============================  ==============================================
SEC (instance-level)          this evaluator (class-level)
============================  ==============================================
four objective modes          one: ``js_editcost``
``f1 = -min(L_f, ln C)`` or   ``f1 = JS(P_ref(D_f) || P(D_f))``
``|L_f - target|``
``f3 = KL`` (duplicated f2)   ``f3 = relative parameter delta``
one ``test_eval`` loader      ``D_f_test`` and ``D_r_test``, measured apart
MIA optionally an objective   MIA is a diagnostic and cannot become one
five selection rules          one: ``class_contrast``
============================  ==============================================

The single-mode design is deliberate. The predecessor accumulated four objective
modes because each was tried and kept, and the resulting branching made it
possible to run a search whose objectives did not mean what the report said they
meant. Here there is one formulation and no flag that changes it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from medus_class.data import (
    build_class_loaders,
    get_or_create_class_split,
    load_cifar10,
)
from medus_class.evaluation.metrics import EvalOutput, evaluate as evaluate_loader
from medus_class.evaluation.objectives import (
    js_to_reference,
    kl_to_reference,
    reference_logits,
    relative_parameter_delta,
    selectivity,
)
from medus_class.evaluation.privacy import PrivacyResult, compute_mia_auc
from medus_class.models import build_model, build_registry, load_checkpoint
from medus_class.operators import OperatorContext, build_operator
from medus_class.search import Chromosome, DecodedStrategy, decode
from medus_class.utils.config import resolve_path
from medus_class.utils.device import get_device
from medus_class.utils.seeding import seed_everything

#: Objective vector for a candidate whose evaluation raised. Large rather than
#: 1.0: f2 is an unbounded loss, so a failure scored at 1.0 would outrank a
#: genuinely bad candidate and be selected for.
PENALTY_OBJECTIVES = (1.0e6, 1.0e6, 1.0e6)


@dataclass
class ClassResult:
    """Everything one evaluation produced."""

    status: str                       # "ok" | "failed"
    canonical_action_key: str
    chromosome_flat: list[int]
    decoded_actions: list[dict[str, Any]]
    operator_names_used: list[str]

    checkpoint_path: str
    reference_path: str
    forget_class: int
    seed: int
    device: str

    # --- objectives, all minimised ---------------------------------------
    obj1_js: float
    obj2_retain_loss: float
    obj3_edit_cost: float

    # --- training-set measurements ---------------------------------------
    forget_train_acc: float = float("nan")
    forget_train_loss: float = float("nan")
    retain_train_acc: float = float("nan")
    retain_train_loss: float = float("nan")

    # --- held-out measurements -------------------------------------------
    #: The headline result: unseen members of the forgotten class.
    forget_test_acc: float = float("nan")
    forget_test_loss: float = float("nan")
    retain_test_acc: float = float("nan")
    retain_test_loss: float = float("nan")

    # --- diagnostics, never objectives -----------------------------------
    kl_to_reference: float = float("nan")
    selectivity_S: float = float("nan")
    mia_auc: float = float("nan")

    runtime_seconds: float = 0.0
    operator_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def objectives(self) -> tuple[float, float, float]:
        """``(f1, f2, f3)`` in NSGA-II order."""
        return (self.obj1_js, self.obj2_retain_loss, self.obj3_edit_cost)

    def to_row(self) -> dict[str, Any]:
        row = {k: v for k, v in asdict(self).items()
               if k not in ("decoded_actions", "operator_results", "chromosome_flat")}
        row["operators"] = "|".join(self.operator_names_used)
        row["chromosome"] = " ".join(str(g) for g in self.chromosome_flat)
        return row


class ClassEvaluator:
    """Evaluates class-unlearning chromosomes against a fixed original model.

    Construction is the expensive part -- CIFAR-10, the loaders, two checkpoints
    and the cached reference logits -- so build once and call :meth:`evaluate`
    repeatedly.
    """

    def __init__(self, cfg: dict[str, Any], device: str | None = None) -> None:
        eval_cfg = cfg["evaluation"]
        data_cfg = dict(cfg["data"])
        model_cfg = cfg["model"]
        split_cfg = cfg["split"]

        self.cfg = cfg
        self.seed = int(cfg.get("seed", 42))
        seed_everything(self.seed, deterministic=cfg.get("deterministic", True))

        info = get_device(prefer=cfg["device"]["prefer"], index=cfg["device"]["index"])
        self.device = device if device is not None else info.device

        # Worker processes are pure overhead for these evaluation subsets, and on
        # Windows each loader iteration re-spawns them.
        if eval_cfg.get("num_workers") is not None:
            data_cfg["num_workers"] = int(eval_cfg["num_workers"])
            data_cfg["persistent_workers"] = False

        # --- data --------------------------------------------------------
        bundle = load_cifar10({**data_cfg, "download": False})
        self.split, _ = get_or_create_class_split(
            train_labels=bundle.train_labels,
            test_labels=bundle.test_labels,
            forget_class=int(split_cfg["forget_class"]),
            path=split_cfg["split_file"],
        )
        self.forget_class = self.split.forget_class

        self.loaders = build_class_loaders(
            bundle,
            self.split,
            data_cfg,
            seed=self.seed,
            batch_size_key=eval_cfg.get("batch_size_key", "train"),
            forget_subset_size=eval_cfg.get("forget_subset_size"),
            retain_subset_size=eval_cfg.get("retain_subset_size"),
        )

        # --- the model being unlearned FROM -------------------------------
        self.checkpoint_path = str(eval_cfg["checkpoint"])
        self.model = build_model(model_cfg, num_classes=int(data_cfg["num_classes"]))
        load_checkpoint(self.checkpoint_path, self.model, map_location="cpu")
        self._original_state = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        self.model.to(self.device)
        self.registry = build_registry(self.model, model_cfg)

        # --- the reference -------------------------------------------------
        self.reference_path = str(eval_cfg["reference_checkpoint"])
        path = resolve_path(self.reference_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"reference checkpoint not found: {path}. Train it with "
                f"experiments/train_class_reference.py, and point this at the "
                f"'_best_dr' file -- the one selected on D_r_test."
            )
        reference = build_model(model_cfg, num_classes=int(data_cfg["num_classes"]))
        load_checkpoint(path, reference, map_location="cpu")
        reference.to(self.device).eval()
        # Computed once: the reference never changes, and recomputing per
        # candidate would double the cost of f1.
        self._reference_logits = reference_logits(
            reference, self.loaders.forget_eval, self.device
        )
        self._reference = reference

        self.batch_cap = eval_cfg.get("batch_cap")
        self.selection_rule = eval_cfg.get("selection_rule", "class_contrast")
        self.operator_allowlist = eval_cfg.get("operator_allowlist")
        self.max_level = eval_cfg.get("max_level")
        self.compute_mia = bool(eval_cfg.get("compute_mia", True))

        # --- baselines, measured once on the pristine model ---------------
        self.original = self.measure_sets(self.model)
        self.reference_metrics = self.measure_sets(reference)

    # -- internals --------------------------------------------------------

    def _reset_model(self) -> nn.Module:
        """Restore the working model to theta_original.

        Reloads weights *and* buffers, so BatchNorm running statistics changed by
        a previous candidate are undone too. The checkpoint file is never
        touched; this restores from the in-memory cache.
        """
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in self._original_state.items()},
            strict=True,
        )
        self.model.to(self.device)
        return self.model

    def measure_sets(
        self, model: nn.Module, keep_per_sample: bool = False
    ) -> dict[str, float]:
        """Accuracy and loss on all four sets.

        ``keep_per_sample`` retains the per-sample arrays for the two
        forget-class sets, which the MIA needs. Off by default: ``D_r`` holds
        45 000 samples and the arrays are pure overhead when nothing reads them.
        """
        forget_train: EvalOutput = evaluate_loader(
            model, self.loaders.forget_eval, self.device,
            collect_per_sample=keep_per_sample)
        retain_train: EvalOutput = evaluate_loader(
            model, self.loaders.retain_eval, self.device, collect_per_sample=False)
        forget_test: EvalOutput = evaluate_loader(
            model, self.loaders.forget_test, self.device,
            collect_per_sample=keep_per_sample)
        retain_test: EvalOutput = evaluate_loader(
            model, self.loaders.retain_test, self.device, collect_per_sample=False)

        if keep_per_sample:
            self._last_forget_train = forget_train
            self._last_forget_test = forget_test

        return {
            "forget_train_acc": forget_train.accuracy,
            "forget_train_loss": forget_train.loss,
            "retain_train_acc": retain_train.accuracy,
            "retain_train_loss": retain_train.loss,
            "forget_test_acc": forget_test.accuracy,
            "forget_test_loss": forget_test.loss,
            "retain_test_acc": retain_test.accuracy,
            "retain_test_loss": retain_test.loss,
        }

    def _execute(self, model: nn.Module, strategy: DecodedStrategy) -> list[Any]:
        """Apply the decoded actions in order, group-major, to ``model`` in place."""
        normalize = self.cfg["data"]["normalize"]
        context = OperatorContext(
            loaders=self.loaders,
            registry=self.registry,
            device=self.device,
            num_classes=int(self.cfg["data"]["num_classes"]),
            seed=self.seed,
            batch_cap=self.batch_cap,
            normalize_mean=tuple(normalize["mean"]),
            normalize_std=tuple(normalize["std"]),
            selection_rule=self.selection_rule,
        )

        results = []
        for action in strategy.actions:
            operator = build_operator(action.family, action.operator_id)
            group = self.registry[action.group_index]
            # Snapshot a group before anything touches it, so RESET can restore
            # the pre-edit values even when a second operator runs on the same
            # group afterwards.
            if group.name not in context.group_snapshots:
                context.take_snapshot(model, group)
            results.append(operator.apply(model, context, group, action.hparams))
        return results

    # -- the public call ---------------------------------------------------

    def evaluate(self, chromosome: Chromosome) -> ClassResult:
        """Execute one strategy and measure it. The model is reset first."""
        started = time.perf_counter()
        strategy = decode(chromosome, self.registry.names)

        base = dict(
            canonical_action_key=json.dumps(
                strategy.canonical_form(), separators=(",", ":")),
            chromosome_flat=[int(g) for g in chromosome.to_vector()],
            decoded_actions=[a.to_dict() for a in strategy.actions],
            operator_names_used=sorted({a.operator_name.split("@")[0]
                                        for a in strategy.actions}),
            checkpoint_path=self.checkpoint_path,
            reference_path=self.reference_path,
            forget_class=self.forget_class,
            seed=self.seed,
            device=self.device,
        )

        try:
            model = self._reset_model()
            operator_results = self._execute(model, strategy)
            result = self._measure(model, base, operator_results, started)
        except Exception as exc:  # noqa: BLE001 -- a failed candidate is data
            return ClassResult(
                status="failed",
                obj1_js=PENALTY_OBJECTIVES[0],
                obj2_retain_loss=PENALTY_OBJECTIVES[1],
                obj3_edit_cost=PENALTY_OBJECTIVES[2],
                runtime_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
                **base,
            )
        return result

    def _measure(
        self,
        model: nn.Module,
        base: dict[str, Any],
        operator_results: list[Any],
        started: float,
    ) -> ClassResult:
        """Forward-pass evaluation of the edited model, and the objectives."""
        measured = self.measure_sets(model, keep_per_sample=self.compute_mia)

        obj1 = js_to_reference(
            model, self.loaders.forget_eval, self._reference_logits, self.device)
        obj2 = float(measured["retain_train_loss"])
        obj3 = relative_parameter_delta(model, self._original_state)

        # An objective that is not a real number cannot be ranked: every
        # comparison against nan is False, so a nan individual is neither
        # dominated nor dominating, and it would poison the min-max
        # normalisation for its whole generation.
        for name, value in (("f1", obj1), ("f2", obj2), ("f3", obj3)):
            if not math.isfinite(value):
                raise ValueError(
                    f"objective {name} is not finite ({value}); the model is "
                    f"numerically destroyed. forget_loss="
                    f"{measured['forget_train_loss']}, "
                    f"retain_loss={measured['retain_train_loss']}"
                )

        mia_auc = float("nan")
        if self.compute_mia:
            # Members are forget-class TRAINING images, non-members are
            # forget-class TEST images. That is the attack that matters for class
            # unlearning, and it is why it needs D_f_test -- the instance-level
            # construction (test set as non-members) cannot express it.
            #
            # Diagnostic only. It is deliberately impossible to make this an
            # objective: nothing reads it back.
            try:
                privacy: PrivacyResult = compute_mia_auc(
                    member_loss=self._last_forget_train.per_sample_loss,
                    member_confidence=self._last_forget_train.per_sample_confidence,
                    nonmember_loss=self._last_forget_test.per_sample_loss,
                    nonmember_confidence=self._last_forget_test.per_sample_confidence,
                )
                mia_auc = privacy.auc
            except Exception:  # noqa: BLE001 -- a diagnostic must never fail a run
                mia_auc = float("nan")

        return ClassResult(
            status="ok",
            obj1_js=obj1,
            obj2_retain_loss=obj2,
            obj3_edit_cost=obj3,
            kl_to_reference=kl_to_reference(
                model, self.loaders.forget_eval, self._reference_logits, self.device),
            selectivity_S=selectivity(
                measured["forget_train_loss"], measured["retain_train_loss"],
                self.original["forget_train_loss"], self.original["retain_train_loss"],
            ),
            mia_auc=mia_auc,
            runtime_seconds=time.perf_counter() - started,
            operator_results=[r.to_dict() for r in operator_results],
            **measured,
            **base,
        )
