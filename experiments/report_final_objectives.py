"""Extract the final objective values for reporting, at full fidelity.

Exists because two of the five rows a report needs are not on any front:

* **W_0** and **W_ref** are baselines, not candidates, so nothing ever computed
  their ``f1``.
* **C*_refined_bn_frozen** was produced by the refinement, whose ``record()``
  measured accuracies and losses but never called the objective function. Its
  ``f1`` is therefore absent from ``refinement.json``.

``f1`` is a Jensen-Shannon divergence between two predictive distributions over
``D_f``. It cannot be recovered from accuracy or loss -- a model's cross-entropy
on ``D_f`` says nothing about how its full ten-class distribution compares with
the reference's. So it is recomputed here with the same function the search used,
against the same cached reference logits over the same ordered 5 000 images.

Run::

    python experiments/report_final_objectives.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.metrics import evaluate as evaluate_loader  # noqa: E402
from medus_class.evaluation.objectives import (  # noqa: E402
    js_to_reference,
    kl_to_reference,
    relative_parameter_delta,
    selectivity,
)
from medus_class.evaluation.privacy import compute_mia_auc  # noqa: E402
from medus_class.models import build_model, load_checkpoint  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402


def spearman(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def measure_model(evaluator: ClassEvaluator, model) -> dict[str, Any]:
    """Every reported quantity for one model, at full fidelity."""
    model.to(evaluator.device).eval()

    forget_train = evaluate_loader(model, evaluator.loaders.forget_eval,
                                   evaluator.device, collect_per_sample=True)
    retain_train = evaluate_loader(model, evaluator.loaders.retain_eval,
                                   evaluator.device, collect_per_sample=False)
    forget_test = evaluate_loader(model, evaluator.loaders.forget_test,
                                  evaluator.device, collect_per_sample=True)
    retain_test = evaluate_loader(model, evaluator.loaders.retain_test,
                                  evaluator.device, collect_per_sample=False)

    try:
        mia = compute_mia_auc(
            member_loss=forget_train.per_sample_loss,
            member_confidence=forget_train.per_sample_confidence,
            nonmember_loss=forget_test.per_sample_loss,
            nonmember_confidence=forget_test.per_sample_confidence,
        ).auc
    except Exception:  # noqa: BLE001
        mia = float("nan")

    return {
        "f1_js": js_to_reference(model, evaluator.loaders.forget_eval,
                                 evaluator._reference_logits, evaluator.device),
        "f2_retain_train_loss": float(retain_train.loss),
        "f3_edit_cost": relative_parameter_delta(model, evaluator._original_state),
        "kl_to_reference": kl_to_reference(model, evaluator.loaders.forget_eval,
                                           evaluator._reference_logits,
                                           evaluator.device),
        "forget_train_acc": forget_train.accuracy,
        "forget_train_loss": forget_train.loss,
        "forget_test_acc": forget_test.accuracy,
        "forget_test_loss": forget_test.loss,
        "retain_train_acc": retain_train.accuracy,
        "retain_train_loss": retain_train.loss,
        "retain_test_acc": retain_test.accuracy,
        "retain_test_loss": retain_test.loss,
        "mia_auc": mia,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--front",
                        default="results/search/plan_a_frog/full_fidelity/"
                                "front_full_fidelity.csv")
    parser.add_argument("--refined",
                        default="results/search/plan_a_frog_bn_frozen_refined/"
                                "refined_best.pt")
    parser.add_argument("--out",
                        default="results/search/plan_a_frog/final_objectives.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True

    print("=" * 100)
    print("FINAL OBJECTIVE VALUES -- full fidelity")
    print("=" * 100)

    evaluator = ClassEvaluator(cfg)
    print(f"  loader sizes  {evaluator.loaders.sizes()}")
    print(f"  W_0           {Path(evaluator.checkpoint_path).name}")
    print(f"  W_ref         {Path(evaluator.reference_path).name}")

    rows: dict[str, dict[str, Any]] = {}

    # --- W_0: the model being unlearned FROM -----------------------------
    w0 = build_model(cfg["model"], num_classes=int(cfg["data"]["num_classes"]))
    load_checkpoint(evaluator.checkpoint_path, w0, map_location="cpu")
    rows["W_0"] = measure_model(evaluator, w0)
    rows["W_0"]["kind"] = "baseline (original model)"

    # --- W_ref: the retain-only reference --------------------------------
    # f1 must come out at exactly 0: it is being compared with itself.
    wref = build_model(cfg["model"], num_classes=int(cfg["data"]["num_classes"]))
    load_checkpoint(evaluator.reference_path, wref, map_location="cpu")
    rows["W_ref"] = measure_model(evaluator, wref)
    rows["W_ref"]["kind"] = "baseline (retain-only reference)"
    # f3 for W_ref is NOT an edit cost: W_ref is an independently trained model,
    # not an edit of W_0. The number is well defined but means something else,
    # so it is recorded under its own name.
    rows["W_ref"]["distance_from_W0_not_an_edit"] = rows["W_ref"].pop("f3_edit_cost")

    # --- the front rows --------------------------------------------------
    with resolve_path(args.front).open(encoding="utf-8-sig") as handle:
        front = list(csv.DictReader(handle))

    def front_row(position: int) -> dict[str, Any]:
        member = next(m for m in front
                      if int(m["front_position"]) == position)
        return {
            "f1_js": float(member["obj1_js"]),
            "f2_retain_train_loss": float(member["obj2_retain_loss"]),
            "f3_edit_cost": float(member["obj3_edit_cost"]),
            "kl_to_reference": float(member["kl_to_reference"]),
            "forget_train_acc": float(member["forget_train_acc"]),
            "forget_train_loss": float(member["forget_train_loss"]),
            "forget_test_acc": float(member["forget_test_acc"]),
            "forget_test_loss": float(member["forget_test_loss"]),
            "retain_train_acc": float(member["retain_train_acc"]),
            "retain_train_loss": float(member["retain_train_loss"]),
            "retain_test_acc": float(member["retain_test_acc"]),
            "retain_test_loss": float(member["retain_test_loss"]),
            "selectivity_S": float(member["selectivity_S"]),
            "mia_auc": float(member["mia_auc"]),
            "operators": member["operators"],
            "front_position": position,
        }

    best_s = max(front, key=lambda m: float(m["selectivity_S"]))
    best_s_position = int(best_s["front_position"])
    rows["best_S"] = front_row(best_s_position)
    rows["best_S"]["kind"] = "PURE GRADIENT-FREE (Pareto front member)"

    rows["C_star"] = front_row(8)
    rows["C_star"]["kind"] = "PURE GRADIENT-FREE (Pareto front member)"

    # --- the refined model -----------------------------------------------
    refined_path = resolve_path(args.refined)
    if refined_path.is_file():
        refined = build_model(cfg["model"],
                              num_classes=int(cfg["data"]["num_classes"]))
        load_checkpoint(refined_path, refined, map_location="cpu")
        rows["C_star_refined_bn_frozen"] = measure_model(evaluator, refined)
        rows["C_star_refined_bn_frozen"]["kind"] = (
            "POST-SEARCH REFINEMENT -- not a Pareto-front member, not part of "
            "the evolutionary search"
        )
        rows["C_star_refined_bn_frozen"]["operators"] = "DAMP|MASK + 2 gradient steps"
    else:
        print(f"\n  refined checkpoint not found: {refined_path}")

    # selectivity for anything measured here rather than read from the front
    original = rows["W_0"]
    for name, row in rows.items():
        if "selectivity_S" not in row:
            row["selectivity_S"] = selectivity(
                row["forget_train_loss"], row["retain_train_loss"],
                original["forget_train_loss"], original["retain_train_loss"],
            )

    # --- the table --------------------------------------------------------
    order = ["W_0", "W_ref", "best_S", "C_star", "C_star_refined_bn_frozen"]
    order = [n for n in order if n in rows]
    labels = {
        "W_0": "W_0 (original)",
        "W_ref": "W_ref (retain-only)",
        "best_S": f"best-S front member (#{best_s_position})",
        "C_star": "C* (front #8)",
        "C_star_refined_bn_frozen": "C*_refined_bn_frozen",
    }
    metrics = [
        ("f1_js", "f1  JS to W_ref"),
        ("f2_retain_train_loss", "f2  retain train loss"),
        ("f3_edit_cost", "f3  edit cost"),
        ("forget_train_acc", "D_f_train acc"),
        ("forget_train_loss", "D_f_train loss"),
        ("forget_test_acc", "D_f_test acc"),
        ("forget_test_loss", "D_f_test loss"),
        ("retain_train_acc", "D_r_train acc"),
        ("retain_train_loss", "D_r_train loss"),
        ("retain_test_acc", "D_r_test acc"),
        ("retain_test_loss", "D_r_test loss"),
        ("selectivity_S", "selectivity S"),
        ("mia_auc", "MIA AUC"),
        ("kl_to_reference", "(KL to W_ref, diag)"),
    ]

    print("\n" + "-" * 118)
    print(f"  {'metric':<24}" + "".join(f"{labels[n][:21]:>19}" for n in order))
    print("-" * 118)
    for key, label in metrics:
        line = f"  {label:<24}"
        for name in order:
            value = rows[name].get(key)
            line += f"{value:>19.6f}" if isinstance(value, float) else f"{'--':>19}"
        print(line)
    print(f"  {'type':<24}" + "".join(
        f"{('refined' if 'REFINEMENT' in rows[n]['kind'] else 'gradient-free' if 'GRADIENT-FREE' in rows[n]['kind'] else 'baseline'):>19}"
        for n in order))

    # --- objective correlations across the front -------------------------
    f1 = [float(m["obj1_js"]) for m in front]
    f2 = [float(m["obj2_retain_loss"]) for m in front]
    f3 = [float(m["obj3_edit_cost"]) for m in front]
    kl = [float(m["kl_to_reference"]) for m in front]

    correlations = {
        "f1_vs_f2": spearman(f1, f2),
        "f1_vs_f3": spearman(f1, f3),
        "f2_vs_f3": spearman(f2, f3),
        "f1_vs_KL": spearman(f1, kl),
        "f2_vs_KL": spearman(f2, kl),
    }
    print("\n" + "-" * 118)
    print(f"OBJECTIVE CORRELATIONS  (Spearman, {len(front)} full-fidelity front members)")
    print("-" * 118)
    for name, value in correlations.items():
        print(f"  {name:<12}{value:+.4f}")

    payload = {
        "config": args.config,
        "front": args.front,
        "refined_checkpoint": str(args.refined),
        "note_f1": (
            "f1 for W_0, W_ref and the refined model was RECOMPUTED here with "
            "js_to_reference over the full 5 000-image D_f, against the same "
            "cached reference logits the search used. It is not derivable from "
            "accuracy or loss."
        ),
        "rows": rows,
        "correlations_full_fidelity_front": correlations,
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
