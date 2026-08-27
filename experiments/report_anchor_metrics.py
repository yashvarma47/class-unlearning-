"""Score the completed frog result under the anchor paper's protocol.

Anchor: Kodge, Saha & Roy, "Deep Unlearning: Fast and Efficient Gradient-free
Class Forgetting", TMLR 07/2024 -- see :mod:`medus_class.evaluation.anchor` for
where each formula was read out of their released code.

This script trains nothing and searches nothing. It measures four models that
already exist:

======================================  ================================================
``W_0``                                 ``results/checkpoints/cifar10_resnet18_seed42_best.pt``
``W_ref``                               ``results/checkpoints/class6_frog_reference_best_dr.pt``
``C_star``                              rebuilt from its stored chromosome
``C_star_refined_bn_frozen``            ``results/search/plan_a_frog_bn_frozen_refined/refined_best.pt``
======================================  ================================================

``C_star`` has no checkpoint of its own -- the search recorded genomes, not
weights -- so it is reconstructed by re-decoding the chromosome saved on its
Pareto-front row and re-executing the same gradient-free operators against the
same ``W_0``. That is deterministic weight surgery, not a search: the
reconstruction is checked against the objective values stored on that row and
the script refuses to report anything if they disagree.

Run::

    python experiments/report_anchor_metrics.py

The anchor's MIA fits an RBF SVC on |D_r_train| + |D_f_train| = 50 000 points,
which is O(n^2). ``--max-shadow-per-group N`` caps each group if that is not
affordable; doing so is a documented deviation and every output file says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.anchor import (  # noqa: E402
    AnchorMiaResult,
    anchor_metrics_from_accuracies,
    anchor_mia,
)
from medus_class.evaluation.metrics import evaluate as evaluate_loader  # noqa: E402
from medus_class.evaluation.objectives import (  # noqa: E402
    js_to_reference,
    kl_to_reference,
    relative_parameter_delta,
    selectivity,
)
from medus_class.evaluation.privacy import compute_mia_auc  # noqa: E402
from medus_class.models import build_model, load_checkpoint  # noqa: E402
from medus_class.search import Chromosome, ChromosomeBounds  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

#: How closely a rebuilt C* must match the objective values stored on its front
#: row. Operators are deterministic given the seed, so the only expected
#: difference is float non-associativity across a different GPU/CPU reduction
#: order.
REBUILD_TOLERANCE = 1e-4


def measure_model(
    evaluator: ClassEvaluator,
    model,
    max_shadow_per_group: int | None,
    seed: int,
) -> dict[str, Any]:
    """Anchor metrics and our existing metrics for one model, at full fidelity.

    The anchor columns come first in the returned mapping because they are the
    ones that go into their Table 1; everything after them is ours, kept so no
    previously reported number disappears from the record.
    """
    model.to(evaluator.device).eval()

    forget_train = evaluate_loader(model, evaluator.loaders.forget_eval,
                                   evaluator.device, collect_per_sample=True)
    retain_train = evaluate_loader(model, evaluator.loaders.retain_eval,
                                   evaluator.device, collect_per_sample=False)
    forget_test = evaluate_loader(model, evaluator.loaders.forget_test,
                                  evaluator.device, collect_per_sample=True)
    retain_test = evaluate_loader(model, evaluator.loaders.retain_test,
                                  evaluator.device, collect_per_sample=False)

    # --- the anchor protocol ---------------------------------------------
    # ACC_r and ACC_f are TEST-set quantities. Their Table 1 is a test-set
    # table, and forget-TRAIN accuracy is a different (easier) number.
    mia_result: AnchorMiaResult | None
    try:
        mia_result = anchor_mia(
            model,
            retain_train_loader=evaluator.loaders.retain_eval,
            forget_train_loader=evaluator.loaders.forget_eval,
            forget_test_loader=evaluator.loaders.forget_test,
            device=evaluator.device,
            max_shadow_per_group=max_shadow_per_group,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 -- a failed attack is data, not a crash
        print(f"      anchor MIA failed: {type(exc).__name__}: {exc}")
        mia_result = None

    anchor = anchor_metrics_from_accuracies(
        retain_test_accuracy=retain_test.accuracy,
        forget_test_accuracy=forget_test.accuracy,
        mia_result=mia_result,
    )

    # --- our metrics, unchanged ------------------------------------------
    try:
        mia_auc = compute_mia_auc(
            member_loss=forget_train.per_sample_loss,
            member_confidence=forget_train.per_sample_confidence,
            nonmember_loss=forget_test.per_sample_loss,
            nonmember_confidence=forget_test.per_sample_confidence,
        ).auc
    except Exception:  # noqa: BLE001
        mia_auc = float("nan")

    return {
        **anchor.to_dict(),
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
        "mia_auc": mia_auc,
    }


def rebuild_candidate(evaluator: ClassEvaluator, cfg: dict[str, Any],
                      member: dict[str, str]) -> Any:
    """Re-execute a front member's operator chain and verify it reproduced.

    Returns the edited model held by ``evaluator``. Raises if the recomputed
    objectives drift from the values stored on the front row, which would mean
    the model being scored is not the model that was reported.
    """
    bounds = ChromosomeBounds.from_registry(
        n_groups=len(evaluator.registry.names),
        implemented_only=True,
        max_level=cfg["evaluation"].get("max_level"),
    )
    chromosome = Chromosome.from_vector(
        np.array([int(x) for x in member["chromosome"].split()]), bounds
    )
    result = evaluator.evaluate(chromosome)
    if result.status != "ok":
        raise RuntimeError(f"rebuilding C* failed: {result.error}")

    for key, stored_key in (("obj1_js", "obj1_js"),
                            ("obj2_retain_loss", "obj2_retain_loss"),
                            ("obj3_edit_cost", "obj3_edit_cost"),
                            ("forget_test_acc", "forget_test_acc"),
                            ("retain_test_acc", "retain_test_acc")):
        rebuilt = float(getattr(result, key))
        stored = float(member[stored_key])
        if abs(rebuilt - stored) > REBUILD_TOLERANCE:
            raise RuntimeError(
                f"rebuilt C* does not match its recorded row: {key} = "
                f"{rebuilt:.8f} but the front says {stored:.8f} "
                f"(tolerance {REBUILD_TOLERANCE}). Refusing to report a model "
                f"that is not the one the result was published for."
            )
    return evaluator.model


def write_csv(path: Path, order: list[str], rows: dict[str, dict[str, Any]],
              columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model"] + columns)
        for name in order:
            writer.writerow([name] + [rows[name].get(c, "") for c in columns])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--front",
                        default="results/search/plan_a_frog/full_fidelity/"
                                "front_full_fidelity.csv")
    parser.add_argument("--front-position", type=int, default=8,
                        help="the front row that is C*")
    parser.add_argument("--refined",
                        default="results/search/plan_a_frog_bn_frozen_refined/"
                                "refined_best.pt")
    parser.add_argument("--out-dir", default="results/literature_alignment")
    parser.add_argument("--max-shadow-per-group", type=int, default=None,
                        help="cap each anchor-MIA shadow group; omit to "
                             "reproduce the anchor exactly")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True
    seed = int(cfg.get("seed", 42))

    print("=" * 100)
    print("ANCHOR-PROTOCOL METRICS -- Kodge, Saha & Roy (TMLR 2024)")
    print("=" * 100)

    started = time.perf_counter()
    evaluator = ClassEvaluator(cfg)
    print(f"  evaluator ready in {time.perf_counter() - started:.1f}s")
    print(f"  loader sizes  {evaluator.loaders.sizes()}")
    print(f"  W_0           {Path(evaluator.checkpoint_path).name}")
    print(f"  W_ref         {Path(evaluator.reference_path).name}")
    if args.max_shadow_per_group is not None:
        print(f"  MIA shadow groups CAPPED at {args.max_shadow_per_group} "
              f"-- this deviates from the anchor")

    rows: dict[str, dict[str, Any]] = {}
    num_classes = int(cfg["data"]["num_classes"])

    def measured(model) -> dict[str, Any]:
        return measure_model(evaluator, model, args.max_shadow_per_group, seed)

    print("\n  measuring W_0 ...")
    w0 = build_model(cfg["model"], num_classes=num_classes)
    load_checkpoint(evaluator.checkpoint_path, w0, map_location="cpu")
    rows["W_0"] = measured(w0)
    rows["W_0"]["kind"] = "baseline (original model)"

    print("  measuring W_ref ...")
    wref = build_model(cfg["model"], num_classes=num_classes)
    load_checkpoint(evaluator.reference_path, wref, map_location="cpu")
    rows["W_ref"] = measured(wref)
    rows["W_ref"]["kind"] = "baseline (retain-only reference, gold standard)"
    # f3 for W_ref is not an edit cost -- W_ref is independently trained, not an
    # edit of W_0. Same treatment as report_final_objectives.py.
    rows["W_ref"]["distance_from_W0_not_an_edit"] = rows["W_ref"].pop("f3_edit_cost")

    print(f"  rebuilding and measuring C* (front #{args.front_position}) ...")
    with resolve_path(args.front).open(encoding="utf-8-sig") as handle:
        front = list(csv.DictReader(handle))
    member = next(m for m in front
                  if int(m["front_position"]) == args.front_position)
    cstar = rebuild_candidate(evaluator, cfg, member)
    rows["C_star"] = measured(cstar)
    rows["C_star"]["kind"] = "PURE GRADIENT-FREE (Pareto front member)"
    rows["C_star"]["operators"] = member["operators"]

    refined_path = resolve_path(args.refined)
    if refined_path.is_file():
        print("  measuring C*_refined_bn_frozen ...")
        refined = build_model(cfg["model"], num_classes=num_classes)
        load_checkpoint(refined_path, refined, map_location="cpu")
        rows["C_star_refined_bn_frozen"] = measured(refined)
        rows["C_star_refined_bn_frozen"]["kind"] = (
            "POST-SEARCH REFINEMENT -- not a Pareto-front member, not part of "
            "the evolutionary search"
        )
        rows["C_star_refined_bn_frozen"]["operators"] = "DAMP|MASK + 2 gradient steps"
    else:
        print(f"  refined checkpoint not found: {refined_path}")

    original = rows["W_0"]
    for row in rows.values():
        row.setdefault("selectivity_S", selectivity(
            row["forget_train_loss"], row["retain_train_loss"],
            original["forget_train_loss"], original["retain_train_loss"],
        ))

    order = [n for n in ("W_0", "W_ref", "C_star", "C_star_refined_bn_frozen")
             if n in rows]
    columns = [
        "kind", "operators",
        "anchor_ACC_r", "anchor_ACC_f", "anchor_composite", "anchor_MIA",
        "f1_js", "f2_retain_train_loss", "f3_edit_cost",
        "distance_from_W0_not_an_edit",
        "selectivity_S", "mia_auc",
        "forget_train_acc", "forget_train_loss",
        "forget_test_acc", "forget_test_loss",
        "retain_train_acc", "retain_train_loss",
        "retain_test_acc", "retain_test_loss",
        "kl_to_reference",
        "n_shadow_member", "n_shadow_nonmember", "n_target_nonmember",
        "subsampled",
    ]

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "frog_anchor_metrics.csv", order, rows, columns)

    payload = {
        "anchor": {
            "citation": ("Kodge, Saha & Roy. Deep Unlearning: Fast and Efficient "
                         "Gradient-free Class Forgetting. TMLR 07/2024."),
            "paper": "https://openreview.net/forum?id=BmI5p6wBi0",
            "code": "https://github.com/sangamesh-kodge/class_forgetting",
        },
        "config": args.config,
        "forget_class": evaluator.forget_class,
        "loader_sizes": evaluator.loaders.sizes(),
        "max_shadow_per_group": args.max_shadow_per_group,
        "rows": rows,
    }
    (out_dir / "frog_anchor_metrics.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")

    # --- console table ----------------------------------------------------
    labels = {
        "W_0": "W_0 (original)",
        "W_ref": "W_ref (retain-only)",
        "C_star": "C* (front #8)",
        "C_star_refined_bn_frozen": "C*_refined_bn_frozen",
    }
    display = [
        ("anchor_ACC_r", "ACC_r  (%)"),
        ("anchor_ACC_f", "ACC_f  (%)"),
        ("anchor_composite", "composite (%)"),
        ("anchor_MIA", "MIA  (%)"),
        ("f1_js", "f1  JS to W_ref"),
        ("f2_retain_train_loss", "f2  retain train loss"),
        ("f3_edit_cost", "f3  edit cost"),
        ("selectivity_S", "selectivity S"),
        ("mia_auc", "our MIA AUC"),
    ]
    width = 22
    print("\n" + "-" * (26 + width * len(order)))
    print(f"  {'metric':<24}" + "".join(f"{labels[n][:width - 1]:>{width}}"
                                        for n in order))
    print("-" * (26 + width * len(order)))
    for key, label in display:
        line = f"  {label:<24}"
        for name in order:
            value = rows[name].get(key)
            line += (f"{'--':>{width}}" if value is None
                     else f"{float(value):>{width}.4f}")
        print(line)
    print("-" * (26 + width * len(order)))

    print(f"\n  wrote {out_dir / 'frog_anchor_metrics.csv'}")
    print(f"  wrote {out_dir / 'frog_anchor_metrics.json'}")
    print(f"\n  Anchor Table 1 reference rows (CIFAR-10 / ResNet-18, "
          f"mean over all 10 classes):")
    print(f"    Original     ACC_r 94.89  ACC_f 94.89  MIA  0.03")
    print(f"    Retraining   ACC_r 94.81  ACC_f  0.00  MIA 100.00")
    print(f"    Kodge et al. ACC_r 94.19  ACC_f  0.03  MIA 95.50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
