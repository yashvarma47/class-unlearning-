"""Full-fidelity evaluation of a Plan A front.

The search screens on subsets. Nothing from it is reportable until it has been
re-measured here, on the complete sets, which is what this does.

What it reports, per candidate
------------------------------
============================  ====================================================
D_f_train acc / loss          the 5 000 frog training images
D_r_train acc / loss          the other 45 000
**D_f_test acc / loss**       1 000 unseen frogs -- THE headline
D_r_test acc / loss           9 000 unseen non-frogs -- utility
full test acc                 both halves together, for comparability
selectivity S                 (forget loss gained) / (retain loss paid)
MIA AUC                       diagnostic only
gap to W_ref                  candidate minus reference, on every metric above
============================  ====================================================

Why the gap to the reference is the target, not zero forget accuracy
--------------------------------------------------------------------
A model at 0% frog accuracy has not forgotten frogs -- it has learned to avoid
answering "frog", which is a signature an attacker can detect and is not what a
model that never saw one looks like. W_ref *did* never see one, and it
misclassifies frogs the way any naive model would. So every number is printed
beside the reference's, and the gap is the thing to minimise.

Selectivity S is carried over from the predecessor project for continuity. There
it never exceeded **1.158** across 10 534 strategies, against ~932 for
retraining.

Run::

    python experiments/evaluate_class_front.py \\
        --front results/search/plan_a_frog/pareto_front.csv
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
from medus_class.search import Chromosome, ChromosomeBounds  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

#: Below this on D_r_test, a candidate has broken the model and its forgetting
#: number is meaningless -- anything forgets frogs if it forgets everything.
MIN_USABLE_RETAIN_TEST_ACC = 0.80


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--front", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Full fidelity: every subset knob off. The test loaders were never subset.
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0

    front_path = resolve_path(args.front)
    with front_path.open(encoding="utf-8-sig") as handle:
        front = list(csv.DictReader(handle))
    targets = front[: args.limit] if args.limit else front

    print("=" * 100)
    print("FULL-FIDELITY EVALUATION -- CLASS UNLEARNING")
    print("=" * 100)
    print(f"  front        {front_path.relative_to(PROJECT_ROOT)}  ({len(front)} members)")
    print(f"  evaluating   {len(targets)}")

    started = time.perf_counter()
    evaluator = ClassEvaluator(cfg)
    print(f"  evaluator ready in {time.perf_counter() - started:.1f}s")
    print(f"  loader sizes {evaluator.loaders.sizes()}")

    original = evaluator.original
    reference = evaluator.reference_metrics

    print("\n" + "-" * 100)
    print("BASELINES")
    print("-" * 100)
    print(f"  {'model':<12}{'D_f acc':>10}{'D_f loss':>11}{'D_r acc':>10}"
          f"{'D_r loss':>11}{'D_f_test':>11}{'D_r_test':>11}")
    for label, m in (("original", original), ("reference", reference)):
        print(f"  {label:<12}{m['forget_train_acc']:>10.4f}{m['forget_train_loss']:>11.4f}"
              f"{m['retain_train_acc']:>10.4f}{m['retain_train_loss']:>11.4f}"
              f"{m['forget_test_acc']:>11.4f}{m['retain_test_acc']:>11.4f}")
    print("\n  The reference is the target. It never saw a frog, so its D_f_test")
    print("  accuracy is what 'forgotten' actually looks like -- not zero.")

    bounds = ChromosomeBounds.from_registry(
        n_groups=len(evaluator.registry.names),
        implemented_only=True,
        max_level=cfg["evaluation"].get("max_level"),
    )

    rows: list[dict[str, Any]] = []
    timings: list[float] = []

    print("\n" + "-" * 100)
    print("CANDIDATES")
    print("-" * 100)
    print(f"  {'#':<4}{'f1 JS':>9}{'f2 L_r':>9}{'f3 edit':>9}{'D_f acc':>9}"
          f"{'D_f_test':>10}{'D_r_test':>10}{'S':>9}{'MIA':>7}")

    for i, member in enumerate(targets):
        chromosome = Chromosome.from_vector(
            np.array([int(x) for x in member["chromosome"].split()]), bounds
        )

        step = time.perf_counter()
        result = evaluator.evaluate(chromosome)
        timings.append(time.perf_counter() - step)

        row = result.to_row()
        row["front_position"] = member.get("front_position", i)
        for key in ("forget_train_acc", "forget_train_loss", "retain_train_acc",
                    "retain_train_loss", "forget_test_acc", "forget_test_loss",
                    "retain_test_acc", "retain_test_loss"):
            row[f"gap_{key}"] = getattr(result, key) - reference[key]

        n_f = len(evaluator.loaders.forget_test.dataset)
        n_r = len(evaluator.loaders.retain_test.dataset)
        row["full_test_acc"] = (
            n_f * result.forget_test_acc + n_r * result.retain_test_acc
        ) / (n_f + n_r)
        rows.append(row)

        print(f"  {i:<4}{result.obj1_js:>9.5f}{result.obj2_retain_loss:>9.4f}"
              f"{result.obj3_edit_cost:>9.5f}{result.forget_train_acc:>9.4f}"
              f"{result.forget_test_acc:>10.4f}{result.retain_test_acc:>10.4f}"
              f"{result.selectivity_S:>9.3f}{result.mia_auc:>7.3f}")

    # --- verdict -----------------------------------------------------------
    ok = [r for r in rows if r["status"] == "ok"]
    finite_s = [r["selectivity_S"] for r in ok if np.isfinite(r["selectivity_S"])]

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    if finite_s:
        best_s = max(finite_s)
        print(f"  best selectivity S      {best_s:.3f}")
        print(f"  median selectivity S    {np.median(finite_s):.3f}")
        print(f"  instance-level ceiling  1.158   (10 534 strategies, 4 objectives)")
        if best_s > 2.0:
            print("\n  S EXCEEDS THE INSTANCE-LEVEL CEILING -- the operators became")
            print("  selective once forget-specific structure existed, which is what")
            print("  the class-structure measurement predicted.")
        else:
            print("\n  S did NOT clear the instance-level ceiling, despite 91% of")
            print("  channels carrying class structure. That would refute the")
            print("  explanation on record and point the cause elsewhere.")

    usable = [r for r in ok if r["retain_test_acc"] > MIN_USABLE_RETAIN_TEST_ACC]
    if usable:
        best = min(usable, key=lambda r: abs(r["gap_forget_test_acc"]))
        print(f"\n  C* -- closest to the reference on D_f_test, with D_r_test > "
              f"{MIN_USABLE_RETAIN_TEST_ACC:.2f}:")
        print(f"    front position    {best['front_position']}")
        print(f"    operators         {best['operators']}")
        print(f"    D_f_test acc      {best['forget_test_acc']:.4f}  "
              f"(reference {reference['forget_test_acc']:.4f}, "
              f"gap {best['gap_forget_test_acc']:+.4f})")
        print(f"    D_r_test acc      {best['retain_test_acc']:.4f}  "
              f"(reference {reference['retain_test_acc']:.4f}, "
              f"gap {best['gap_retain_test_acc']:+.4f})")
        print(f"    edit cost         {best['obj3_edit_cost']:.5f}")
        print(f"    MIA AUC           {best['mia_auc']:.3f}  (diagnostic)")
    else:
        print(f"\n  No candidate held D_r_test above {MIN_USABLE_RETAIN_TEST_ACC:.2f};")
        print("  there is no C*. Every front member broke the model.")

    if timings:
        print(f"\n  timing  mean {np.mean(timings):.1f}s  total {sum(timings)/60:.1f} min")

    out_dir = resolve_path(args.out) if args.out else front_path.parent / "full_fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "front_full_fidelity.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "baselines.json").write_text(json.dumps({
        "forget_class": evaluator.forget_class,
        "original": original,
        "reference": reference,
        "reference_checkpoint": evaluator.reference_path,
    }, indent=2), encoding="utf-8")

    print(f"\n  wrote {out_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
