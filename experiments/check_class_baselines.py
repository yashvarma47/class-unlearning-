"""Prove a class config is wired to the right models before anything expensive runs.

A MED-US search reads two fixed models: ``W_0``, the model being unlearned from,
and ``W_ref``, the retain-only reference that ``f1`` is measured against. Point
either at the wrong file and the search still runs, still produces a Pareto
front, and still writes plausible numbers -- they are just numbers about a
different experiment. Nothing downstream would catch it.

So before the ship search (or any later class), this measures both baselines
through the same :class:`~medus_class.evaluation.ClassEvaluator` the search
uses, and asserts the three things that cannot be true by accident:

* ``W_ref`` scores **~0** on ``D_f_test`` -- it never saw the class
* ``W_ref`` scores **high** on ``D_r_test`` -- it is still a good CIFAR-9 model
* ``W_0`` scores **high** on ``D_f_test`` -- it very much did see the class

and that no path anywhere in the resolved config belongs to another class.

Run::

    python experiments/check_class_baselines.py \\
        --config search/plan_a_ship_smoke.yaml --expect-class 8 \\
        --expect-reference-retain-test-acc 0.9502
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

#: A retain-only reference must be near-useless on the class it never saw.
MAX_REFERENCE_FORGET_TEST_ACC = 0.02
#: ...and still a good model on the other nine.
MIN_REFERENCE_RETAIN_TEST_ACC = 0.90
#: The original model must still recognise the class. If it does not, ``W_0`` is
#: not the model the project thinks it is.
MIN_ORIGINAL_FORGET_TEST_ACC = 0.80
#: How close the measured reference accuracy must be to the value recorded when
#: the checkpoint was imported. Forward passes are deterministic, so this is
#: tight on purpose -- a drift means a different file.
ACC_TOLERANCE = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expect-class", type=int, required=True)
    parser.add_argument("--expect-reference-forget-test-acc", type=float, default=0.0)
    parser.add_argument("--expect-reference-retain-test-acc", type=float, default=None)
    parser.add_argument("--out", default=None,
                        help="write a markdown report here")
    args = parser.parse_args()

    class_id = args.expect_class
    class_name = CIFAR10_CLASS_NAMES[class_id]

    cfg = load_config(args.config)
    # Full fidelity: the baselines are reported numbers, not screening numbers.
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True

    print("=" * 100)
    print(f"BASELINE WIRING CHECK -- class {class_id} ({class_name})")
    print("=" * 100)
    print(f"  config  {args.config}")

    failures: list[str] = []

    # --- static checks on the resolved config, before anything loads ------
    if int(cfg["split"]["forget_class"]) != class_id:
        failures.append(f"config forget_class is {cfg['split']['forget_class']}, "
                        f"expected {class_id}")
    other_names = [n for i, n in enumerate(CIFAR10_CLASS_NAMES) if i != class_id]
    blob = json.dumps(cfg)
    for name in other_names:
        if name in blob:
            failures.append(f"resolved config mentions another class: '{name}'")
    for other_id in range(len(CIFAR10_CLASS_NAMES)):
        if other_id != class_id and f"class{other_id}" in blob:
            failures.append(f"resolved config contains a 'class{other_id}' path")

    evaluator = ClassEvaluator(cfg)

    print(f"\n  split file           {cfg['split']['split_file']}")
    print(f"  W_0                  {evaluator.checkpoint_path}")
    print(f"  W_ref                {evaluator.reference_path}")
    print(f"  forget class in use  {evaluator.forget_class} "
          f"({CIFAR10_CLASS_NAMES[evaluator.forget_class]})")
    print(f"  loader sizes         {evaluator.loaders.sizes()}")
    print(f"  device               {evaluator.device}")

    if evaluator.forget_class != class_id:
        failures.append(f"evaluator loaded class {evaluator.forget_class}, "
                        f"expected {class_id}")
    expected_reference = f"class{class_id}_{class_name}_reference_best_dr.pt"
    if Path(evaluator.reference_path).name != expected_reference:
        failures.append(f"reference is {Path(evaluator.reference_path).name}, "
                        f"expected {expected_reference}")
    expected_split = f"cifar10_class{class_id}_{class_name}.json"
    if Path(cfg["split"]["split_file"]).name != expected_split:
        failures.append(f"split file is {Path(cfg['split']['split_file']).name}, "
                        f"expected {expected_split}")

    original = evaluator.original
    reference = evaluator.reference_metrics

    print("\n" + "-" * 100)
    print(f"  {'model':<22}{'D_f_test acc':>14}{'D_f_test loss':>15}"
          f"{'D_r_test acc':>14}{'D_r_test loss':>15}")
    print("-" * 100)
    for label, row in (("W_0 (original)", original), ("W_ref (retain-only)", reference)):
        print(f"  {label:<22}{row['forget_test_acc']:>14.4f}"
              f"{row['forget_test_loss']:>15.4f}"
              f"{row['retain_test_acc']:>14.4f}{row['retain_test_loss']:>15.4f}")

    # --- the checks that matter -------------------------------------------
    if reference["forget_test_acc"] > MAX_REFERENCE_FORGET_TEST_ACC:
        failures.append(
            f"W_ref D_f_test accuracy {reference['forget_test_acc']:.4f} > "
            f"{MAX_REFERENCE_FORGET_TEST_ACC} -- this reference has seen "
            f"{class_name}"
        )
    if reference["retain_test_acc"] < MIN_REFERENCE_RETAIN_TEST_ACC:
        failures.append(
            f"W_ref D_r_test accuracy {reference['retain_test_acc']:.4f} < "
            f"{MIN_REFERENCE_RETAIN_TEST_ACC}"
        )
    if original["forget_test_acc"] < MIN_ORIGINAL_FORGET_TEST_ACC:
        failures.append(
            f"W_0 D_f_test accuracy {original['forget_test_acc']:.4f} < "
            f"{MIN_ORIGINAL_FORGET_TEST_ACC} -- W_0 does not recognise "
            f"{class_name}, so it is not the model this project unlearns from"
        )
    if abs(reference["forget_test_acc"] - args.expect_reference_forget_test_acc) \
            > ACC_TOLERANCE:
        failures.append(
            f"W_ref D_f_test accuracy {reference['forget_test_acc']:.6f} != "
            f"recorded {args.expect_reference_forget_test_acc}"
        )
    if args.expect_reference_retain_test_acc is not None and \
            abs(reference["retain_test_acc"]
                - args.expect_reference_retain_test_acc) > ACC_TOLERANCE:
        failures.append(
            f"W_ref D_r_test accuracy {reference['retain_test_acc']:.6f} != "
            f"recorded {args.expect_reference_retain_test_acc}"
        )

    print()
    print(f"  W_ref forgot {class_name}      "
          f"{'OK' if reference['forget_test_acc'] <= MAX_REFERENCE_FORGET_TEST_ACC else 'FAIL'}"
          f"   ({reference['forget_test_acc']:.4f} <= {MAX_REFERENCE_FORGET_TEST_ACC})")
    print(f"  W_ref still useful        "
          f"{'OK' if reference['retain_test_acc'] >= MIN_REFERENCE_RETAIN_TEST_ACC else 'FAIL'}"
          f"   ({reference['retain_test_acc']:.4f} >= {MIN_REFERENCE_RETAIN_TEST_ACC})")
    print(f"  W_0 knows {class_name}         "
          f"{'OK' if original['forget_test_acc'] >= MIN_ORIGINAL_FORGET_TEST_ACC else 'FAIL'}"
          f"   ({original['forget_test_acc']:.4f} >= {MIN_ORIGINAL_FORGET_TEST_ACC})")

    print("\n" + "=" * 100)
    if failures:
        print(f"RESULT: FAIL -- {len(failures)} problem(s)")
        for line in failures:
            print(f"  - {line}")
    else:
        print("RESULT: PASS -- the config is wired to the right models")
    print("=" * 100)

    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = (("`W_0` (original)", original), ("`W_ref` (retain-only)", reference))
        out.write_text(
            f"# Baseline wiring check — class {class_id} ({class_name})\n\n"
            f"Produced by `experiments/check_class_baselines.py` from "
            f"`{args.config}` at "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n\n"
            f"**Result: {'FAIL' if failures else 'PASS'}**\n\n"
            f"| | |\n|---|---|\n"
            f"| forget class | {class_id} ({class_name}) |\n"
            f"| split file | `{cfg['split']['split_file']}` |\n"
            f"| `W_0` | `{evaluator.checkpoint_path}` |\n"
            f"| `W_ref` | `{evaluator.reference_path}` |\n"
            f"| loader sizes | `{evaluator.loaders.sizes()}` |\n"
            f"| device | {evaluator.device} |\n\n"
            f"| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | "
            f"`D_r_test` loss |\n|---|---:|---:|---:|---:|\n"
            + "".join(
                f"| {label} | {r['forget_test_acc']:.4f} | "
                f"{r['forget_test_loss']:.4f} | {r['retain_test_acc']:.4f} | "
                f"{r['retain_test_loss']:.4f} |\n" for label, r in rows)
            + "\n## Checks\n\n"
            f"| check | threshold | measured | verdict |\n|---|---|---:|---|\n"
            f"| `W_ref` forgot {class_name} | ≤ {MAX_REFERENCE_FORGET_TEST_ACC} | "
            f"{reference['forget_test_acc']:.4f} | "
            f"{'PASS' if reference['forget_test_acc'] <= MAX_REFERENCE_FORGET_TEST_ACC else 'FAIL'} |\n"
            f"| `W_ref` still useful | ≥ {MIN_REFERENCE_RETAIN_TEST_ACC} | "
            f"{reference['retain_test_acc']:.4f} | "
            f"{'PASS' if reference['retain_test_acc'] >= MIN_REFERENCE_RETAIN_TEST_ACC else 'FAIL'} |\n"
            f"| `W_0` knows {class_name} | ≥ {MIN_ORIGINAL_FORGET_TEST_ACC} | "
            f"{original['forget_test_acc']:.4f} | "
            f"{'PASS' if original['forget_test_acc'] >= MIN_ORIGINAL_FORGET_TEST_ACC else 'FAIL'} |\n"
            + ("\n## Failures\n\n" + "".join(f"* {f}\n" for f in failures)
               if failures else ""),
            encoding="utf-8",
        )
        print(f"  report written to {out.relative_to(PROJECT_ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
