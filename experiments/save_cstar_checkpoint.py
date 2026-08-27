"""Materialise the pure gradient-free ``C*`` as a checkpoint, and prove it.

Why this is needed
------------------
The Plan A search recorded genomes, not weights. Every other model in the result
set has a ``.pt`` -- ``W_0``, ``W_ref``, ``C*_refined_bn_frozen`` -- but ``C*``
itself, the headline **pure gradient-free** result, existed only as a row in
``front_full_fidelity.csv`` plus the operator code needed to replay it. That is
reproducible in principle and fragile in practice: any change to an operator, a
selector, the registry or the seeding would silently produce a different ``C*``
while the CSV row kept saying what it always said.

So the model is rebuilt once, written to disk, and then **re-loaded from that
file and re-measured**. The verification is deliberately done against the
reloaded checkpoint rather than against the in-memory model, because what needs
proving is that the file is right, not that the rebuild was.

This trains nothing and searches nothing. Replaying one recorded chromosome
through deterministic weight-surgery operators is the same act
``experiments/report_anchor_metrics.py`` already performs to score ``C*``.

Run::

    python experiments/save_cstar_checkpoint.py

On any verification failure the checkpoint and its sidecar are deleted and the
script exits non-zero -- a wrong ``C*.pt`` on disk is worse than none, because
everything downstream would trust it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_anchor_metrics import (  # noqa: E402
    REBUILD_TOLERANCE,
    measure_model,
    rebuild_candidate,
)

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.objectives import selectivity  # noqa: E402
from medus_class.models import (  # noqa: E402
    CheckpointMetadata,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from medus_class.utils.config import load_config, resolve_path  # noqa: E402

#: What the reloaded checkpoint must reproduce, and where the expected value
#: comes from on the recorded Pareto-front row.
VERIFIED_AGAINST_FRONT: list[tuple[str, str]] = [
    ("f1_js", "obj1_js"),
    ("f2_retain_train_loss", "obj2_retain_loss"),
    ("f3_edit_cost", "obj3_edit_cost"),
    ("forget_test_acc", "forget_test_acc"),
    ("retain_test_acc", "retain_test_acc"),
    ("forget_train_acc", "forget_train_acc"),
    ("retain_train_acc", "retain_train_acc"),
    ("selectivity_S", "selectivity_S"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--front",
                        default="results/search/plan_a_frog/full_fidelity/"
                                "front_full_fidelity.csv")
    parser.add_argument("--front-position", type=int, default=8)
    parser.add_argument("--out",
                        default="results/checkpoints/"
                                "class6_frog_C_star_pure_gradient_free.pt")
    parser.add_argument("--max-shadow-per-group", type=int, default=None)
    args = parser.parse_args()

    out_path = resolve_path(args.out)
    if out_path.is_file():
        print(f"already exists, refusing to overwrite: {out_path}")
        return 0

    cfg = load_config(args.config)
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True
    seed = int(cfg.get("seed", 42))

    print("=" * 100)
    print("MATERIALISE C* -- pure gradient-free Pareto-front member")
    print("=" * 100)

    evaluator = ClassEvaluator(cfg)
    print(f"  loader sizes  {evaluator.loaders.sizes()}")

    with resolve_path(args.front).open(encoding="utf-8-sig") as handle:
        front = list(csv.DictReader(handle))
    member = next(m for m in front
                  if int(m["front_position"]) == args.front_position)
    print(f"  front row     #{args.front_position}  operators={member['operators']}")
    print(f"  chromosome    {member['chromosome']}")

    # rebuild_candidate already refuses to return a model whose recomputed
    # objectives drift from the recorded row.
    print("\n  rebuilding ...")
    model = rebuild_candidate(evaluator, cfg, member)

    save_checkpoint(
        path=out_path,
        model=model,
        metadata=CheckpointMetadata(
            model_name=cfg["model"]["name"],
            dataset=cfg["data"]["name"],
            seed=seed,
            metrics={key: float(member[source])
                     for key, source in VERIFIED_AGAINST_FRONT},
            split_file=str(cfg["split"]["split_file"]),
            notes=(
                "PURE GRADIENT-FREE C*: Pareto-front member "
                f"#{args.front_position} of the Plan A frog search, rebuilt by "
                f"replaying its recorded chromosome through the same "
                f"deterministic operators against W_0. No gradient step was "
                f"ever applied. operators={member['operators']} "
                f"chromosome={member['chromosome']} "
                f"source_front={args.front}"
            ),
        ),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  wrote {out_path}  ({size_mb:.1f} MB)")

    # --- the verification that matters: reload from disk ------------------
    print("\n  reloading from disk and re-measuring ...")
    reloaded = build_model(cfg["model"], num_classes=int(cfg["data"]["num_classes"]))
    load_checkpoint(out_path, reloaded, map_location="cpu")
    measured: dict[str, Any] = measure_model(
        evaluator, reloaded, args.max_shadow_per_group, seed
    )
    # measure_model reports raw losses; S is a ratio against the pristine model,
    # so it is computed here from the baselines the evaluator measured on W_0 --
    # the same two numbers ClassEvaluator._measure uses during a search.
    measured["selectivity_S"] = selectivity(
        measured["forget_train_loss"], measured["retain_train_loss"],
        evaluator.original["forget_train_loss"],
        evaluator.original["retain_train_loss"],
    )

    failures: list[str] = []
    print(f"\n  {'metric':<24}{'reloaded':>18}{'recorded':>18}{'delta':>14}")
    print("  " + "-" * 74)
    for key, source in VERIFIED_AGAINST_FRONT:
        got = float(measured[key])
        want = float(member[source])
        delta = got - want
        # Mixed absolute/relative: the accuracies and objectives are O(1) and an
        # absolute 1e-4 is right for them, but S is O(100) and a bit-for-bit
        # reload can still differ in its last float digits.
        allowed = REBUILD_TOLERANCE * max(1.0, abs(want))
        flag = ""
        if abs(delta) > allowed:
            failures.append(f"{key}: {got:.8f} != {want:.8f} (delta {delta:+.2e})")
            flag = "  <-- MISMATCH"
        print(f"  {key:<24}{got:>18.8f}{want:>18.8f}{delta:>14.2e}{flag}")

    print(f"\n  {'anchor ACC_r (%)':<24}{measured['anchor_ACC_r']:>18.4f}")
    print(f"  {'anchor ACC_f (%)':<24}{measured['anchor_ACC_f']:>18.4f}")
    print(f"  {'anchor composite (%)':<24}{measured['anchor_composite']:>18.4f}")
    anchor_mia_value = measured.get("anchor_MIA")
    print(f"  {'anchor MIA (%)':<24}"
          + ("{:>18.4f}".format(anchor_mia_value) if anchor_mia_value is not None
             else f"{'not computed':>18}"))

    if failures:
        print("\n  VERIFICATION FAILED -- deleting the checkpoint:")
        for line in failures:
            print(f"    {line}")
        out_path.unlink(missing_ok=True)
        out_path.with_suffix(".json").unlink(missing_ok=True)
        print(f"  removed {out_path}")
        return 1

    print(f"\n  VERIFIED: every recorded value reproduced within "
          f"{REBUILD_TOLERANCE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
