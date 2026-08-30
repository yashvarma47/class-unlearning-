"""Where do truck images go after unlearning? Inference only.

Truck is the weakest class in the sweep -- pure ``ACC_f`` 42.10 against a
ten-class mean of 12.55, and still 30.90 after the BN-frozen refinement. The
tables say *how much* is left; they do not say *where the rest went*. This script
answers that by classifying the 1,000 truck test images with four models and
recording the full predicted-class distribution of each.

    W_0        the original model, which should call them all trucks
    W_ref      the retain-only reference, which never saw a truck
    C* pure    the selected gradient-free front member
    C* hybrid  the same C* after one BN-frozen refinement step

**Nothing is trained, searched or refined here, and no committed result is
touched.** The only computation is a forward pass. ``C*`` has no checkpoint of
its own -- the search recorded genomes, not weights -- so it is reconstructed by
replaying its stored chromosome through the same deterministic operators, which
is exactly what ``report_anchor_metrics.py`` already does to score it.
``rebuild_candidate`` refuses to return a model whose recomputed objectives drift
from the recorded front row, so a reconstruction that is not the published ``C*``
raises rather than plots.

Output: ``results/writeup_package/truck_prediction_distribution.csv``, one row per
model per predicted class, counts and percentages over the 1,000 images.

Run::

    python experiments/analyse_truck_predictions.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_anchor_metrics import rebuild_candidate  # noqa: E402

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.models import build_model, load_checkpoint  # noqa: E402
from medus_class.utils.config import load_config, resolve_path  # noqa: E402

OUT_CSV = PROJECT_ROOT / "results" / "writeup_package" / "truck_prediction_distribution.csv"


@torch.no_grad()
def predict_distribution(model: torch.nn.Module, loader, device: str,
                         num_classes: int) -> list[int]:
    """Count argmax predictions over a loader. One forward pass, no gradients."""
    model = model.to(device).eval()
    counts = [0] * num_classes
    total = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        predicted = model(images).argmax(dim=1)
        for cls in predicted.detach().cpu().tolist():
            counts[cls] += 1
        total += images.shape[0]
    if total != sum(counts):
        raise RuntimeError(f"counted {sum(counts)} predictions over {total} images")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_truck.yaml")
    parser.add_argument("--front", default="results/search/plan_a_truck/"
                                           "full_fidelity/front_full_fidelity.csv")
    parser.add_argument("--front-position", type=int, default=0)
    parser.add_argument("--refined", default="results/search/"
                                             "plan_a_truck_bn_frozen_refined/refined_best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True

    num_classes = int(cfg["data"]["num_classes"])
    forget_class = int(cfg["split"]["forget_class"])

    print("=" * 78)
    print("TRUCK PREDICTION DISTRIBUTION -- inference only, nothing is trained")
    print("=" * 78)

    evaluator = ClassEvaluator(cfg)
    device = evaluator.device
    loader = evaluator.loaders.forget_test
    print(f"  device        {device}")
    print(f"  forget class  {forget_class} ({CIFAR10_CLASS_NAMES[forget_class]})")
    print(f"  loader sizes  {evaluator.loaders.sizes()}")

    rows: list[tuple[str, str, list[int]]] = []

    # C* first: rebuild_candidate hands back the evaluator's own model, and any
    # later evaluate() call would overwrite it.
    with resolve_path(args.front).open(encoding="utf-8-sig") as handle:
        front = list(csv.DictReader(handle))
    member = next(m for m in front
                  if int(m["front_position"]) == args.front_position)
    print(f"\n  rebuilding C* (front #{args.front_position}, "
          f"operators={member['operators']}) ...")
    cstar = rebuild_candidate(evaluator, cfg, member)
    rows.append(("C_star_pure", "pure gradient-free C*",
                 predict_distribution(cstar, loader, device, num_classes)))

    def from_checkpoint(path: str) -> Any:
        model = build_model(cfg["model"], num_classes=num_classes)
        load_checkpoint(resolve_path(path), model, map_location="cpu")
        return model

    print("  classifying with W_0 ...")
    rows.insert(0, ("W_0", "original model",
                    predict_distribution(from_checkpoint(evaluator.checkpoint_path),
                                         loader, device, num_classes)))

    print("  classifying with W_ref ...")
    rows.insert(1, ("W_ref", "retain-only reference (gold standard)",
                    predict_distribution(from_checkpoint(evaluator.reference_path),
                                         loader, device, num_classes)))

    print("  classifying with the refined C* ...")
    rows.append(("C_star_hybrid", "C* + BN-frozen refinement (hybrid)",
                 predict_distribution(from_checkpoint(args.refined),
                                      loader, device, num_classes)))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "model_label", "predicted_class_id",
                         "predicted_class_name", "count", "percent",
                         "is_forget_class"])
        for key, label, counts in rows:
            total = sum(counts)
            for cid, count in enumerate(counts):
                writer.writerow([key, label, cid, CIFAR10_CLASS_NAMES[cid], count,
                                 f"{100.0 * count / total:.2f}",
                                 cid == forget_class])

    print(f"\n  wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"\n  {'model':<16} {'still truck':>12}   top non-truck destination")
    print("  " + "-" * 62)
    for key, _label, counts in rows:
        total = sum(counts)
        still = 100.0 * counts[forget_class] / total
        others = [(c, i) for i, c in enumerate(counts) if i != forget_class]
        top_count, top_id = max(others)
        print(f"  {key:<16} {still:>11.2f}%   {CIFAR10_CLASS_NAMES[top_id]} "
              f"({100.0 * top_count / total:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
