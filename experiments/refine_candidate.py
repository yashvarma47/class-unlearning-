"""Optional post-search refinement of C* (pipeline step 7).

One small gradient ascent step on ``D_f``, then one repair step on ``D_r``.

Deliberately **outside** the evolutionary search. The chromosome describes a
gradient-free weight edit; folding gradient steps into the genome would change
what the search space means and make every result incomparable with the ones
before it. This is a finishing pass applied to a single chosen strategy,
reported separately, and easy to discard.

Why it can help
---------------
The search optimises under cheap-stage evaluation. Its winner sits close to the
reference but rarely exactly on it, and the two residual errors have opposite
signs: some forget-class memory survives, and some retain damage was paid to
remove the rest. A gradient carries direction information a gradient-free
operator does not, so one step of each can close part of both gaps.

Why it is dangerous, and what guards it
---------------------------------------
An ascent step on ``D_f`` is unbounded -- it will happily run the forget loss to
infinity and take the retain set with it. Three guards:

* the step is **small** and taken once, not to convergence;
* the update is **clipped** so neither step can move the weights further than
  ``--max-delta`` in relative norm;
* every intermediate model is measured, and a refinement costing more than
  ``--max-retain-drop`` on ``D_r_test`` is **not written to disk**. A refinement
  that makes things worse should leave no artefact anyone can pick up later.

Run::

    python experiments/refine_candidate.py \\
        --front results/search/plan_a_frog/pareto_front.csv --position 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.objectives import (  # noqa: E402
    relative_parameter_delta,
    selectivity,
)
from medus_class.models import CheckpointMetadata, save_checkpoint  # noqa: E402
from medus_class.search import Chromosome, ChromosomeBounds  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402


def one_step(
    model: nn.Module, loader, device: str, lr: float, ascend: bool, batches: int
) -> float:
    """One SGD step on the mean gradient over ``batches`` batches.

    ``ascend`` flips the sign: ascent raises the loss on ``D_f`` (forgetting),
    descent lowers it on ``D_r`` (repair). Gradients are accumulated and a single
    ``step()`` is taken, so this is one update on a low-variance gradient rather
    than ``batches`` separate noisy ones.
    """
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    optimiser.zero_grad(set_to_none=True)

    total, seen = 0.0, 0
    for index, (images, labels) in enumerate(loader):
        if index >= batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss = criterion(model(images), labels)
        ((-loss if ascend else loss) / batches).backward()
        total += float(loss.item())
        seen += 1

    if seen == 0:
        raise ValueError("loader yielded no batches")

    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    model.eval()
    return total / seen


def clip_to_budget(
    model: nn.Module, anchor: dict[str, torch.Tensor], max_delta: float
) -> float:
    """Pull the weights back toward ``anchor`` if they moved further than allowed.

    Scales the whole update uniformly rather than clipping per tensor, which
    would change the update's *direction* as well as its size.
    """
    moved = relative_parameter_delta(model, anchor)
    if moved <= max_delta or moved == 0.0:
        return moved

    scale = max_delta / moved
    with torch.no_grad():
        state = model.state_dict()
        for name, base in anchor.items():
            if not name.endswith("weight") or not torch.is_floating_point(base):
                continue
            if name not in state:
                continue
            base_on_device = base.to(state[name].device)
            state[name].copy_(base_on_device + (state[name] - base_on_device) * scale)
    return relative_parameter_delta(model, anchor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--front", required=True)
    parser.add_argument("--position", type=int, default=None,
                        help="front_position of C*. Default: the first row.")
    parser.add_argument("--forget-lr", type=float, default=1e-4)
    parser.add_argument("--retain-lr", type=float, default=1e-4)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--max-delta", type=float, default=0.02,
                        help="Relative weight-movement budget per step.")
    parser.add_argument("--max-retain-drop", type=float, default=0.02,
                        help="Refuse to save if D_r_test accuracy falls by more.")
    parser.add_argument("--out", default="results/search/plan_a_frog_refined")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True

    front_path = resolve_path(args.front)
    with front_path.open(encoding="utf-8-sig") as handle:
        front = list(csv.DictReader(handle))

    print("=" * 100)
    print("REFINEMENT -- one forget step, one retain repair (outside the search)")
    print("=" * 100)

    evaluator = ClassEvaluator(cfg)
    original = evaluator.original
    reference = evaluator.reference_metrics

    if args.position is not None:
        member = next(m for m in front
                      if int(m.get("front_position", -1)) == args.position)
    else:
        member = front[0]

    bounds = ChromosomeBounds.from_registry(
        n_groups=len(evaluator.registry.names),
        implemented_only=True,
        max_level=cfg["evaluation"].get("max_level"),
    )
    chromosome = Chromosome.from_vector(
        np.array([int(x) for x in member["chromosome"].split()]), bounds
    )

    print(f"  C*            front position {member.get('front_position')}")
    print(f"  operators     {member.get('operators', '')}")
    print(f"  forget lr     {args.forget_lr}   retain lr {args.retain_lr}")
    print(f"  batches/step  {args.batches}     delta budget {args.max_delta}")

    stages: list[dict[str, Any]] = []

    def record(label: str, model) -> dict[str, float]:
        measured = evaluator.measure_sets(model)
        measured["selectivity_S"] = selectivity(
            measured["forget_train_loss"], measured["retain_train_loss"],
            original["forget_train_loss"], original["retain_train_loss"],
        )
        measured["stage"] = label
        stages.append(measured)
        return measured

    # Stage 0 -- the strategy exactly as the search produced it.
    evaluator.evaluate(chromosome)
    model = evaluator.model
    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    record("C* (search output)", model)

    # Stage 1 -- one clipped forget ascent step.
    forget_loss = one_step(model, evaluator.loaders.forget_train, evaluator.device,
                           args.forget_lr, ascend=True, batches=args.batches)
    moved = clip_to_budget(model, anchor, args.max_delta)
    print(f"\n  forget step   mean L_f {forget_loss:.4f}   movement {moved:.5f}")
    record("+ forget step", model)

    # Stage 2 -- one retain repair step, budgeted against the post-ascent model.
    repair_anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    retain_loss = one_step(model, evaluator.loaders.retain_train, evaluator.device,
                           args.retain_lr, ascend=False, batches=args.batches)
    moved = clip_to_budget(model, repair_anchor, args.max_delta)
    print(f"  retain step   mean L_r {retain_loss:.4f}   movement {moved:.5f}")
    final = record("+ retain repair", model)

    print("\n" + "-" * 100)
    print("BEFORE / AFTER")
    print("-" * 100)
    print(f"  {'stage':<22}{'D_f acc':>9}{'D_f_test':>10}{'D_r acc':>9}"
          f"{'D_r_test':>10}{'S':>9}")
    for s in stages:
        print(f"  {s['stage']:<22}{s['forget_train_acc']:>9.4f}"
              f"{s['forget_test_acc']:>10.4f}{s['retain_train_acc']:>9.4f}"
              f"{s['retain_test_acc']:>10.4f}{s['selectivity_S']:>9.3f}")
    print(f"  {'reference (target)':<22}{reference['forget_train_acc']:>9.4f}"
          f"{reference['forget_test_acc']:>10.4f}{reference['retain_train_acc']:>9.4f}"
          f"{reference['retain_test_acc']:>10.4f}{'--':>9}")

    drop = stages[0]["retain_test_acc"] - final["retain_test_acc"]

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refinement.json").write_text(json.dumps({
        "front_position": member.get("front_position"),
        "chromosome": member["chromosome"],
        "hyperparameters": {
            "forget_lr": args.forget_lr, "retain_lr": args.retain_lr,
            "batches": args.batches, "max_delta": args.max_delta,
        },
        "original": original,
        "reference": reference,
        "stages": stages,
        "retain_test_drop": drop,
        "accepted": bool(drop <= args.max_retain_drop),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    if drop > args.max_retain_drop:
        print(f"REFINEMENT REJECTED -- D_r_test fell {drop:.4f}, budget "
              f"{args.max_retain_drop}")
        print("Checkpoint NOT written. The search output stands.")
    else:
        path = out_dir / "refined_best.pt"
        save_checkpoint(
            path=path,
            model=model,
            metadata=CheckpointMetadata(
                model_name=cfg["model"]["name"],
                dataset=cfg["data"]["name"],
                seed=evaluator.seed,
                metrics={k: float(v) for k, v in final.items()
                         if isinstance(v, (int, float))},
                split_file=str(cfg["split"]["split_file"]),
                notes=(
                    "REFINEMENT: one clipped forget ascent step on D_f then one "
                    "retain repair step on D_r, applied to C* OUTSIDE the "
                    f"evolutionary search. front_position="
                    f"{member.get('front_position')} chromosome="
                    f"{member['chromosome']}"
                ),
            ),
        )
        print(f"REFINEMENT ACCEPTED -- D_r_test change {-drop:+.4f}")
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")

    print(f"wrote {(out_dir / 'refinement.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
