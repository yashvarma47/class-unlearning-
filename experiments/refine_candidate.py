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
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.privacy import compute_mia_auc  # noqa: E402
from medus_class.evaluation.objectives import (  # noqa: E402
    relative_parameter_delta,
    selectivity,
)
from medus_class.models import CheckpointMetadata, save_checkpoint  # noqa: E402
from medus_class.search import Chromosome, ChromosomeBounds  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402


def assert_batchnorm_frozen(model: nn.Module) -> int:
    """Every BatchNorm module must be in eval mode. Returns how many were checked.

    Asserted rather than assumed, because getting this wrong is silent and
    catastrophic: the first version of this script called ``model.train()``, and
    eight batches of D_r re-estimated ``running_mean``/``running_var`` from
    retain data. That undid the DAMP|MASK edit on the forget class -- D_f_test
    went 0.0830 -> 0.6700 -- while the weights moved by 3e-5 and every
    weight-based guard reported success.
    """
    checked = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            checked += 1
            if module.training:
                raise RuntimeError(
                    f"BatchNorm module {name!r} is in training mode; its running "
                    f"statistics would update during the refinement step and the "
                    f"result would measure recalibration rather than gradients."
                )
    return checked


@torch.no_grad()
def movement_report(
    model: nn.Module, anchor: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Relative movement, split into weights and BatchNorm buffers.

    The two are reported separately and on purpose. Buffers are not edits -- a
    strategy should not be charged for them -- but they *do* change the model's
    behaviour, so a refinement that moves them is not doing what it claims. The
    previous run measured only weights and therefore could not see the
    recalibration that produced its entire effect.

    ``num_batches_tracked`` is an integer counter, excluded from the norms and
    reported as a separate count: any change to it proves BatchNorm updated.
    """
    current = model.state_dict()

    def relative(names: list[str]) -> float:
        moved_sq, base_sq = 0.0, 0.0
        for name in names:
            base = anchor[name].detach().to(torch.float64)
            now = current[name].detach().to(base.device, torch.float64)
            moved_sq += float((now - base).pow(2).sum().item())
            base_sq += float(base.pow(2).sum().item())
        if base_sq <= 0.0:
            return 0.0
        return math.sqrt(moved_sq) / math.sqrt(base_sq)

    weight_names, buffer_names, counters = [], [], []
    for name, tensor in anchor.items():
        if name.endswith("num_batches_tracked"):
            counters.append(name)
        elif not torch.is_floating_point(tensor):
            continue
        elif name.endswith("weight") or name.endswith("bias"):
            weight_names.append(name)
        else:                       # running_mean, running_var
            buffer_names.append(name)

    counter_changes = sum(
        1 for name in counters
        if int(current[name].item()) != int(anchor[name].item())
    )

    return {
        "parameter_movement": relative(weight_names),
        "buffer_movement": relative(buffer_names),
        "batchnorm_counters_changed": counter_changes,
    }


def one_step(
    model: nn.Module, loader, device: str, lr: float, ascend: bool, batches: int
) -> float:
    """One SGD step on the mean gradient over ``batches`` batches.

    ``ascend`` flips the sign: ascent raises the loss on ``D_f`` (forgetting),
    descent lowers it on ``D_r`` (repair). Gradients are accumulated and a single
    ``step()`` is taken, so this is one update on a low-variance gradient rather
    than ``batches`` separate noisy ones.

    **The model stays in eval mode throughout.** Gradients flow perfectly well in
    eval mode; what eval mode changes is that BatchNorm uses its stored running
    statistics instead of recomputing and updating them from the batch. This
    architecture has no dropout, so eval mode costs nothing else. That is the
    whole fix: the refinement now tests gradient weight updates, which is what it
    was always supposed to test.
    """
    model.eval()
    assert_batchnorm_frozen(model)

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
    parser.add_argument("--max-retain-drop", type=float, default=0.01,
                        help="Reject if D_r_test accuracy falls by more than this.")
    parser.add_argument("--max-loss-ratio", type=float, default=1.25,
                        help="Reject if a retain LOSS grows by more than this "
                             "factor. Accuracy can hold while calibration "
                             "collapses, so loss is the earlier warning.")
    parser.add_argument("--max-edit-cost", type=float, default=0.30,
                        help="Reject if total movement from W_0 exceeds this "
                             "relative norm.")
    parser.add_argument("--max-buffer-movement", type=float, default=1e-9,
                        help="Reject if BatchNorm buffers move at all. Non-zero "
                             "means the steps recalibrated rather than updated "
                             "weights.")
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
    cstar_anchor: dict[str, Any] | None = None

    n_f = len(evaluator.loaders.forget_test.dataset)
    n_r = len(evaluator.loaders.retain_test.dataset)

    def record(label: str, model) -> dict[str, float]:
        # keep_per_sample so the MIA can be recomputed at every stage: a
        # refinement that improved forgetting while making membership MORE
        # detectable would be a bad trade, and invisible without it.
        measured = evaluator.measure_sets(
            model, keep_per_sample=True, include_retain_test=True
        )
        measured["selectivity_S"] = selectivity(
            measured["forget_train_loss"], measured["retain_train_loss"],
            original["forget_train_loss"], original["retain_train_loss"],
        )
        # Edit cost is always measured against W_0, never against the previous
        # stage: it is the total surgery the model has undergone, and the
        # gradient steps add to the chromosome's edit rather than replacing it.
        measured["edit_cost"] = relative_parameter_delta(
            model, evaluator._original_state
        )
        # Movement since C*, split so BatchNorm drift cannot hide inside a
        # weight-only number. Against the C* anchor, not W_0: this reports what
        # the REFINEMENT did, on top of what the chromosome already did.
        if cstar_anchor is not None:
            measured.update(movement_report(model, cstar_anchor))
        measured["full_test_acc"] = (
            n_f * measured["forget_test_acc"] + n_r * measured["retain_test_acc"]
        ) / (n_f + n_r)
        try:
            measured["mia_auc"] = compute_mia_auc(
                member_loss=evaluator._last_forget_train.per_sample_loss,
                member_confidence=evaluator._last_forget_train.per_sample_confidence,
                nonmember_loss=evaluator._last_forget_test.per_sample_loss,
                nonmember_confidence=evaluator._last_forget_test.per_sample_confidence,
            ).auc
        except Exception:  # noqa: BLE001 -- a diagnostic must not fail the run
            measured["mia_auc"] = float("nan")

        for key in ("forget_train_acc", "forget_test_acc",
                    "retain_train_acc", "retain_test_acc"):
            measured[f"gap_{key}"] = measured[key] - reference[key]

        measured["stage"] = label
        stages.append(measured)
        return measured

    # Stage 0 -- the strategy exactly as the search produced it.
    evaluator.evaluate(chromosome)
    model = evaluator.model
    # Full state_dict: weights AND BatchNorm buffers, so movement_report can
    # measure both against the same fixed point.
    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cstar_anchor = anchor

    # The model object sits in PyTorch's default training=True state: nothing in
    # the evaluator switches it globally, because every component that runs a
    # forward pass sets eval mode locally and restores it (metrics.evaluate and
    # the selector's _input_activation_norms both do). That is harmless for the
    # search -- no forward pass ever runs with BatchNorm live -- but the
    # refinement takes gradient steps here, so the mode has to be made explicit
    # rather than inherited.
    model.eval()
    frozen = assert_batchnorm_frozen(model)
    print(f"  BatchNorm     {frozen} modules verified frozen (eval mode)")
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
          f"{'D_r_test':>10}{'S':>10}{'dW':>10}{'dBN':>10}")
    for stage in stages:
        print(f"  {stage['stage']:<22}{stage['forget_train_acc']:>9.4f}"
              f"{stage['forget_test_acc']:>10.4f}{stage['retain_train_acc']:>9.4f}"
              f"{stage['retain_test_acc']:>10.4f}{stage['selectivity_S']:>10.3f}"
              f"{stage.get('parameter_movement', float('nan')):>10.6f}"
              f"{stage.get('buffer_movement', float('nan')):>10.6f}")
    print(f"  {'reference (target)':<22}{reference['forget_train_acc']:>9.4f}"
          f"{reference['forget_test_acc']:>10.4f}{reference['retain_train_acc']:>9.4f}"
          f"{reference['retain_test_acc']:>10.4f}{'--':>10}{'--':>10}{'--':>10}")
    print()
    print("  dW = parameter movement since C*,  dBN = BatchNorm buffer movement")

    before, after = stages[0], final
    drop = before["retain_test_acc"] - after["retain_test_acc"]

    # --- the acceptance rule ------------------------------------------------
    # Four conditions, ALL required. The point of stating them as data rather
    # than as one boolean is that a rejection has to say which one failed --
    # "the refinement did not help" is not a reportable finding on its own.
    forget_improved = after["forget_test_acc"] < before["forget_test_acc"]
    retain_held = drop <= args.max_retain_drop
    # Utility collapse: retain loss is the early-warning signal, because
    # accuracy can hold while the model becomes badly calibrated on D_r.
    no_collapse = (
        after["retain_train_loss"] <= before["retain_train_loss"] * args.max_loss_ratio
        and after["retain_test_loss"] <= before["retain_test_loss"] * args.max_loss_ratio
    )
    edit_reasonable = after["edit_cost"] <= args.max_edit_cost
    # Movement of the refinement itself, measured against C*.
    param_moved = after.get("parameter_movement", float("nan"))
    buffer_moved = after.get("buffer_movement", float("nan"))
    counters_changed = int(after.get("batchnorm_counters_changed", 0))
    param_in_budget = param_moved <= args.max_delta * 2 + 1e-12
    # BatchNorm must not have moved AT ALL. Anything else means the steps were
    # recalibrating rather than updating weights, which is the exact failure the
    # previous run hit and could not see.
    buffers_frozen = buffer_moved <= args.max_buffer_movement and counters_changed == 0

    checks = {
        "forget improved on D_f_test": forget_improved,
        f"D_r_test drop <= {args.max_retain_drop:.3f}": retain_held,
        f"no utility collapse (retain losses <= {args.max_loss_ratio}x)": no_collapse,
        f"edit cost <= {args.max_edit_cost}": edit_reasonable,
        f"parameter movement <= {args.max_delta * 2:.4f}": param_in_budget,
        "BatchNorm buffers unchanged": buffers_frozen,
    }
    accepted = all(checks.values())

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refinement.json").write_text(json.dumps({
        "front_position": member.get("front_position"),
        "chromosome": member["chromosome"],
        "operators": member.get("operators", ""),
        "hyperparameters": {
            "forget_step": "SGD gradient ASCENT on cross-entropy over D_f",
            "retain_step": "SGD gradient DESCENT on cross-entropy over D_r",
            "forget_lr": args.forget_lr,
            "retain_lr": args.retain_lr,
            "batches_per_step": args.batches,
            "steps": "one optimiser step per stage, on the mean gradient",
            "gradient_norm_clipping": None,
            "batchnorm": "FROZEN -- model held in eval mode for both steps; "
                         "running_mean/running_var/num_batches_tracked cannot update",
            "max_buffer_movement": args.max_buffer_movement,
            "movement_budget_relative": args.max_delta,
            "max_retain_test_drop": args.max_retain_drop,
            "max_loss_ratio": args.max_loss_ratio,
            "max_edit_cost": args.max_edit_cost,
            "seed": evaluator.seed,
        },
        "original_W0": original,
        "reference_Wref": reference,
        "stages": stages,
        "retain_test_drop": drop,
        "parameter_movement": param_moved,
        "buffer_movement": buffer_moved,
        "batchnorm_counters_changed": counters_changed,
        "batchnorm_frozen": True,
        "acceptance_checks": checks,
        "accepted": accepted,
    }, indent=2), encoding="utf-8")

    print("\n" + "-" * 100)
    print("ACCEPTANCE RULE")
    print("-" * 100)
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")

    print("\n" + "=" * 100)
    if not accepted:
        failed = [n for n, ok in checks.items() if not ok]
        print("REFINEMENT REJECTED")
        print(f"  failed: {'; '.join(failed)}")
        print(f"  D_f_test {before['forget_test_acc']:.4f} -> "
              f"{after['forget_test_acc']:.4f}")
        print(f"  D_r_test {before['retain_test_acc']:.4f} -> "
              f"{after['retain_test_acc']:.4f}  (drop {drop:+.4f})")
        print("\n  Checkpoint NOT written. C* from the Plan A search stands as")
        print("  the main result.")
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
        print("REFINEMENT ACCEPTED -- all four conditions hold")
        print(f"  D_f_test {before['forget_test_acc']:.4f} -> "
              f"{after['forget_test_acc']:.4f}")
        print(f"  D_r_test {before['retain_test_acc']:.4f} -> "
              f"{after['retain_test_acc']:.4f}  (drop {drop:+.4f})")
        print()
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")

    print(f"wrote {(out_dir / 'refinement.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
