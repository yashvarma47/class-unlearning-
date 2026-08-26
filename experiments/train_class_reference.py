"""Train W_ref: the model that never saw the forget class.

W_ref is trained on ``D_r_train`` only -- all 45 000 CIFAR-10 images that are not
frogs -- with the same recipe as the original model. It is the target the search
aims at: a model that has genuinely forgotten frogs should behave like this one,
not like a model with 0% frog accuracy.

Checkpoint selection is the point of this script
------------------------------------------------
The predecessor project selected its reference by accuracy on the FULL 10 000
image test set. For a retain-only reference that is the wrong criterion twice
over:

* **Diluted.** 1 000 of those images are frogs the model never trained on, so the
  number is capped near 0.90 and is not comparable with the original model's.
* **Backwards.** The frog logit is never positively trained, so frog test
  accuracy sits near zero -- but it fluctuates, and an epoch that happens to
  place a few more frogs in class 6 scores *higher*. That rewards the reference
  for recognising the thing it is supposed never to have seen.

Selecting after the fact from ``{best, latest, final}`` does not fix it either:
the best D_r_test epoch is usually none of those three. So D_r_test is measured
every epoch, here, during the run.

Selection rule
--------------
1. highest ``D_r_test`` accuracy
2. ties broken by lowest ``D_r_test`` loss (common once accuracy plateaus to
   three decimal places; the lower-loss model is the better-calibrated one)

``D_f_test`` is measured and logged every epoch as a **diagnostic only**. It must
never influence selection -- a reference that scores well on frogs is a worse
reference, not a better one.

Outputs ``*_best_dr.pt`` (use this one), plus ``_latest`` and ``_final``.

Run::

    python experiments/train_class_reference.py
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
import torch
import torch.nn as nn
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import (  # noqa: E402
    CIFAR10_CLASS_NAMES,
    build_class_loaders,
    get_or_create_class_split,
    load_cifar10,
    make_loader,
)
from medus_class.evaluation.metrics import evaluate  # noqa: E402
from medus_class.models import (  # noqa: E402
    CheckpointMetadata,
    build_model,
    save_checkpoint,
)
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from medus_class.utils.device import describe_environment, get_device  # noqa: E402
from medus_class.utils.seeding import seed_everything  # noqa: E402

CSV_COLUMNS = [
    "epoch", "train_loss", "train_acc",
    "retain_test_loss", "retain_test_acc",
    "forget_test_loss", "forget_test_acc",
    "full_test_acc",
    "lr", "epoch_time", "total_elapsed_time",
]


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """One pass over D_r_train. Returns ``(mean loss, accuracy)``."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite training loss ({loss.item()}); aborting rather than "
                f"writing a corrupt checkpoint"
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/cifar10_class.yaml")
    parser.add_argument("--model-config", default="model/resnet18.yaml")
    parser.add_argument("--base-config", default="base.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.base_config)
    cfg.update(load_config(args.config))
    cfg.update(load_config(args.model_config))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training = cfg["training"]
    forget_class = int(cfg["split"]["forget_class"])
    class_name = CIFAR10_CLASS_NAMES[forget_class]

    seed = int(cfg.get("seed", 42))
    seed_everything(seed, deterministic=False)  # throughput over bit-determinism
    device = get_device(prefer=cfg["device"]["prefer"], index=cfg["device"]["index"]).device

    run_name = args.run_name or f"class{forget_class}_{class_name}_reference"
    total_epochs = int(args.epochs or training["epochs"])

    print("=" * 100)
    print(f"TRAINING W_ref -- the model that never saw '{class_name}'")
    print("=" * 100)

    bundle = load_cifar10(data_cfg)
    split, created = get_or_create_class_split(
        train_labels=bundle.train_labels,
        test_labels=bundle.test_labels,
        forget_class=forget_class,
        path=cfg["split"]["split_file"],
    )
    # Evaluation loaders are forward-only over data already resident in memory,
    # so DataLoader workers buy nothing -- and on Windows each persistent pool is
    # a live process. Building all six through the default config spawned twelve
    # of them alongside the training loader's two, which cost ~50s/epoch in
    # contention. The training loader keeps its workers: augmentation is
    # genuinely CPU-bound and does benefit.
    eval_data_cfg = {**data_cfg, "num_workers": 0, "persistent_workers": False}
    loaders = build_class_loaders(bundle, split, eval_data_cfg, seed=seed)

    print(f"  forget class      {forget_class} ({class_name})")
    print(f"  split             {'created' if created else 'loaded'} "
          f"{cfg['split']['split_file']}")
    print(f"  D_f_train         {split.forget_train.size}   (EXCLUDED from training)")
    print(f"  D_r_train         {split.retain_train.size}   <- the training set")
    print(f"  D_f_test          {split.forget_test.size}   (diagnostic only)")
    print(f"  D_r_test          {split.retain_test.size}   <- selection criterion")
    print(f"  device            {device}")
    print(f"  epochs            {total_epochs}")

    train_loader = make_loader(
        Subset(bundle.train_augmented, split.retain_train.tolist()),
        int(data_cfg["batch_size"]["train"]), True, data_cfg, seed,
    )

    model = build_model(model_cfg, num_classes=int(data_cfg["num_classes"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["lr"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["lr_step_size"]),
        gamma=float(training["lr_gamma"]),
    )

    checkpoint_dir = resolve_path(cfg["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dr_path = checkpoint_dir / f"{run_name}_best_dr.pt"
    latest_path = checkpoint_dir / f"{run_name}_latest.pt"
    final_path = checkpoint_dir / f"{run_name}_final.pt"

    csv_path = PROJECT_ROOT / "results" / f"{run_name}_log.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()

    (csv_path.parent / f"{run_name}_environment.json").write_text(
        json.dumps(describe_environment(), indent=2), encoding="utf-8"
    )

    training_config = {
        "train_size": int(split.retain_train.size),
        "epochs": total_epochs,
        "batch_size": int(data_cfg["batch_size"]["train"]),
        "optimizer": {"name": "sgd", "lr": training["lr"],
                      "momentum": training["momentum"],
                      "weight_decay": training["weight_decay"]},
        "scheduler": {"name": "step", "step_size": training["lr_step_size"],
                      "gamma": training["lr_gamma"]},
        "precision": "fp32",
        "forget_class": forget_class,
        "run_name": run_name,
    }

    best_dr_acc, best_dr_loss, best_dr_epoch = -1.0, float("inf"), -1
    history: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    print("\n" + "-" * 100)
    print(f"{'epoch':>6} {'tr_loss':>9} {'tr_acc':>8} {'D_r_test':>10} "
          f"{'D_r_loss':>10} {'D_f_test':>10} {'lr':>8} {'elapsed':>10}   best_dr")
    print("-" * 100)

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        retain_test = evaluate(model, loaders.retain_test, device, collect_per_sample=False)
        forget_test = evaluate(model, loaders.forget_test, device, collect_per_sample=False)

        # D_f_test and D_r_test partition the test set exactly, so full-test
        # accuracy is their sample-weighted mean. Running a third pass over the
        # same 10 000 images would cost a third of the per-epoch evaluation
        # budget to recompute a number already in hand.
        n_f, n_r = forget_test.n_samples, retain_test.n_samples
        full_test_acc = (
            n_f * forget_test.accuracy + n_r * retain_test.accuracy
        ) / (n_f + n_r)

        scheduler.step()

        # SELECTION. D_r_test accuracy, ties broken by D_r_test loss.
        # forget_test is deliberately absent from this expression.
        is_best_dr = retain_test.accuracy > best_dr_acc or (
            retain_test.accuracy == best_dr_acc and retain_test.loss < best_dr_loss
        )
        if is_best_dr:
            best_dr_acc, best_dr_loss, best_dr_epoch = (
                retain_test.accuracy, retain_test.loss, epoch
            )

        epoch_time = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - run_start

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "retain_test_loss": round(retain_test.loss, 6),
            "retain_test_acc": round(retain_test.accuracy, 6),
            "forget_test_loss": round(forget_test.loss, 6),
            "forget_test_acc": round(forget_test.accuracy, 6),
            "full_test_acc": round(full_test_acc, 6),
            "lr": round(current_lr, 8),
            "epoch_time": round(epoch_time, 2),
            "total_elapsed_time": round(total_elapsed, 2),
        }
        history.append(row)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerow(row)

        print(f"{epoch:>6} {train_loss:>9.4f} {train_acc:>8.4f} "
              f"{retain_test.accuracy:>10.4f} {retain_test.loss:>10.4f} "
              f"{forget_test.accuracy:>10.4f} {current_lr:>8.5f} "
              f"{total_elapsed/60:>9.1f}m   {'<-- NEW' if is_best_dr else ''}",
              flush=True)

        metadata = CheckpointMetadata(
            model_name=model_cfg["name"],
            dataset=data_cfg["name"],
            seed=seed,
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "retain_test_acc": retain_test.accuracy,
                "retain_test_loss": retain_test.loss,
                "forget_test_acc": forget_test.accuracy,
                "forget_test_loss": forget_test.loss,
                "full_test_acc": full_test_acc,
                "best_retain_test_acc": best_dr_acc,
                "best_retain_test_loss": best_dr_loss,
                "forget_class": forget_class,
            },
            split_file=str(cfg["split"]["split_file"]),
            training_config=training_config,
            notes=(
                f"W_ref for CLASS unlearning: trained on D_r only "
                f"({split.retain_train.size} images), class {forget_class} "
                f"({class_name}) EXCLUDED. Checkpoint selection is on D_r_test "
                f"accuracy then D_r_test loss; D_f_test is diagnostic only."
            ),
        )

        save_checkpoint(latest_path, model, metadata, optimizer, scheduler)
        if is_best_dr:
            save_checkpoint(best_dr_path, model, metadata, optimizer, scheduler)

    save_checkpoint(final_path, model, metadata)

    best_row = max(history, key=lambda r: (r["retain_test_acc"], -r["retain_test_loss"]))
    best_full = max(history, key=lambda r: r["full_test_acc"])

    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)
    print(f"  epochs                 {total_epochs}")
    print(f"  total time             {history[-1]['total_elapsed_time']/3600:.2f} h")
    print(f"\n  SELECTED (D_r_test accuracy, then loss)")
    print(f"    epoch                {best_row['epoch']}")
    print(f"    D_r_test acc / loss  {best_row['retain_test_acc']:.4f} / "
          f"{best_row['retain_test_loss']:.4f}")
    print(f"    D_f_test acc         {best_row['forget_test_acc']:.4f}   "
          f"<- diagnostic; near zero is CORRECT")
    print(f"    checkpoint           {best_dr_path.name}")

    if best_full["epoch"] != best_row["epoch"]:
        print(f"\n  Selecting on FULL test accuracy would have picked epoch "
              f"{best_full['epoch']} instead")
        print(f"  (D_r_test {best_full['retain_test_acc']:.4f} vs "
              f"{best_row['retain_test_acc']:.4f}). This is exactly why "
              f"_best_dr exists.")
    else:
        print(f"\n  Full-test accuracy would have picked the same epoch this time.")
        print(f"  It is still the wrong criterion; it merely agreed here.")

    print(f"\n  log                    {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  Point evaluation.reference_checkpoint at {best_dr_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
