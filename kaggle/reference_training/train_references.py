"""Train the retain-only reference (``W_ref``) for each class in an assignment.

One driver, one assignment file, no guesswork. It reads a YAML listing exactly
which classes the person at the keyboard is responsible for, prints a safety
block for each, refuses anything that does not add up, and then hands the actual
training to ``experiments/train_class_reference.py`` -- unmodified, the same
script and the same recipe that produced the finished frog reference. Nothing
about the training loop lives here, deliberately: a second implementation would
drift from the frog reference and the ten models would stop being comparable.

What it will not do
-------------------
* **Train every class by accident.** It iterates the assignment's ``classes``
  list and nothing else. A missing or empty list is a hard error, not a licence
  to train all ten.
* **Retrain frog.** Class 6's reference is finished and committed, and every
  existing frog result was measured against it. Class 6 in an assignment is
  refused outright.
* **Start on a bad split.** The four split sizes are checked before the first
  epoch, here and again inside the trainer. Four hours of GPU time spent on a
  set nobody can name is the failure this prevents.
* **Overwrite a finished reference.** An existing ``*_best_dr.pt`` for the class
  stops the run unless ``--overwrite`` is passed.

Run::

    python kaggle/reference_training/train_references.py \\
        --assignment kaggle/reference_training/trial_ship.yaml

    # print the plan and the split sizes, train nothing
    python kaggle/reference_training/train_references.py \\
        --assignment kaggle/reference_training/trial_ship.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medus_class.data import (  # noqa: E402
    CIFAR10_CLASS_NAMES,
    build_class_split,
    load_cifar10,
    save_class_split,
)
from medus_class.models import read_metadata  # noqa: E402
from medus_class.utils.config import load_config, resolve_path  # noqa: E402
from medus_class.utils.device import describe_environment  # noqa: E402

#: ``(D_f_train, D_r_train, D_f_test, D_r_test)`` -- see the trainer's constant.
EXPECTED_SPLIT_SIZES = (5_000, 45_000, 1_000, 9_000)

#: Its reference is finished, committed, and is what every published frog number
#: was measured against. Retraining it would silently invalidate them.
FROZEN_CLASSES = {6: "frog"}

#: Stated in every safety block, because it is the one thing about this training
#: run that is easy to get wrong and impossible to notice afterwards.
SELECTION_RULE = (
    "best checkpoint selected by D_r_test accuracy, with D_r_test loss as "
    "tie-breaker (D_f_test is logged every epoch but NEVER influences selection)"
)


def load_assignment(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read and validate an assignment YAML.

    Raises
    ------
    SystemExit
        On anything ambiguous. Every check here is cheap and every one of them
        has a failure mode that would otherwise cost hours of GPU time.
    """
    if not path.is_file():
        raise SystemExit(f"assignment file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    person = payload.get("assigned_person")
    if not person:
        raise SystemExit(f"{path.name}: 'assigned_person' is missing")

    classes = payload.get("classes")
    if not classes:
        raise SystemExit(
            f"{path.name}: 'classes' is missing or empty. This is an error on "
            f"purpose -- there is no 'train everything' default."
        )
    if not isinstance(classes, list):
        raise SystemExit(f"{path.name}: 'classes' must be a list")

    seen: set[int] = set()
    for entry in classes:
        if not isinstance(entry, dict) or "id" not in entry or "name" not in entry:
            raise SystemExit(f"{path.name}: every class needs an 'id' and a 'name'")
        class_id, name = int(entry["id"]), str(entry["name"])

        if not 0 <= class_id < len(CIFAR10_CLASS_NAMES):
            raise SystemExit(f"{path.name}: class id {class_id} is not a CIFAR-10 label")
        # The name is redundant with the id, which is exactly why it is checked:
        # a mismatch means the file was hand-edited and one of the two is wrong.
        if CIFAR10_CLASS_NAMES[class_id] != name:
            raise SystemExit(
                f"{path.name}: class {class_id} is "
                f"'{CIFAR10_CLASS_NAMES[class_id]}', not '{name}'. Fix the file "
                f"rather than the code."
            )
        if class_id in FROZEN_CLASSES:
            raise SystemExit(
                f"{path.name}: class {class_id} ({FROZEN_CLASSES[class_id]}) is "
                f"already finished and must not be retrained -- every published "
                f"result for it was measured against the committed checkpoint."
            )
        if class_id in seen:
            raise SystemExit(f"{path.name}: class {class_id} is listed twice")
        seen.add(class_id)

    return str(person), [{"id": int(e["id"]), "name": str(e["name"])} for e in classes]


def preflight(class_id: int, name: str, cfg: dict[str, Any]) -> tuple[int, ...]:
    """Build (or load) the class split and return its four sizes.

    Runs before any training so a wrong split costs seconds rather than hours.
    """
    bundle = load_cifar10(cfg["data"])
    split = build_class_split(bundle.train_labels, bundle.test_labels, class_id)
    split_path = resolve_path(
        f"results/splits/cifar10_class{class_id}_{name}.json"
    )
    if not split_path.is_file():
        save_class_split(split, split_path)
    return (
        int(split.forget_train.size), int(split.retain_train.size),
        int(split.forget_test.size), int(split.retain_test.size),
    )


def write_summary(out_path: Path, class_id: int, name: str, person: str,
                  log_rows: list[dict[str, str]], metadata: dict[str, Any],
                  environment: dict[str, Any]) -> None:
    """Write the human-readable per-class training summary."""
    best = max(log_rows, key=lambda r: (float(r["retain_test_acc"]),
                                        -float(r["retain_test_loss"])))
    last = log_rows[-1]
    gpu = (environment.get("device") or {}).get("gpu_name", "unknown")
    hours = float(last["total_elapsed_time"]) / 3600

    out_path.write_text(
        f"# W_ref for class {class_id} ({name})\n"
        f"\n"
        f"Retain-only reference: trained on `D_r_train` only, the "
        f"{EXPECTED_SPLIT_SIZES[1]:,} CIFAR-10 training images that are **not** "
        f"{name}. It never saw a single {name}.\n"
        f"\n"
        f"| | |\n|---|---|\n"
        f"| assigned to | {person} |\n"
        f"| forget class | {class_id} ({name}) |\n"
        f"| epochs | {last['epoch']} |\n"
        f"| seed | {metadata.get('seed')} |\n"
        f"| GPU | {gpu} |\n"
        f"| wall clock | {hours:.2f} h |\n"
        f"| selection rule | {SELECTION_RULE} |\n"
        f"\n"
        f"## Selected checkpoint\n"
        f"\n"
        f"| | |\n|---|---|\n"
        f"| epoch | {best['epoch']} |\n"
        f"| `D_r_test` accuracy | {float(best['retain_test_acc']):.4f} |\n"
        f"| `D_r_test` loss | {float(best['retain_test_loss']):.4f} |\n"
        f"| `D_f_test` accuracy | {float(best['forget_test_acc']):.4f} "
        f"(diagnostic only -- near zero is CORRECT) |\n"
        f"| `D_f_test` loss | {float(best['forget_test_loss']):.4f} |\n"
        f"| full test accuracy | {float(best['full_test_acc']):.4f} "
        f"(diluted by the held-out class; not a utility number) |\n"
        f"\n"
        f"File: `class{class_id}_{name}_reference_best_dr.pt`\n"
        f"\n"
        f"## Split\n"
        f"\n"
        f"| set | size |\n|---|---:|\n"
        f"| `D_f_train` (excluded from training) | {EXPECTED_SPLIT_SIZES[0]:,} |\n"
        f"| `D_r_train` (the training set) | {EXPECTED_SPLIT_SIZES[1]:,} |\n"
        f"| `D_f_test` (diagnostic) | {EXPECTED_SPLIT_SIZES[2]:,} |\n"
        f"| `D_r_test` (selection criterion) | {EXPECTED_SPLIT_SIZES[3]:,} |\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", required=True,
                        help="path to an assignment YAML, e.g. trial_ship.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the config's 200; leave alone for a real run")
    parser.add_argument("--staging-dir", default="results/reference_training/staging")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and split sizes, train nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="retrain a class whose reference already exists")
    args = parser.parse_args()

    assignment_path = Path(args.assignment)
    if not assignment_path.is_absolute():
        assignment_path = (PROJECT_ROOT / assignment_path).resolve()
    person, classes = load_assignment(assignment_path)

    cfg = load_config("base.yaml")
    cfg.update(load_config("data/cifar10_class.yaml"))
    cfg.update(load_config("model/resnet18.yaml"))
    seed = int(cfg.get("seed", 42))
    epochs = int(args.epochs or cfg["training"]["epochs"])
    environment = describe_environment()
    gpu = (environment.get("device") or {}).get("gpu_name", "unknown")
    device_str = (environment.get("device") or {}).get("device", "unknown")

    staging = resolve_path(args.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("W_ref REFERENCE TRAINING")
    print("=" * 100)
    print(f"  assignment file   {assignment_path.name}")
    print(f"  assigned person   {person}")
    print(f"  classes to train  {len(classes)}: "
          + ", ".join(f"{c['id']} ({c['name']})" for c in classes))
    print(f"  frozen, never trained here: "
          + ", ".join(f"{k} ({v})" for k, v in FROZEN_CLASSES.items()))
    if args.dry_run:
        print("  DRY RUN -- nothing will be trained")

    trainer = PROJECT_ROOT / "experiments" / "train_class_reference.py"
    trained: list[dict[str, Any]] = []

    for position, entry in enumerate(classes, start=1):
        class_id, name = entry["id"], entry["name"]
        run_name = f"class{class_id}_{name}_reference"
        best_dr = resolve_path(f"results/checkpoints/{run_name}_best_dr.pt")

        sizes = preflight(class_id, name, cfg)

        print("\n" + "-" * 100)
        print(f"  [{position}/{len(classes)}]  SAFETY CHECK")
        print("-" * 100)
        print(f"    assigned person      {person}")
        print(f"    forget class id      {class_id}")
        print(f"    forget class name    {name}")
        print(f"    D_f_train            {sizes[0]}   (EXCLUDED from training)")
        print(f"    D_r_train            {sizes[1]}   <- the training set")
        print(f"    D_f_test             {sizes[2]}   (diagnostic only)")
        print(f"    D_r_test             {sizes[3]}   <- selection criterion")
        print(f"    seed                 {seed}")
        print(f"    epochs               {epochs}")
        print(f"    device               {device_str}  ({gpu})")
        print(f"    output checkpoint    {best_dr}")
        print(f"    selection rule       {SELECTION_RULE}")

        if sizes != EXPECTED_SPLIT_SIZES:
            print(f"\n    ABORT: expected split sizes {EXPECTED_SPLIT_SIZES}, "
                  f"measured {sizes}")
            return 1
        print(f"    split sizes          OK")

        if best_dr.is_file() and not args.overwrite:
            existing = read_metadata(best_dr)
            print(f"\n    ABORT: {best_dr.name} already exists "
                  f"(epoch {existing.get('epoch')}). Pass --overwrite only if "
                  f"you mean to replace a finished reference.")
            return 1

        if args.dry_run:
            print("\n    dry run -- skipping training")
            continue

        command = [
            sys.executable, str(trainer),
            "--forget-class", str(class_id),
            "--epochs", str(epochs),
        ]
        print(f"\n    running: {' '.join(command)}\n", flush=True)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
        if completed.returncode != 0:
            print(f"\n    TRAINING FAILED for class {class_id} ({name}) "
                  f"-- exit code {completed.returncode}")
            return completed.returncode

        # --- collect, under the names the validator expects ----------------
        log_source = resolve_path(f"results/{run_name}_log.csv")
        env_source = resolve_path(f"results/{run_name}_environment.json")
        split_source = resolve_path(
            f"results/splits/cifar10_class{class_id}_{name}.json")

        shutil.copy2(best_dr, staging / f"{run_name}_best_dr.pt")
        shutil.copy2(best_dr.with_suffix(".json"),
                     staging / f"{run_name}_best_dr.json")
        shutil.copy2(log_source, staging / f"class{class_id}_{name}_training_log.csv")
        shutil.copy2(split_source, staging / split_source.name)
        if env_source.is_file():
            shutil.copy2(env_source,
                         staging / f"class{class_id}_{name}_environment.json")

        with log_source.open(encoding="utf-8-sig") as handle:
            log_rows = list(csv.DictReader(handle))
        write_summary(
            staging / f"class{class_id}_{name}_training_summary.md",
            class_id, name, person, log_rows,
            read_metadata(best_dr), environment,
        )
        trained.append({"id": class_id, "name": name,
                        "epochs": len(log_rows),
                        "hours": round(float(log_rows[-1]["total_elapsed_time"])
                                       / 3600, 3)})
        print(f"\n    collected into {staging}")

    manifest = {
        "assignment_file": assignment_path.name,
        "assigned_person": person,
        "requested_classes": classes,
        "trained_classes": trained,
        "seed": seed,
        "epochs": epochs,
        "selection_rule": SELECTION_RULE,
        "expected_split_sizes": {
            "D_f_train": EXPECTED_SPLIT_SIZES[0],
            "D_r_train": EXPECTED_SPLIT_SIZES[1],
            "D_f_test": EXPECTED_SPLIT_SIZES[2],
            "D_r_test": EXPECTED_SPLIT_SIZES[3],
        },
        "environment": environment,
        "dry_run": bool(args.dry_run),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    if args.dry_run:
        print(f"DRY RUN COMPLETE -- {len(classes)} class(es) validated, none trained")
    else:
        print(f"DONE -- trained {len(trained)} of {len(classes)} requested class(es)")
        for row in trained:
            print(f"  class {row['id']} ({row['name']}): "
                  f"{row['epochs']} epochs, {row['hours']:.2f} h")
    print(f"  staging directory  {staging}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
