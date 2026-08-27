"""Install a validated reference zip into the project's standard result folders.

The last step of the Kaggle round trip: train on Kaggle, download the zip,
validate it, then import it here so the checkpoint sits exactly where the frog
reference sits and every config path works without special cases.

Where things land
-----------------

===========================================  ==================================
``results/checkpoints/<run>_best_dr.pt``     the reference; what configs point at
``results/checkpoints/<run>_best_dr.json``   provenance sidecar
``results/<run>_log.csv``                    per-epoch history
``results/<run>_environment.json``           torch / CUDA / GPU snapshot
``results/splits/cifar10_class<ID>_<name>.json``  the split it was trained on
``results/reference_training/<...>``         summary and the Kaggle manifest
===========================================  ==================================

with ``<run> = class<ID>_<name>_reference``, matching
``class6_frog_reference_best_dr.pt`` exactly.

Refusals
--------
* **Not validated.** The class must have a ``PASS`` row in
  ``reference_validation_summary.csv`` naming this zip. Import is not the place
  to decide whether a model is good; it is the place to refuse one that was
  never checked.
* **Split disagreement.** If a split file for the class already exists locally
  and differs from the one in the zip, the model was trained on a different
  partition than the one this project would use. That is silent and fatal, so
  it stops here.
* **Overwrite.** An existing destination file stops the run unless
  ``--overwrite`` is passed.

Run::

    python kaggle/reference_training/import_reference_zip.py \\
        --zip reference_outputs_trial_ship.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.utils.config import resolve_path  # noqa: E402

CHECKPOINT_RE = re.compile(r"^class(\d)_([a-z]+)_reference_best_dr\.pt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def passed_validation(class_id: int, zip_name: str, summary: Path) -> tuple[bool, str]:
    """Has this exact zip been validated for this class?"""
    if not summary.is_file():
        return False, f"no validation summary at {summary}"
    with summary.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matching = [r for r in rows
                if str(r.get("class_id")) == str(class_id)
                and r.get("source_zip") == zip_name]
    if not matching:
        return False, (f"no row for class {class_id} from {zip_name} in "
                       f"{summary.name}")
    # Last row wins: validation appends, so a re-validation supersedes.
    verdict = matching[-1].get("verdict")
    if verdict != "PASS":
        return False, f"validation verdict for class {class_id} is {verdict}"
    return True, "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--summary",
                        default="results/reference_training/"
                                "reference_validation_summary.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be written, copy nothing")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.is_absolute():
        zip_path = (Path.cwd() / zip_path).resolve()
    if not zip_path.is_file():
        raise SystemExit(f"zip not found: {zip_path}")

    print("=" * 100)
    print(f"IMPORTING  {zip_path.name}  ({zip_path.stat().st_size:,} bytes)")
    print("=" * 100)
    print(f"  sha256  {sha256(zip_path)}")

    with tempfile.TemporaryDirectory() as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise SystemExit(f"zip is corrupt at member: {bad}")
            archive.extractall(extracted)

        checkpoints = sorted(p for p in extracted.rglob("*.pt")
                             if CHECKPOINT_RE.match(p.name))
        if not checkpoints:
            raise SystemExit("no *_reference_best_dr.pt in the zip")

        summary_path = resolve_path(args.summary)
        planned: list[tuple[Path, Path]] = []

        for checkpoint in checkpoints:
            match = CHECKPOINT_RE.match(checkpoint.name)
            class_id, name = int(match.group(1)), match.group(2)
            if CIFAR10_CLASS_NAMES[class_id] != name:
                raise SystemExit(
                    f"{checkpoint.name}: class {class_id} is "
                    f"'{CIFAR10_CLASS_NAMES[class_id]}', not '{name}'"
                )

            ok, why = passed_validation(class_id, zip_path.name, summary_path)
            print(f"\n  class {class_id} ({name})")
            print(f"    validation      {why}")
            if not ok:
                raise SystemExit(
                    f"    REFUSING to import an unvalidated reference. Run:\n"
                    f"      python kaggle/reference_training/"
                    f"validate_reference_zip.py --zip {zip_path.name} "
                    f"--expect-class {class_id}"
                )

            run = f"class{class_id}_{name}_reference"
            split_name = f"cifar10_class{class_id}_{name}.json"

            # The split is the one thing that must not merely be copied: if a
            # local split already exists and differs, every accuracy in this
            # project is defined on a different partition than the model was
            # trained on.
            zip_split = extracted / split_name
            local_split = resolve_path(f"results/splits/{split_name}")
            if zip_split.is_file() and local_split.is_file():
                if json.loads(zip_split.read_text(encoding="utf-8")) != \
                        json.loads(local_split.read_text(encoding="utf-8")):
                    raise SystemExit(
                        f"    ABORT: {split_name} in the zip differs from the "
                        f"local one. The model was trained on a different "
                        f"partition; importing it would make every downstream "
                        f"number wrong."
                    )
                print(f"    split           matches local {split_name}")

            mapping = [
                (checkpoint, f"results/checkpoints/{run}_best_dr.pt"),
                (extracted / f"{run}_best_dr.json",
                 f"results/checkpoints/{run}_best_dr.json"),
                (extracted / f"class{class_id}_{name}_training_log.csv",
                 f"results/{run}_log.csv"),
                (extracted / f"class{class_id}_{name}_environment.json",
                 f"results/{run}_environment.json"),
                (zip_split, f"results/splits/{split_name}"),
                (extracted / f"class{class_id}_{name}_training_summary.md",
                 f"results/reference_training/"
                 f"class{class_id}_{name}_training_summary.md"),
                (extracted / "manifest.json",
                 f"results/reference_training/"
                 f"class{class_id}_{name}_kaggle_manifest.json"),
            ]
            for source, destination in mapping:
                if not source.is_file():
                    print(f"    (absent in zip, skipped) {source.name}")
                    continue
                target = resolve_path(destination)
                if target.is_file() and not args.overwrite:
                    if source.name.startswith("cifar10_class"):
                        continue  # already checked identical above
                    raise SystemExit(
                        f"    ABORT: {destination} already exists. Pass "
                        f"--overwrite only if you mean to replace it."
                    )
                planned.append((source, target))

        print("\n" + "-" * 100)
        print("  WRITING" if not args.dry_run else "  WOULD WRITE (dry run)")
        print("-" * 100)
        for source, target in planned:
            rel = target.relative_to(PROJECT_ROOT)
            print(f"    {source.stat().st_size:>12,}  {rel}")
            if not args.dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    print("\n" + "=" * 100)
    print("DRY RUN -- nothing written" if args.dry_run
          else f"IMPORTED {len(planned)} file(s)")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
