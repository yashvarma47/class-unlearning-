"""Build a clean Kaggle upload bundle, and the manifest of what must go with it.

Produces TWO artefacts, because Kaggle treats them differently:

``medus_class_code.zip``
    Source, configs and tests. Small (tens of KB). Upload as a Kaggle Dataset.

``medus_class_weights/``
    A staging directory of the checkpoints and the class split. Upload as a
    SECOND Kaggle Dataset -- weights are ~90 MB each and change on a different
    schedule from the code, so bundling them together would mean re-uploading
    170 MB every time a config comment is edited.

What is excluded, and why it matters
------------------------------------
* ``results/`` (except the class-structure measurement and the split) -- run
  outputs are reproducible and would make the bundle misleading: a stale
  ``pareto_front.csv`` shipped to Kaggle could be mistaken for that run's own
  output.
* ``data/`` -- 341 MB of CIFAR-10 that Kaggle can download or mount.
* ``__pycache__``, ``.pytest_cache``, ``.git``, ``.venv`` -- caches and history.
* **Anything from the instance-unlearning project.** That repository is a
  separate experiment held at commit a6005ef; nothing from it belongs here. The
  bundle is built only from this project's tree, so this is structural rather
  than a filter that could miss something.

Two profiles, because the two Kaggle workflows need different things
--------------------------------------------------------------------
``--profile search`` (default)
    A MED-US run. Stages ``W_0``, the split for ``--forget-class``, and that
    class's ``W_ref`` once it is finished.

``--profile reference``
    TRAINING a ``W_ref`` on Kaggle. Code bundle **only**. Reference training
    starts from random initialisation and derives its own split from the
    CIFAR-10 labels, so there is nothing for a weights bundle to carry -- and
    shipping one is actively harmful: a bundle built for cat/deer/dog once went
    out carrying a *frog* split and a frog-era ``W_0``, which is exactly the
    kind of thing that makes a reviewer doubt every number in the sweep.

Run::

    # reference training (Pragati, Aditya, and the rest of the sweep)
    python scripts/package_for_kaggle.py --profile reference

    # a search on a given class
    python scripts/package_for_kaggle.py --profile search --forget-class 8
    python scripts/package_for_kaggle.py --forget-class 6 --require-reference
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT  # noqa: E402

#: Directories copied into the code bundle in full.
CODE_TREES = ("src", "configs", "experiments", "tests", "kaggle", "scripts")

#: Individual files copied into the code bundle.
CODE_FILES = ("README.md", "requirements.txt", "requirements-torch.txt")

#: Never included, at any depth.
EXCLUDE_PATTERNS = (
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".git", ".venv",
    "*.pt", "*.pth",            # weights travel in the other bundle
    "*.out", "*.err", "*.log",  # local run logs
    ".DS_Store", "Thumbs.db",
)

#: W_0 is needed by every SEARCH, whatever the class: there is one original
#: model and every class is unlearned from it.
ORIGINAL_ASSET = (
    "results/checkpoints/cifar10_resnet18_seed42_best.pt",
    "W_0 -- the original model, trained on all 50 000 images. "
    "The model being unlearned FROM.",
)


def split_asset(class_id: int) -> tuple[str, str]:
    """The class split a SEARCH needs, for whichever class it is searching.

    Was hard-coded to frog. That was harmless while frog was the only class,
    and became a real hazard the moment a bundle built for reference training
    shipped a frog split next to it -- see --profile below.
    """
    name = CIFAR10_CLASS_NAMES[class_id]
    return (
        f"results/splits/cifar10_class{class_id}_{name}.json",
        f"The class-{class_id} ({name}) split: D_f/D_r over train and test. "
        f"Regenerable, but shipping it means Kaggle cannot silently build a "
        f"different one.",
    )


def reference_asset(class_id: int) -> tuple[str, str]:
    """The retain-only reference a SEARCH aims at. Absent until it is trained."""
    name = CIFAR10_CLASS_NAMES[class_id]
    return (
        f"results/checkpoints/class{class_id}_{name}_reference_best_dr.pt",
        f"W_ref for class {class_id} ({name}) -- trained on D_r only, selected "
        f"on D_r_test accuracy then loss. MUST be the _best_dr file, NOT _best.",
    )

#: Small measurement outputs worth shipping: they motivate the whole experiment
#: and cost nothing to carry.
OPTIONAL_ASSETS = (
    "results/analysis/class_structure/summary.json",
    "results/analysis/class_structure/per_class_groups.csv",
)


def reference_training_state(path: Path) -> tuple[bool, str]:
    """Is this W_ref checkpoint from a FINISHED run, or a live one?

    ``train_class_reference.py`` rewrites ``_best_dr.pt`` every time D_r_test
    improves, so the file exists and loads perfectly from the first epoch
    onwards. Existence therefore proves nothing about completeness, and shipping
    an epoch-30 checkpoint as W_ref would silently invalidate the entire Kaggle
    run -- the search would be aiming at a half-trained reference while every
    report called it "the model that never saw a frog".

    The checkpoint records both its own epoch and the run's planned total, so
    the two can simply be compared.

    Returns
    -------
    tuple
        ``(is_complete, human-readable description)``.
    """
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
    except Exception as exc:  # noqa: BLE001
        return False, f"could not be read ({type(exc).__name__}: {exc})"

    epoch = metadata.get("epoch")
    planned = (metadata.get("training_config") or {}).get("epochs")
    metrics = metadata.get("metrics", {})
    accuracy = metrics.get("retain_test_acc")

    if epoch is None or planned is None:
        return False, "carries no epoch metadata; cannot verify completeness"

    # The best D_r_test epoch is rarely the last one, so equality is not
    # required -- but a checkpoint from the first half of training is
    # unambiguously live. The threshold is deliberately loose: it is here to
    # catch "training is still running", not to second-guess the selection.
    description = (
        f"epoch {epoch} of {planned} planned, D_r_test acc "
        f"{accuracy:.4f}" if accuracy is not None else f"epoch {epoch}/{planned}"
    )
    return epoch >= planned * 0.9, description


def excluded(path: Path) -> bool:
    return any(
        fnmatch.fnmatch(part, pattern)
        for part in path.parts
        for pattern in EXCLUDE_PATTERNS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def collect_code_files() -> list[Path]:
    """Every file destined for the code bundle, project-relative."""
    files: list[Path] = []
    for tree in CODE_TREES:
        root = PROJECT_ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not excluded(path.relative_to(PROJECT_ROOT)):
                files.append(path.relative_to(PROJECT_ROOT))

    for name in CODE_FILES:
        if (PROJECT_ROOT / name).is_file():
            files.append(Path(name))

    for name in OPTIONAL_ASSETS:
        if (PROJECT_ROOT / name).is_file():
            files.append(Path(name))

    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="dist")
    parser.add_argument(
        "--profile", choices=("search", "reference"), default="search",
        help="'search' (default) stages W_0, the class split and W_ref for a "
             "MED-US run. 'reference' is for TRAINING a W_ref on Kaggle: it "
             "builds the code bundle ONLY, because reference training starts "
             "from random initialisation and derives its own split -- there is "
             "nothing for a weights bundle to carry.",
    )
    parser.add_argument("--forget-class", type=int, default=6,
                        help="which class the SEARCH bundle is for; ignored by "
                             "--profile reference")
    parser.add_argument("--require-reference", action="store_true",
                        help="Fail if W_ref is missing. Use once training has "
                             "finished, to guarantee a complete bundle.")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("KAGGLE UPLOAD BUNDLE")
    print("=" * 100)

    # --- 1. code bundle ---------------------------------------------------
    code_files = collect_code_files()
    zip_path = out_dir / "medus_class_code.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in code_files:
            archive.write(PROJECT_ROOT / relative,
                          arcname=str(Path("MEDUS_Class_Unlearning") / relative))

    print(f"\n  CODE BUNDLE  {zip_path.name}")
    print(f"    files      {len(code_files)}")
    print(f"    size       {human(zip_path.stat().st_size)}")
    print(f"    path       {zip_path}")

    # --- 2. weights staging ------------------------------------------------
    # Reference TRAINING needs no weights at all: it starts from random
    # initialisation and derives its own split from the CIFAR-10 labels. A
    # weights directory here would ship a checkpoint and a split for some class
    # nobody in the batch is training -- which is precisely the confusion that
    # prompted --profile: a bundle for cat/deer/dog/horse/truck arrived carrying
    # a frog split.
    weights_dir = out_dir / "medus_class_weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)

    manifest_entries: list[dict[str, Any]] = []
    assets: list[tuple[str, str]] = []
    reference_ready = False
    reference_state = "n/a -- reference profile ships no weights"

    if args.profile == "reference":
        print("\n  WEIGHTS BUNDLE  none (--profile reference)")
        print("    Reference training starts from random initialisation and")
        print("    builds its own split per class. Do NOT attach a weights")
        print("    dataset to the reference-training notebook.")
    else:
        weights_dir.mkdir(parents=True)
        assets = [ORIGINAL_ASSET, split_asset(args.forget_class)]
        reference_relative, reference_description = reference_asset(args.forget_class)
        reference_path = PROJECT_ROOT / reference_relative

        reference_state = "not produced yet"
        if reference_path.is_file():
            reference_ready, reference_state = reference_training_state(reference_path)
            if reference_ready:
                assets.append((reference_relative, reference_description))

        print(f"\n  WEIGHTS BUNDLE  {weights_dir.name}/")

    for relative, description in assets:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            print(f"    MISSING   {relative}")
            continue
        destination = weights_dir / Path(relative).name
        shutil.copy2(source, destination)
        entry = {
            "file": destination.name,
            "from": relative,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "purpose": description,
        }
        manifest_entries.append(entry)
        print(f"    {destination.name:<45} {human(entry['bytes']):>10}  "
              f"{entry['sha256'][:12]}")

    # --- 3. the manifest ---------------------------------------------------
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "MEDUS_Class_Unlearning",
        "profile": args.profile,
        "forget_class": None if args.profile == "reference" else args.forget_class,
        "forget_class_name": (
            None if args.profile == "reference"
            else CIFAR10_CLASS_NAMES[args.forget_class]),
        "code_bundle": {
            "file": zip_path.name,
            "files": len(code_files),
            "bytes": zip_path.stat().st_size,
        },
        "weights_bundle": manifest_entries,
        "reference_included": reference_ready,
        "reference_state": reference_state,
        "upload_as": (
            {"medus-class-code": "Kaggle Dataset from medus_class_code.zip"}
            if args.profile == "reference" else
            {"medus-class-code": "Kaggle Dataset from medus_class_code.zip",
             "medus-class-weights": "Kaggle Dataset from medus_class_weights/"}
        ),
        "excluded_deliberately": [
            "data/ -- CIFAR-10, downloaded or mounted on Kaggle",
            "results/search/ -- run outputs, reproducible",
            "results/checkpoints/*_latest.pt, *_final.pt -- only _best_dr is used",
            "the instance-unlearning project -- a separate experiment at a6005ef",
            "__pycache__, .pytest_cache, .git, .venv",
        ],
    }
    manifest_path = out_dir / "kaggle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n  MANIFEST     {manifest_path.name}")

    if args.profile == "reference":
        print()
        print("  Upload ONLY medus_class_code.zip, as the Kaggle Dataset")
        print("  'medus-class-code'. There is no second dataset.")
        print()
        print("  No split and no checkpoint is shipped on purpose:")
        print("  kaggle/reference_training/train_references.py builds the split")
        print("  for each assigned class from the CIFAR-10 labels, verifies its")
        print("  four sizes before the first epoch, and refuses class 6 (frog),")
        print("  whose reference is already finished.")
    elif not reference_ready:
        forget_name = CIFAR10_CLASS_NAMES[args.forget_class]
        print("\n" + "!" * 100)
        print("  W_ref IS NOT INCLUDED -- it is not from a finished run.")
        print(f"    {reference_asset(args.forget_class)[0]}")
        print(f"    state: {reference_state}")
        print()
        print("  _best_dr.pt is rewritten every time D_r_test improves, so it")
        print("  exists and loads correctly from epoch 1 onwards. Existence does")
        print("  NOT mean training finished. Shipping a half-trained reference")
        print("  would leave the search aiming at the wrong target while every")
        print(f"  report called it 'the model that never saw a {forget_name}'.")
        print()
        print("  The CODE bundle is complete and can be uploaded now.")
        print("  Re-run this script once training finishes to stage the weights.")
        print("!" * 100)
        if args.require_reference:
            return 1
    else:
        print(f"\n  Bundle is COMPLETE -- W_ref included ({reference_state}).")

    if args.profile == "reference":
        print("\n  Next: upload the ONE bundle as a Kaggle Dataset, then follow")
        print("  kaggle/reference_training/README.md.")
    else:
        print("\n  Next: upload both bundles as Kaggle Datasets, then follow")
        print("  kaggle/README_KAGGLE.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
