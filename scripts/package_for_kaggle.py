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

Run::

    python scripts/package_for_kaggle.py
    python scripts/package_for_kaggle.py --out-dir dist --require-reference
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

#: Checkpoints and metadata the run cannot start without.
REQUIRED_ASSETS = (
    ("results/checkpoints/cifar10_resnet18_seed42_best.pt",
     "W_0 -- the original model, trained on all 50 000 images. "
     "The model being unlearned FROM."),
    ("results/splits/cifar10_class6_frog.json",
     "The class split: D_f/D_r over train and test. Regenerable, but shipping "
     "it means Kaggle cannot silently build a different one."),
)

#: Produced by train_class_reference.py; absent until that finishes.
REFERENCE_ASSET = (
    "results/checkpoints/class6_frog_reference_best_dr.pt",
    "W_ref -- trained on D_r only, selected on D_r_test accuracy then loss. "
    "MUST be the _best_dr file, NOT _best.",
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
    weights_dir = out_dir / "medus_class_weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir)
    weights_dir.mkdir(parents=True)

    assets: list[tuple[str, str]] = list(REQUIRED_ASSETS)
    reference_path = PROJECT_ROOT / REFERENCE_ASSET[0]

    reference_ready = False
    reference_state = "not produced yet"
    if reference_path.is_file():
        reference_ready, reference_state = reference_training_state(reference_path)
        if reference_ready:
            assets.append(REFERENCE_ASSET)

    manifest_entries: list[dict[str, Any]] = []
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
        "forget_class": 6,
        "forget_class_name": "frog",
        "code_bundle": {
            "file": zip_path.name,
            "files": len(code_files),
            "bytes": zip_path.stat().st_size,
        },
        "weights_bundle": manifest_entries,
        "reference_included": reference_ready,
        "reference_state": reference_state,
        "upload_as": {
            "medus-class-code": "Kaggle Dataset from medus_class_code.zip",
            "medus-class-weights": "Kaggle Dataset from medus_class_weights/",
        },
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

    if not reference_ready:
        print("\n" + "!" * 100)
        print("  W_ref IS NOT INCLUDED -- it is not from a finished run.")
        print(f"    {REFERENCE_ASSET[0]}")
        print(f"    state: {reference_state}")
        print()
        print("  _best_dr.pt is rewritten every time D_r_test improves, so it")
        print("  exists and loads correctly from epoch 1 onwards. Existence does")
        print("  NOT mean training finished. Shipping a half-trained reference")
        print("  would leave the search aiming at the wrong target while every")
        print("  report called it 'the model that never saw a frog'.")
        print()
        print("  The CODE bundle is complete and can be uploaded now.")
        print("  Re-run this script once training finishes to stage the weights.")
        print("!" * 100)
        if args.require_reference:
            return 1
    else:
        print(f"\n  Bundle is COMPLETE -- W_ref included ({reference_state}).")

    print(f"\n  Next: upload both bundles as Kaggle Datasets, then follow")
    print(f"  kaggle/README_KAGGLE.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
