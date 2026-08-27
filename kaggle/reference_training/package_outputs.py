"""Zip the trained references for download, and nothing else.

Runs on Kaggle after ``train_references.py``. The zip is what comes back across
the network and gets committed, so it is built from an **allowlist** rather than
by excluding things: a deny-list quietly ships whatever nobody thought of, and
the obvious accidents here are 341 MB of CIFAR-10 and a pile of ``_latest.pt``
checkpoints that are 85 MB each and mean nothing once training has finished.

Included, per class:

* ``class<ID>_<name>_reference_best_dr.pt``    the reference itself
* ``class<ID>_<name>_reference_best_dr.json``  its provenance sidecar
* ``class<ID>_<name>_training_log.csv``        per-epoch history
* ``class<ID>_<name>_training_summary.md``     the readable summary
* ``class<ID>_<name>_environment.json``        torch/CUDA/GPU snapshot
* ``cifar10_class<ID>_<name>.json``            the split, so it can be verified
* ``manifest.json``                            what was requested and produced

Excluded, always: CIFAR-10, caches, ``.venv``, ``__pycache__``, ``_latest.pt``
and ``_final.pt``. The ``_best_dr`` checkpoint is the only one anything reads.

Run::

    python kaggle/reference_training/package_outputs.py --tag trial_ship
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medus_class.utils.config import resolve_path  # noqa: E402

#: Only files matching one of these are ever added.
ALLOWED_PATTERNS = (
    "class*_reference_best_dr.pt",
    "class*_reference_best_dr.json",
    "class*_training_log.csv",
    "class*_training_summary.md",
    "class*_environment.json",
    "cifar10_class*.json",
    "manifest.json",
)

#: Belt and braces. Anything matching these is refused even if a future pattern
#: above would otherwise let it through.
FORBIDDEN_PATTERNS = (
    "*_latest.pt", "*_final.pt", "*.tar.gz", "*data_batch*", "*test_batch*",
    "*__pycache__*", "*.pyc", "*.venv*", "*.pytest_cache*",
)


def is_allowed(name: str) -> bool:
    if any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN_PATTERNS):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in ALLOWED_PATTERNS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="trial_ship",
                        help="names the zip: reference_outputs_<tag>.zip")
    parser.add_argument("--staging-dir", default="results/reference_training/staging")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    staging = resolve_path(args.staging_dir)
    if not staging.is_dir():
        raise SystemExit(
            f"staging directory not found: {staging}. Run train_references.py "
            f"first."
        )

    candidates = sorted(p for p in staging.iterdir() if p.is_file())
    included = [p for p in candidates if is_allowed(p.name)]
    skipped = [p for p in candidates if not is_allowed(p.name)]

    if not any(p.name.endswith("_reference_best_dr.pt") for p in included):
        raise SystemExit(
            f"no *_reference_best_dr.pt in {staging}; refusing to build an "
            f"empty package."
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve() if args.out_dir != "." \
            else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"reference_outputs_{args.tag}.zip"

    contents = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in included:
            archive.write(path, arcname=path.name)
            contents.append({"name": path.name, "bytes": path.stat().st_size,
                             "sha256": sha256(path)})
        archive.writestr(
            "package_manifest.json",
            json.dumps({"tag": args.tag, "files": contents}, indent=2),
        )

    print("=" * 100)
    print(f"PACKAGED  {zip_path}")
    print("=" * 100)
    for row in contents:
        print(f"  {row['bytes']:>12,}  {row['name']}")
    print(f"\n  {len(contents)} files, "
          f"{sum(r['bytes'] for r in contents):,} bytes uncompressed")
    print(f"  zip size {zip_path.stat().st_size:,} bytes")
    if skipped:
        print(f"\n  excluded {len(skipped)} file(s) not on the allowlist:")
        for path in skipped:
            print(f"    {path.name}")
    print(f"\n  Download this file from the Kaggle output panel:")
    print(f"    {zip_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
