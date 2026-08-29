"""The 10-class W_ref availability table, built from what is actually on disk.

Answers one question the sweep depends on: **for which classes do we have a
usable retain-only reference?** It is deliberately not built from the validation
CSV alone. A row saying PASS proves a zip was once validated; it does not prove
the checkpoint is still sitting in ``results/checkpoints/`` where every config
expects it. So both are checked, and a class counts as available only when the
file exists *and* a PASS row names it.

Writes ``results/reference_training/all_reference_models_summary.md`` and prints
the same table.

Run::

    python experiments/summarise_references.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, resolve_path  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary",
                        default="results/reference_training/"
                                "reference_validation_summary.csv")
    parser.add_argument("--out",
                        default="results/reference_training/"
                                "all_reference_models_summary.md")
    args = parser.parse_args()

    summary_path = resolve_path(args.summary)
    validated: dict[int, dict[str, Any]] = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                # Later rows supersede earlier ones: validation appends, so a
                # re-validation of the same class is the current verdict.
                validated[int(row["class_id"])] = row

    rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(CIFAR10_CLASS_NAMES):
        checkpoint = resolve_path(
            f"results/checkpoints/class{class_id}_{name}_reference_best_dr.pt")
        sidecar = checkpoint.with_suffix(".json")
        record = validated.get(class_id)

        on_disk = checkpoint.is_file()
        verdict = (record or {}).get("verdict", "")
        available = on_disk and verdict == "PASS"

        entry: dict[str, Any] = {
            "class_id": class_id,
            "name": name,
            "available": available,
            "on_disk": on_disk,
            "verdict": verdict or "never validated",
            "bytes": checkpoint.stat().st_size if on_disk else None,
            "sha256": sha256(checkpoint)[:12] if on_disk else None,
            "source_zip": (record or {}).get("source_zip", ""),
            "forget_test_acc": (record or {}).get("forget_test_acc", ""),
            "retain_test_acc": (record or {}).get("retain_test_acc", ""),
            "epoch": (record or {}).get("metadata_epoch", ""),
            "log_epochs": (record or {}).get("log_epochs", ""),
            "seed": (record or {}).get("metadata_seed", ""),
        }
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            entry["epoch"] = entry["epoch"] or metadata.get("epoch", "")
            entry["seed"] = entry["seed"] or metadata.get("seed", "")
        rows.append(entry)

    ready = [r for r in rows if r["available"]]

    def cell(value: Any, fmt: str = "{}") -> str:
        if value in (None, ""):
            return "--"
        try:
            return fmt.format(float(value))
        except (TypeError, ValueError):
            return fmt.format(value)

    lines = [
        "# W_ref availability -- all ten CIFAR-10 classes",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        f"`experiments/summarise_references.py`.",
        "",
        f"**{len(ready)} of 10 classes have a usable retain-only reference.**",
        "",
        "A class counts as available only when the checkpoint is present at "
        "`results/checkpoints/class<ID>_<name>_reference_best_dr.pt` **and** a "
        "`PASS` row names it in `reference_validation_summary.csv`. A PASS row "
        "alone proves a zip was validated once; it does not prove the file is "
        "still where every config expects it.",
        "",
        "| id | class | W_ref | `D_f_test` acc | `D_r_test` acc | epoch | log epochs | seed | sha256 | source zip |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        mark = "**yes**" if row["available"] else ("on disk, unvalidated"
                                                  if row["on_disk"] else "no")
        lines.append(
            f"| {row['class_id']} | {row['name']} | {mark} | "
            f"{cell(row['forget_test_acc'], '{:.4f}')} | "
            f"{cell(row['retain_test_acc'], '{:.4f}')} | "
            f"{cell(row['epoch'], '{:.0f}')} | "
            f"{cell(row['log_epochs'], '{:.0f}')} | "
            f"{cell(row['seed'], '{:.0f}')} | "
            f"`{row['sha256'] or '--'}` | `{row['source_zip'] or '--'}` |"
        )

    missing = [r for r in rows if not r["available"]]
    lines += [
        "",
        "## Storage",
        "",
        "Every one of these checkpoints is **external and git-ignored**. Ten "
        "references at ~85 MiB each is ~850 MiB, which alone would consume most "
        "of the free 1 GiB Git LFS allowance on top of the 256 MiB the frog "
        "chain already uses. The storage decision is still open -- see section 6 of "
        "`docs/artifact_manifest.md`. What *is* tracked is enough to identify "
        "the right file: the sha256 above, the split, the validation verdict and "
        "the per-class training summary.",
        "",
    ]
    if missing:
        lines += ["## Still missing", ""]
        lines += [f"* class {r['class_id']} ({r['name']}) -- {r['verdict']}"
                  for r in missing]
        lines += [""]
    else:
        lines += ["## Complete", "",
                  "All ten classes have a validated reference. The 10-class "
                  "sweep is no longer blocked on reference training.", ""]

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[10:]))
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
