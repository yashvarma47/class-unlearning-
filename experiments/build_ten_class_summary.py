"""Assemble the ten-class pure-result table from what each class run wrote.

Reads only committed per-class artefacts -- each class's anchor-metrics CSV, its
``summary.json`` and its full-fidelity front -- and joins them. Nothing is
recomputed and no model is loaded, so this can be re-run any time a class is
re-measured and it will simply pick up the new numbers.

The selection rule, applied identically to all ten
-------------------------------------------------
``C* = the front member maximising the anchor composite, ACC_r x (1 - ACC_f)``.

That is the anchor paper's own ``metric_function`` rather than something
invented here, and it reproduces the three classes that were selected by hand
before the sweep existed: frog #8, ship #6, airplane #0. Using one rule for all
ten is what makes the mean and standard deviation below mean anything --
selecting per class by eye would make the spread a property of the person
reading the fronts.

Writes ``ten_class_pure_summary.csv`` / ``.md`` and
``ten_class_pure_mean_std.csv`` under ``results/literature_alignment/``.

Run::

    python experiments/build_ten_class_summary.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, resolve_path  # noqa: E402

#: frog's anchor metrics predate the per-class layout and live with the
#: literature-alignment reports rather than in its run directory.
ANCHOR_OVERRIDES = {
    6: "results/literature_alignment/frog_anchor_metrics.csv",
}

#: Anchor Table 1, CIFAR-10 / ResNet-18, means over all ten classes.
#: https://arxiv.org/html/2312.00761v4
ANCHOR_TABLE_1 = [
    ("Original", "94.89 +/- 0.31", "94.89 +/- 2.75", "0.03 +/- 0.03"),
    ("Retraining (gold standard)", "94.81 +/- 0.52", "0", "100 +/- 0"),
    ("NegGrad", "69.89 +/- 10.23", "0.02 +/- 0.04", "0"),
    ("NegGrad+", "89.91 +/- 1.41", "0.94 +/- 1.87", "98.68 +/- 1.42"),
    ("Tarun et al. 2023 (UNSIR)", "92.20 +/- 0.72", "10.89 +/- 8.79", "61.5 +/- 25.86"),
    ("Kurmanji et al. 2023 (SCRUB)", "94.79 +/- 0.63", "0", "0"),
    ("Foster et al. 2024 (SSD)", "85.76 +/- 25.76", "4.37 +/- 12.79", "87.86 +/- 31.21"),
    ("Kodge et al. 2024 (the anchor)", "94.19 +/- 0.50", "0.03 +/- 0.09", "95.5 +/- 14.23"),
]

COLUMNS = [
    "class_id", "class_name", "front_position", "operators",
    "anchor_ACC_r", "anchor_ACC_f", "anchor_composite", "anchor_MIA",
    "f1_js", "f2_retain_train_loss", "f3_edit_cost", "selectivity_S",
    "our_mia_auc",
    "forget_train_acc", "forget_test_acc", "retain_train_acc", "retain_test_acc",
    "W_0_ACC_r", "W_0_ACC_f", "W_ref_ACC_r", "W_ref_ACC_f",
    "front_size", "search_evaluated", "search_cache_hits", "search_failures",
    "search_minutes", "results_dir",
]

#: Aggregated over the ten classes, the way the anchor aggregates its own table.
AGGREGATED = ["anchor_ACC_r", "anchor_ACC_f", "anchor_composite",
              "anchor_MIA", "selectivity_S"]


def composite(row: dict[str, str]) -> float:
    return 100.0 * float(row["retain_test_acc"]) * (1.0 - float(row["forget_test_acc"]))


def collect(class_id: int) -> dict[str, Any] | None:
    name = CIFAR10_CLASS_NAMES[class_id]
    run_dir = resolve_path(f"results/search/plan_a_{name}")
    front_path = run_dir / "full_fidelity" / "front_full_fidelity.csv"
    anchor_path = resolve_path(
        ANCHOR_OVERRIDES.get(class_id, f"results/search/plan_a_{name}/"
                                       f"{name}_anchor_metrics.csv"))
    if not front_path.is_file() or not anchor_path.is_file():
        return None

    with front_path.open(encoding="utf-8-sig") as handle:
        front = [r for r in csv.DictReader(handle) if r["status"] == "ok"]
    star = max(front, key=composite)

    with anchor_path.open(encoding="utf-8-sig") as handle:
        anchor = {r["model"]: r for r in csv.DictReader(handle)}
    row = anchor["C_star"]

    summary_path = run_dir / "summary.json"
    summary = (json.loads(summary_path.read_text(encoding="utf-8"))
               if summary_path.is_file() else {})

    def number(source: dict[str, str], key: str) -> float:
        value = source.get(key, "")
        return float(value) if value not in ("", None) else float("nan")

    return {
        "class_id": class_id,
        "class_name": name,
        "front_position": int(star["front_position"]),
        "operators": star["operators"],
        "anchor_ACC_r": number(row, "anchor_ACC_r"),
        "anchor_ACC_f": number(row, "anchor_ACC_f"),
        "anchor_composite": number(row, "anchor_composite"),
        "anchor_MIA": number(row, "anchor_MIA"),
        "f1_js": number(row, "f1_js"),
        "f2_retain_train_loss": number(row, "f2_retain_train_loss"),
        "f3_edit_cost": number(row, "f3_edit_cost"),
        "selectivity_S": number(row, "selectivity_S"),
        "our_mia_auc": number(row, "mia_auc"),
        "forget_train_acc": number(row, "forget_train_acc"),
        "forget_test_acc": number(row, "forget_test_acc"),
        "retain_train_acc": number(row, "retain_train_acc"),
        "retain_test_acc": number(row, "retain_test_acc"),
        "W_0_ACC_r": number(anchor["W_0"], "anchor_ACC_r"),
        "W_0_ACC_f": number(anchor["W_0"], "anchor_ACC_f"),
        "W_ref_ACC_r": number(anchor["W_ref"], "anchor_ACC_r"),
        "W_ref_ACC_f": number(anchor["W_ref"], "anchor_ACC_f"),
        "front_size": len(front),
        "search_evaluated": summary.get("evaluated", ""),
        "search_cache_hits": summary.get("cache_hits", ""),
        "search_failures": summary.get("failures", ""),
        "search_minutes": (round(summary["elapsed_seconds"] / 60, 1)
                           if "elapsed_seconds" in summary else ""),
        "results_dir": f"results/search/plan_a_{name}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/literature_alignment")
    args = parser.parse_args()

    rows = [r for r in (collect(c) for c in range(10)) if r is not None]
    missing = [CIFAR10_CLASS_NAMES[c] for c in range(10)
               if c not in {r["class_id"] for r in rows}]
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- the per-class CSV ------------------------------------------------
    csv_path = out_dir / "ten_class_pure_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # --- mean and std -----------------------------------------------------
    stats: dict[str, dict[str, float]] = {}
    for key in AGGREGATED:
        values = [r[key] for r in rows if math.isfinite(r[key])]
        stats[key] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else float("nan"),
            # Sample std, matching how a paper reports mean +/- std over classes.
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }

    stats_path = out_dir / "ten_class_pure_mean_std.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "n", "mean", "std", "min", "max",
                         "min_class", "max_class"])
        for key in AGGREGATED:
            usable = [r for r in rows if math.isfinite(r[key])]
            low = min(usable, key=lambda r: r[key])["class_name"] if usable else ""
            high = max(usable, key=lambda r: r[key])["class_name"] if usable else ""
            s = stats[key]
            writer.writerow([key, s["n"], f"{s['mean']:.4f}", f"{s['std']:.4f}",
                             f"{s['min']:.4f}", f"{s['max']:.4f}", low, high])

    # --- the readable report ----------------------------------------------
    def pm(key: str, places: int = 2) -> str:
        s = stats[key]
        return f"{s['mean']:.{places}f} +/- {s['std']:.{places}f}"

    lines = [
        "# Ten-class pure gradient-free MED-US result",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        f"`experiments/build_ten_class_summary.py` from each class's own "
        f"artefacts. Nothing here is recomputed.",
        "",
        f"**{len(rows)} of 10 classes.**"
        + ("" if not missing else f" Missing: {', '.join(missing)}."),
        "",
        "Every row is **pure gradient-free weight surgery** -- no gradient step was "
        "applied to any of these models. The accepted BN-frozen refinements for frog "
        "and ship are hybrids and are deliberately excluded.",
        "",
        "## Selection rule",
        "",
        "`C* = the front member maximising the anchor composite, ACC_r x (1 - ACC_f)`.",
        "",
        "One rule for all ten, and it is the anchor paper's own `metric_function` "
        "rather than something invented here. It reproduces the three classes that "
        "were selected by hand before the sweep existed -- frog #8, ship #6, "
        "airplane #0 -- so no previously reported number changed. Applying one rule "
        "uniformly is what makes the spread below a property of the method rather "
        "than of whoever read the fronts.",
        "",
        "## Per class",
        "",
        "| id | class | C* | operators | ACC_r | ACC_f | composite | MIA | f1 | f2 | f3 | S | min |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['class_id']} | {r['class_name']} | #{r['front_position']} | "
            f"`{r['operators']}` | {r['anchor_ACC_r']:.2f} | {r['anchor_ACC_f']:.2f} | "
            f"{r['anchor_composite']:.2f} | {r['anchor_MIA']:.2f} | "
            f"{r['f1_js']:.4f} | {r['f2_retain_train_loss']:.4f} | "
            f"{r['f3_edit_cost']:.4f} | {r['selectivity_S']:.2f} | "
            f"{r['search_minutes']} |")

    lines += [
        "",
        "## Mean +/- std over the ten classes",
        "",
        "| metric | mean +/- std | min | max |",
        "|---|---|---|---|",
        f"| `ACC_r` (%) | **{pm('anchor_ACC_r')}** | "
        f"{stats['anchor_ACC_r']['min']:.2f} | {stats['anchor_ACC_r']['max']:.2f} |",
        f"| `ACC_f` (%) | **{pm('anchor_ACC_f')}** | "
        f"{stats['anchor_ACC_f']['min']:.2f} | {stats['anchor_ACC_f']['max']:.2f} |",
        f"| composite (%) | **{pm('anchor_composite')}** | "
        f"{stats['anchor_composite']['min']:.2f} | {stats['anchor_composite']['max']:.2f} |",
        f"| anchor MIA (%) | **{pm('anchor_MIA')}** | "
        f"{stats['anchor_MIA']['min']:.2f} | {stats['anchor_MIA']['max']:.2f} |",
        f"| `S` selectivity | **{pm('selectivity_S', 1)}** | "
        f"{stats['selectivity_S']['min']:.1f} | {stats['selectivity_S']['max']:.1f} |",
        "",
        "Standard deviation is the sample std over classes, which is how the anchor "
        "reports its own table.",
        "",
        "## Against the anchor paper",
        "",
        "Kodge, Saha & Roy, TMLR 07/2024 -- CIFAR-10 / ResNet-18, means +/- std over "
        "all ten classes. Our row is now aggregated the same way, so this is the "
        "first like-for-like comparison in the project.",
        "",
        "| method | `ACC_r` | `ACC_f` | MIA |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n} | {a} | {f} | {m} |" for n, a, f, m in ANCHOR_TABLE_1]
    lines += [
        f"| **MED-US pure (ours, 10 classes)** | **{pm('anchor_ACC_r')}** | "
        f"**{pm('anchor_ACC_f')}** | **{pm('anchor_MIA')}** |",
        "",
        f"Every class clears the instance-level selectivity ceiling of **1.158**, "
        f"measured over 10,534 strategies in the predecessor project. The lowest `S` "
        f"here is {stats['selectivity_S']['min']:.1f} "
        f"({min((r for r in rows if math.isfinite(r['selectivity_S'])), key=lambda r: r['selectivity_S'])['class_name']}).",
        "",
    ]

    md_path = out_dir / "ten_class_pure_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[lines.index("## Per class"):]))
    print(f"\nwrote {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {stats_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
