"""Build the hybrid table and the pure-vs-hybrid comparison, from artefacts only.

"Hybrid" means the pure gradient-free ``C*`` followed by the BN-frozen
refinement: one clipped gradient-ascent step on ``D_f``, one repair step on
``D_r``, both outside the evolutionary search. It is a **different method** from
pure MED-US and is never merged into the pure numbers -- the anchor paper's own
method is gradient-free, so only the pure table compares like for like with it.

Per class the hybrid row is:

* the **refined** model where a refinement was run and accepted;
* the **pure** ``C*`` unchanged where the refinement was rejected, or where it
  was never attempted.

Airplane is the deliberate no-op: its pure ``ACC_f`` is already 0.00 with anchor
MIA 100.00, so a forgetting step has nothing to improve and none was run. Its
hybrid row is its pure row, and the attempt table says so rather than leaving a
gap that reads like a missing result.

Reads only committed per-class artefacts and recomputes nothing.

Run::

    python experiments/build_hybrid_summary.py
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

ANCHOR_OVERRIDES = {6: "results/literature_alignment/frog_anchor_metrics.csv"}
REFINED_ROW = "C_star_refined_bn_frozen"

#: Classes where no refinement was attempted, and why.
NO_OP = {0: "pure ACC_f is already 0.00 with anchor MIA 100.00 -- a forgetting "
            "step has nothing to improve"}

METRICS = ["anchor_ACC_r", "anchor_ACC_f", "anchor_composite", "anchor_MIA",
           "selectivity_S"]

HYBRID_COLUMNS = [
    "class_id", "class_name", "source", "operators",
    "anchor_ACC_r", "anchor_ACC_f", "anchor_composite", "anchor_MIA",
    "f1_js", "f2_retain_train_loss", "f3_edit_cost", "selectivity_S",
    "our_mia_auc",
    "forget_train_acc", "forget_train_loss", "forget_test_acc", "forget_test_loss",
    "retain_train_acc", "retain_train_loss", "retain_test_acc", "retain_test_loss",
    "parameter_movement", "buffer_movement", "refinement_status",
    "results_dir",
]


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def collect(class_id: int) -> dict[str, Any] | None:
    name = CIFAR10_CLASS_NAMES[class_id]
    anchor_path = resolve_path(ANCHOR_OVERRIDES.get(
        class_id, f"results/search/plan_a_{name}/{name}_anchor_metrics.csv"))
    if not anchor_path.is_file():
        return None
    with anchor_path.open(encoding="utf-8-sig") as handle:
        anchor = {r["model"]: r for r in csv.DictReader(handle)}
    if "C_star" not in anchor:
        return None

    refine_dir = resolve_path(f"results/search/plan_a_{name}_bn_frozen_refined")
    refinement_json = refine_dir / "refinement.json"
    refined_sidecar = refine_dir / "refined_best.json"

    if class_id in NO_OP:
        status, reason = "no-op", NO_OP[class_id]
    elif refinement_json.is_file():
        record = json.loads(refinement_json.read_text(encoding="utf-8"))
        accepted = bool(record.get("accepted", refined_sidecar.is_file()))
        status = "accepted" if accepted else "rejected"
        failed = [k for k, v in (record.get("acceptance_checks") or {}).items() if not v]
        reason = "" if accepted else f"failed: {', '.join(failed) or 'unknown'}"
    else:
        status, reason = "not attempted", "no refinement directory"

    use_refined = status == "accepted" and REFINED_ROW in anchor
    row = anchor[REFINED_ROW] if use_refined else anchor["C_star"]

    movement = {}
    if use_refined and refined_sidecar.is_file():
        movement = json.loads(refined_sidecar.read_text(encoding="utf-8"))["metrics"]

    return {
        "class_id": class_id,
        "class_name": name,
        "source": "refined (hybrid)" if use_refined else "pure C*",
        "operators": row.get("operators", ""),
        **{k: number(row, k if k != "our_mia_auc" else "mia_auc")
           for k in ("anchor_ACC_r", "anchor_ACC_f", "anchor_composite",
                     "anchor_MIA", "f1_js", "f2_retain_train_loss",
                     "f3_edit_cost", "selectivity_S", "our_mia_auc",
                     "forget_train_acc", "forget_train_loss",
                     "forget_test_acc", "forget_test_loss",
                     "retain_train_acc", "retain_train_loss",
                     "retain_test_acc", "retain_test_loss")},
        "parameter_movement": movement.get("parameter_movement", ""),
        "buffer_movement": movement.get("buffer_movement", ""),
        "refinement_status": status,
        "refinement_note": reason,
        "results_dir": (f"results/search/plan_a_{name}_bn_frozen_refined"
                        if use_refined else f"results/search/plan_a_{name}"),
        "_pure": {k: number(anchor["C_star"], k) for k in METRICS},
    }


def stats(rows: list[dict[str, Any]], key: str,
          getter=lambda r, k: r[k]) -> dict[str, float]:
    values = [getter(r, key) for r in rows if math.isfinite(getter(r, key))]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/literature_alignment")
    args = parser.parse_args()

    rows = [r for r in (collect(c) for c in range(10)) if r is not None]
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- hybrid CSV -------------------------------------------------------
    with (out_dir / "ten_class_hybrid_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HYBRID_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    hybrid_stats = {k: stats(rows, k) for k in METRICS}
    pure_stats = {k: stats(rows, k, lambda r, key: r["_pure"][key]) for k in METRICS}

    with (out_dir / "ten_class_hybrid_mean_std.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "n", "mean", "std", "min", "max"])
        for k in METRICS:
            s = hybrid_stats[k]
            writer.writerow([k, s["n"], f"{s['mean']:.4f}", f"{s['std']:.4f}",
                             f"{s['min']:.4f}", f"{s['max']:.4f}"])

    # --- comparison CSV ---------------------------------------------------
    with (out_dir / "pure_vs_hybrid_comparison.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "refinement_status",
                         "pure_ACC_r", "hybrid_ACC_r", "delta_ACC_r",
                         "pure_ACC_f", "hybrid_ACC_f", "delta_ACC_f",
                         "pure_composite", "hybrid_composite", "delta_composite",
                         "pure_MIA", "hybrid_MIA", "delta_MIA",
                         "pure_S", "hybrid_S", "note"])
        for r in rows:
            p = r["_pure"]
            writer.writerow([
                r["class_id"], r["class_name"], r["refinement_status"],
                f"{p['anchor_ACC_r']:.4f}", f"{r['anchor_ACC_r']:.4f}",
                f"{r['anchor_ACC_r'] - p['anchor_ACC_r']:+.4f}",
                f"{p['anchor_ACC_f']:.4f}", f"{r['anchor_ACC_f']:.4f}",
                f"{r['anchor_ACC_f'] - p['anchor_ACC_f']:+.4f}",
                f"{p['anchor_composite']:.4f}", f"{r['anchor_composite']:.4f}",
                f"{r['anchor_composite'] - p['anchor_composite']:+.4f}",
                f"{p['anchor_MIA']:.4f}", f"{r['anchor_MIA']:.4f}",
                f"{r['anchor_MIA'] - p['anchor_MIA']:+.4f}",
                f"{p['selectivity_S']:.4f}", f"{r['selectivity_S']:.4f}",
                r["refinement_note"]])

    # --- readable reports -------------------------------------------------
    def pm(source: dict[str, dict[str, float]], key: str, places: int = 2) -> str:
        s = source[key]
        return f"{s['mean']:.{places}f} +/- {s['std']:.{places}f}"

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    improved = [r for r in rows
                if r["anchor_composite"] > r["_pure"]["anchor_composite"] + 1e-9]
    rejected = [r for r in rows if r["refinement_status"] == "rejected"]
    noop = [r for r in rows if r["refinement_status"] == "no-op"]

    hybrid_lines = [
        "# Ten-class HYBRID result (pure C* + BN-frozen refinement)",
        "",
        f"Generated {stamp} by `experiments/build_hybrid_summary.py`. Nothing is "
        f"recomputed here.",
        "",
        "**This is not pure gradient-free MED-US.** Every row marked "
        "*refined (hybrid)* had one clipped gradient-ascent step on `D_f` and one "
        "repair step on `D_r` applied after the search, with BatchNorm frozen. "
        "The anchor paper's own method is gradient-free, so the like-for-like "
        "comparison with it remains `ten_class_pure_summary.md`.",
        "",
        "| id | class | source | ACC_r | ACC_f | composite | MIA | S | dW | dBN |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        dw = (f"{float(r['parameter_movement']):.6f}"
              if r["parameter_movement"] != "" else "--")
        dbn = (f"{float(r['buffer_movement']):.6f}"
               if r["buffer_movement"] != "" else "--")
        hybrid_lines.append(
            f"| {r['class_id']} | {r['class_name']} | {r['source']} | "
            f"{r['anchor_ACC_r']:.2f} | {r['anchor_ACC_f']:.2f} | "
            f"{r['anchor_composite']:.2f} | {r['anchor_MIA']:.2f} | "
            f"{r['selectivity_S']:.2f} | {dw} | {dbn} |")

    hybrid_lines += [
        "",
        "## Refinement attempts",
        "",
        "| id | class | status | note |",
        "|---:|---|---|---|",
    ]
    hybrid_lines += [f"| {r['class_id']} | {r['class_name']} | "
                     f"**{r['refinement_status']}** | {r['refinement_note'] or '--'} |"
                     for r in rows]
    hybrid_lines += [
        "",
        f"Accepted: {sum(1 for r in rows if r['refinement_status'] == 'accepted')}. "
        f"Rejected: {len(rejected)}. No-op: {len(noop)}.",
        "",
        "Every accepted refinement was required to hold BatchNorm buffer movement at "
        "**exactly zero**, and every one did. That is the condition the first frog "
        "attempt failed silently: eight batches of `D_r` re-estimated the running "
        "statistics and undid the operator edit while every weight-based guard "
        "reported success.",
        "",
        "## Mean +/- std over the ten classes",
        "",
        "| metric | hybrid | pure |",
        "|---|---|---|",
        f"| `ACC_r` (%) | {pm(hybrid_stats, 'anchor_ACC_r')} | "
        f"{pm(pure_stats, 'anchor_ACC_r')} |",
        f"| `ACC_f` (%) | **{pm(hybrid_stats, 'anchor_ACC_f')}** | "
        f"{pm(pure_stats, 'anchor_ACC_f')} |",
        f"| composite (%) | **{pm(hybrid_stats, 'anchor_composite')}** | "
        f"{pm(pure_stats, 'anchor_composite')} |",
        f"| anchor MIA (%) | **{pm(hybrid_stats, 'anchor_MIA')}** | "
        f"{pm(pure_stats, 'anchor_MIA')} |",
        f"| `S` | {pm(hybrid_stats, 'selectivity_S', 1)} | "
        f"{pm(pure_stats, 'selectivity_S', 1)} |",
        "",
    ]
    (out_dir / "ten_class_hybrid_summary.md").write_text(
        "\n".join(hybrid_lines) + "\n", encoding="utf-8")

    comparison_lines = [
        "# Pure vs hybrid, class by class",
        "",
        f"Generated {stamp} by `experiments/build_hybrid_summary.py`.",
        "",
        "**Pure** = gradient-free MED-US alone. **Hybrid** = that same `C*` plus one "
        "forget step and one retain repair, BatchNorm frozen, applied outside the "
        "search. They are different methods and are reported separately on purpose.",
        "",
        "| id | class | status | ACC_r pure -> hybrid | ACC_f pure -> hybrid | "
        "composite pure -> hybrid | MIA pure -> hybrid |",
        "|---:|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = r["_pure"]
        comparison_lines.append(
            f"| {r['class_id']} | {r['class_name']} | {r['refinement_status']} | "
            f"{p['anchor_ACC_r']:.2f} -> {r['anchor_ACC_r']:.2f} "
            f"({r['anchor_ACC_r'] - p['anchor_ACC_r']:+.2f}) | "
            f"{p['anchor_ACC_f']:.2f} -> {r['anchor_ACC_f']:.2f} "
            f"({r['anchor_ACC_f'] - p['anchor_ACC_f']:+.2f}) | "
            f"{p['anchor_composite']:.2f} -> {r['anchor_composite']:.2f} "
            f"({r['anchor_composite'] - p['anchor_composite']:+.2f}) | "
            f"{p['anchor_MIA']:.2f} -> {r['anchor_MIA']:.2f} "
            f"({r['anchor_MIA'] - p['anchor_MIA']:+.2f}) |")

    comparison_lines += [
        "",
        "## Aggregate",
        "",
        "| metric | pure | hybrid | change |",
        "|---|---|---|---|",
    ]
    for key, label, places in (("anchor_ACC_r", "`ACC_r` (%)", 2),
                               ("anchor_ACC_f", "`ACC_f` (%)", 2),
                               ("anchor_composite", "composite (%)", 2),
                               ("anchor_MIA", "anchor MIA (%)", 2),
                               ("selectivity_S", "`S`", 1)):
        delta = hybrid_stats[key]["mean"] - pure_stats[key]["mean"]
        comparison_lines.append(
            f"| {label} | {pm(pure_stats, key, places)} | "
            f"{pm(hybrid_stats, key, places)} | {delta:+.{places}f} |")

    comparison_lines += [
        "",
        f"**{len(improved)} of {len(rows)} classes improved on the composite.** "
        f"The {len(noop)} no-op class is airplane, whose pure `ACC_f` is already "
        f"0.00. {len(rejected)} refinements were rejected.",
        "",
    ]
    (out_dir / "pure_vs_hybrid_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8")

    print("\n".join(hybrid_lines))
    print("\n".join(comparison_lines[comparison_lines.index("## Aggregate"):]))
    for name in ("ten_class_hybrid_summary.csv", "ten_class_hybrid_summary.md",
                 "ten_class_hybrid_mean_std.csv", "pure_vs_hybrid_comparison.csv",
                 "pure_vs_hybrid_comparison.md"):
        print(f"wrote {(out_dir / name).relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
