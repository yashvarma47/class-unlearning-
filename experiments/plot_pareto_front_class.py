"""Plot a class Pareto front. Read-only: no model is loaded, nothing is re-run.

The class-agnostic sibling of ``plot_pareto_front.py``. That one is frog's: its
axis labels say "frog", its highlighted positions are hard-coded to that run's
result, and it is left exactly as it is so the committed frog figure stays
reproducible from the script that made it. This one takes the class as an
argument and derives its highlights from the data.

Sources, both produced by a completed Plan A run:

``full_fidelity/front_full_fidelity.csv``
    The front members, re-measured on the complete sets. Objectives here are
    RAW -- the search saw min-max normalised values, which would plot as a
    meaningless unit square.
``full_fidelity/baselines.json``
    ``W_0`` and ``W_ref`` on the same sets.

Two presentation decisions carried over from the frog figure:

* **f2 is drawn on a log axis** when its range spans more than two orders of
  magnitude, which happens whenever one front member destroyed the model. On a
  linear axis the survivors collapse onto the origin.
* **W_ref is omitted from the edit-cost panels.** Its ``f3`` is not an edit
  cost -- it is an independently trained model, not an edit of ``W_0`` -- so
  drawing it there would put a meaningless point on the axis.

Run::

    python experiments/plot_pareto_front_class.py --class-id 8 \\
        --run-dir results/search/plan_a_ship --selected 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, resolve_path  # noqa: E402

FRONT_COLOUR = "#8c9196"
SELECTED_COLOUR = "#c1440e"
BEST_S_COLOUR = "#1b6ca8"
STRONGEST_COLOUR = "#7a1fa2"

NUMERIC_COLUMNS = (
    "obj1_js", "obj2_retain_loss", "obj3_edit_cost",
    "forget_train_acc", "forget_train_loss",
    "retain_train_acc", "retain_train_loss",
    "forget_test_acc", "forget_test_loss",
    "retain_test_acc", "retain_test_loss",
    "selectivity_S", "mia_auc",
)


def load_front(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in NUMERIC_COLUMNS:
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
        row["front_position"] = int(row["front_position"])
    return rows


def panel(ax, rows, xkey, ykey, xlabel, ylabel, baselines, highlights,
          show_reference: bool) -> None:
    ax.scatter([r[xkey] for r in rows], [r[ykey] for r in rows],
               s=70, c=FRONT_COLOUR, edgecolors="white", linewidths=0.8,
               zorder=3, label="_front")

    for position, style in highlights.items():
        member = next((r for r in rows if r["front_position"] == position), None)
        if member is None:
            continue
        ax.scatter([member[xkey]], [member[ykey]], s=style["size"],
                   c=style["colour"], marker=style["marker"],
                   edgecolors="white", linewidths=1.2, zorder=5)

    # W_0 is a real point wherever its coordinates are defined: f3 = 0 by
    # definition (no edit), f2 its retain loss. Its f1 is omitted -- it is a
    # genuine divergence from W_ref, but plotting it would stretch the f1 axis
    # to cover a point that is not a candidate.
    w0 = baselines.get("original_point")
    if w0 and xkey in w0 and ykey in w0:
        ax.scatter([w0[xkey]], [w0[ykey]], s=160, marker="s", c="#2b2b2b",
                   edgecolors="white", linewidths=1.0, zorder=4)

    # W_ref only where f3 is not involved.
    wref = baselines.get("reference_point")
    if show_reference and wref and xkey in wref and ykey in wref:
        ax.scatter([wref[xkey]], [wref[ykey]], s=200, marker="P", c="#1a7f37",
                   edgecolors="white", linewidths=1.0, zorder=4)

    values = [r[ykey] for r in rows if r[ykey] > 0]
    if ykey == "obj2_retain_loss" and values and max(values) / min(values) > 100:
        ax.set_yscale("log")
    values = [r[xkey] for r in rows if r[xkey] > 0]
    if xkey == "obj2_retain_loss" and values and max(values) / min(values) > 100:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selected", type=int, required=True,
                        help="front_position of the selected C*")
    parser.add_argument("--out-png", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    class_id = args.class_id
    class_name = CIFAR10_CLASS_NAMES[class_id]
    run_dir = resolve_path(args.run_dir)

    front = load_front(run_dir / "full_fidelity" / "front_full_fidelity.csv")
    baselines_raw = json.loads(
        (run_dir / "full_fidelity" / "baselines.json").read_text(encoding="utf-8"))

    # W_0 is the identity edit: f1 is its real divergence from W_ref, f2 its
    # retain loss, f3 exactly zero.
    baselines = {
        "original_point": {
            "obj2_retain_loss": baselines_raw["original"]["retain_train_loss"],
            "obj3_edit_cost": 0.0,
        },
        "reference_point": {
            "obj1_js": 0.0,
            "obj2_retain_loss": baselines_raw["reference"]["retain_train_loss"],
        },
    }

    # Identity edits have S = nan (a ratio of deltas against W_0, so 0/0), and
    # every comparison against nan is False -- a plain max() would return the
    # first nan row and label the UNEDITED model as most selective.
    finite_s = [r for r in front if math.isfinite(r["selectivity_S"])]
    best_s = max(finite_s or front, key=lambda r: r["selectivity_S"])
    strongest = min(front, key=lambda r: r["forget_test_acc"])

    highlights: dict[int, dict[str, Any]] = {
        args.selected: dict(colour=SELECTED_COLOUR, marker="*", size=420),
    }
    if best_s["front_position"] != args.selected:
        highlights[best_s["front_position"]] = dict(
            colour=BEST_S_COLOUR, marker="D", size=150)
    if strongest["front_position"] not in highlights:
        highlights[strongest["front_position"]] = dict(
            colour=STRONGEST_COLOUR, marker="X", size=200)

    label_f1 = (f"f1  —  {class_name}-forgetting match to $W_{{ref}}$\n"
                f"(JS divergence on $D_f$; lower = closer to a model that never "
                f"saw a {class_name})")
    label_f2 = ("f2  —  retain loss $L_r$\n"
                "(cross-entropy on the 9 kept classes; lower = better preserved)")
    label_f3 = ("f3  —  edit cost\n"
                "($\\|\\Delta\\theta\\|_2 / \\|\\theta_0\\|_2$; lower = less surgery)")

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    panel(axes[0], front, "obj1_js", "obj2_retain_loss", label_f1, label_f2,
          baselines, highlights, show_reference=True)
    panel(axes[1], front, "obj1_js", "obj3_edit_cost", label_f1, label_f3,
          baselines, highlights, show_reference=False)
    panel(axes[2], front, "obj2_retain_loss", "obj3_edit_cost", label_f2, label_f3,
          baselines, highlights, show_reference=False)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8,
               markerfacecolor=FRONT_COLOUR, markeredgecolor="white",
               label=f"Pareto front ({len(front)} members)"),
        Line2D([], [], marker="*", linestyle="", markersize=18,
               markerfacecolor=SELECTED_COLOUR, markeredgecolor="white",
               label=f"C*  selected (front #{args.selected})"),
    ]
    if best_s["front_position"] != args.selected:
        handles.append(Line2D(
            [], [], marker="D", linestyle="", markersize=9,
            markerfacecolor=BEST_S_COLOUR, markeredgecolor="white",
            label=f"best selectivity (#{best_s['front_position']}, "
                  f"S={best_s['selectivity_S']:.0f})"))
    if strongest["front_position"] not in (args.selected, best_s["front_position"]):
        handles.append(Line2D(
            [], [], marker="X", linestyle="", markersize=10,
            markerfacecolor=STRONGEST_COLOUR, markeredgecolor="white",
            label=f"strongest forgetting (#{strongest['front_position']}, "
                  f"$D_f$ test {strongest['forget_test_acc']:.3f})"))
    handles += [
        Line2D([], [], marker="s", linestyle="", markersize=9,
               markerfacecolor="#2b2b2b", markeredgecolor="white", label="$W_0$"),
        Line2D([], [], marker="P", linestyle="", markersize=10,
               markerfacecolor="#1a7f37", markeredgecolor="white",
               label="$W_{ref}$  (f1/f2 panels only)"),
    ]

    figure.suptitle(
        f"MED-US Plan A — class {class_id} ({class_name}) Pareto front, "
        f"full fidelity",
        fontsize=13, y=0.99,
    )
    figure.legend(handles=handles, loc="lower center", ncol=len(handles),
                  frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))

    out_png = resolve_path(
        args.out_png or run_dir / f"pareto_front_plan_a_{class_name}.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    out_csv = resolve_path(args.out_csv or run_dir / "pareto_front_plot_data.csv")
    columns = ["front_position", "obj1_js", "obj2_retain_loss", "obj3_edit_cost",
               "forget_train_acc", "forget_train_loss",
               "forget_test_acc", "forget_test_loss",
               "retain_train_acc", "retain_train_loss",
               "retain_test_acc", "retain_test_loss",
               "selectivity_S", "mia_auc", "operators", "role"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(front, key=lambda r: r["front_position"]):
            roles = []
            if row["front_position"] == args.selected:
                roles.append("selected C*")
            if row["front_position"] == best_s["front_position"]:
                roles.append("best selectivity")
            if row["front_position"] == strongest["front_position"]:
                roles.append("strongest forgetting")
            writer.writerow({**row, "role": "; ".join(roles)})

    print(f"wrote {out_png.relative_to(PROJECT_ROOT)}")
    print(f"wrote {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"  front members       {len(front)}")
    print(f"  selected C*         #{args.selected}")
    print(f"  best selectivity    #{best_s['front_position']}  "
          f"S={best_s['selectivity_S']:.2f}")
    print(f"  strongest forgetting #{strongest['front_position']}  "
          f"D_f test {strongest['forget_test_acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
