"""Plot the Plan A Pareto front. Read-only: no model is loaded, nothing is re-run.

Sources, both produced by the completed class-unlearning Plan A run:

``full_fidelity/front_full_fidelity.csv``
    The 10 front members, re-measured on the complete sets. Objectives here are
    RAW -- the search itself saw min-max normalised values, which would plot as
    a meaningless unit square.
``final_objectives.json``
    W_0, W_ref and the post-search refined model, recomputed at full fidelity.

Two presentation decisions worth stating:

* **f2 is drawn on a log axis.** It spans 0.0015 to 4.51 across the front, a
  factor of 3000, because one member (position 3) destroyed the model. On a
  linear axis the other nine collapse onto the origin.
* **W_ref appears only where it is meaningful.** Its ``f3`` is not an edit cost:
  it is an independently trained model, not an edit of W_0, so it is omitted
  from the two edit-cost panels rather than drawn at a misleading position.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.utils.config import PROJECT_ROOT, resolve_path  # noqa: E402

#: Plain-English axis labels, as they should read in a dissertation figure.
LABEL_F1 = "f1  —  frog-forgetting match to $W_{ref}$\n(JS divergence on $D_f$; lower = closer to a model that never saw a frog)"
LABEL_F2 = "f2  —  retain loss $L_r$\n(cross-entropy on the 9 kept classes; lower = better preserved)"
LABEL_F3 = "f3  —  edit cost\n($\\|\\Delta\\theta\\|_2 / \\|\\theta_0\\|_2$; lower = less surgery)"

FRONT_COLOUR = "#8c9196"
HIGHLIGHTS = {
    "C*  (selected)":            dict(position=8, colour="#c1440e", marker="*", size=420),
    "best selectivity (S=4447)": dict(position=1, colour="#1b6ca8", marker="D", size=150),
    "strongest forgetting":      dict(position=3, colour="#7a1fa2", marker="X", size=200),
}


def load_front(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("obj1_js", "obj2_retain_loss", "obj3_edit_cost",
                    "forget_test_acc", "retain_test_acc", "selectivity_S",
                    "forget_train_acc", "retain_train_acc"):
            row[key] = float(row[key])
        row["front_position"] = int(row["front_position"])
    return sorted(rows, key=lambda r: r["front_position"])


def panel(ax, rows, xkey, ykey, xlabel, ylabel, baselines, refined, logx, logy):
    """One scatter subplot, with the front, the highlights and the baselines."""
    ax.scatter([r[xkey] for r in rows], [r[ykey] for r in rows],
               s=95, c=FRONT_COLOUR, edgecolor="white", linewidth=1.1,
               zorder=3, label="Pareto front member")

    for label, spec in HIGHLIGHTS.items():
        match = [r for r in rows if r["front_position"] == spec["position"]]
        if not match:
            continue
        row = match[0]
        ax.scatter(row[xkey], row[ykey], s=spec["size"], c=spec["colour"],
                   marker=spec["marker"], edgecolor="white", linewidth=1.2,
                   zorder=6, label=label)

    # Baselines. Any whose value is None for this pair is skipped -- see module
    # docstring on W_ref and edit cost.
    for label, spec in baselines.items():
        x, y = spec.get(xkey), spec.get(ykey)
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=190, facecolor="none", edgecolor=spec["colour"],
                   marker=spec["marker"], linewidth=2.1, zorder=5, label=label)

    if refined is not None:
        x, y = refined.get(xkey), refined.get(ykey)
        if x is not None and y is not None:
            ax.scatter(x, y, s=175, facecolor="none", edgecolor="#c1440e",
                       marker="*", linewidth=1.8, zorder=6,
                       label="C*_refined (post-search)")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front",
                        default="results/search/plan_a_frog/full_fidelity/"
                                "front_full_fidelity.csv")
    parser.add_argument("--objectives",
                        default="results/search/plan_a_frog/final_objectives.json")
    parser.add_argument("--out-png",
                        default="results/search/plan_a_frog/"
                                "pareto_front_plan_a_frog.png")
    parser.add_argument("--out-csv",
                        default="results/search/plan_a_frog/"
                                "pareto_front_plot_table.csv")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    front_path = resolve_path(args.front)
    rows = load_front(front_path)

    baselines: dict[str, dict[str, Any]] = {}
    refined = None
    objectives_path = resolve_path(args.objectives)
    if objectives_path.is_file():
        payload = json.loads(objectives_path.read_text(encoding="utf-8"))["rows"]
        w0, wref = payload["W_0"], payload["W_ref"]
        baselines = {
            "$W_0$  (original)": {
                "obj1_js": w0["f1_js"],
                "obj2_retain_loss": w0["f2_retain_train_loss"],
                "obj3_edit_cost": w0["f3_edit_cost"],
                "colour": "#111111", "marker": "s",
            },
            "$W_{ref}$  (retain-only target)": {
                "obj1_js": wref["f1_js"],
                "obj2_retain_loss": wref["f2_retain_train_loss"],
                # Deliberately absent: W_ref is not an edit of W_0.
                "obj3_edit_cost": None,
                "colour": "#0b7a4b", "marker": "^",
            },
        }
        if "C_star_refined_bn_frozen" in payload:
            r = payload["C_star_refined_bn_frozen"]
            refined = {
                "obj1_js": r["f1_js"],
                "obj2_retain_loss": r["f2_retain_train_loss"],
                "obj3_edit_cost": r["f3_edit_cost"],
            }

    figure, axes = plt.subplots(1, 3, figsize=(16.4, 5.4))
    figure.suptitle(
        "Pareto front — class unlearning of CIFAR-10 “frog” (Plan A, full fidelity)",
        fontsize=13.5, y=0.985)

    panel(axes[0], rows, "obj1_js", "obj2_retain_loss", LABEL_F1, LABEL_F2,
          baselines, refined, logx=False, logy=True)
    axes[0].set_title("(a)  forgetting vs retained-class loss", fontsize=10.5, pad=8)

    panel(axes[1], rows, "obj1_js", "obj3_edit_cost", LABEL_F1, LABEL_F3,
          baselines, refined, logx=False, logy=False)
    axes[1].set_title("(b)  forgetting vs edit cost", fontsize=10.5, pad=8)

    panel(axes[2], rows, "obj2_retain_loss", "obj3_edit_cost", LABEL_F2, LABEL_F3,
          baselines, refined, logx=True, logy=False)
    axes[2].set_title("(c)  retained-class loss vs edit cost", fontsize=10.5, pad=8)

    for ax in axes:
        ax.annotate("better", xy=(0.055, 0.055), xytext=(0.30, 0.30),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=8.5, color="#555555", ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2))

    # One legend for the whole figure, built from the first panel plus the
    # W_ref entry the first panel does carry.
    handles, labels = axes[0].get_legend_handles_labels()
    seen, ordered_h, ordered_l = set(), [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            ordered_h.append(handle)
            ordered_l.append(label)
    figure.legend(ordered_h, ordered_l, loc="lower center", ncol=len(ordered_l),
                  frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.015))

    distinct = len({(r["obj1_js"], r["obj2_retain_loss"], r["obj3_edit_cost"])
                    for r in rows})
    figure.text(
        0.5, 0.055,
        f"{len(rows)} front members ({distinct} distinct objective vectors; "
        f"positions 0≡6, 1≡2, 5≡7 coincide).  f2 on a log axis: it spans "
        f"0.0015–4.51 because position 3 destroyed the model.  "
        f"$W_{{ref}}$ is omitted from (b) and (c) — it is not an edit of $W_0$.",
        ha="center", fontsize=8.2, color="#555555")

    figure.tight_layout(rect=(0, 0.115, 1, 0.96))

    png_path = resolve_path(args.out_png)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)

    # --- the plotting table ------------------------------------------------
    csv_path = resolve_path(args.out_csv)
    fields = ["front_position", "operators", "strategy", "f1_js",
              "f2_retain_train_loss", "f3_edit_cost", "forget_test_acc",
              "retain_test_acc", "selectivity_S", "role"]
    role_by_position = {
        spec["position"]: label for label, spec in HIGHLIGHTS.items()
    }
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "front_position": row["front_position"],
                "operators": row["operators"],
                "strategy": row.get("canonical_action_key", ""),
                "f1_js": row["obj1_js"],
                "f2_retain_train_loss": row["obj2_retain_loss"],
                "f3_edit_cost": row["obj3_edit_cost"],
                "forget_test_acc": row["forget_test_acc"],
                "retain_test_acc": row["retain_test_acc"],
                "selectivity_S": row["selectivity_S"],
                "role": role_by_position.get(row["front_position"], ""),
            })

    print("=" * 100)
    print("PARETO FRONT FIGURE")
    print("=" * 100)
    print(f"  source front    {front_path.relative_to(PROJECT_ROOT)}")
    print(f"  source baselines{objectives_path.relative_to(PROJECT_ROOT)}"
          if objectives_path.is_file() else "  source baselines(none)")
    print(f"  points plotted  {len(rows)} front members "
          f"({distinct} distinct objective vectors)")
    print(f"  baselines       {len(baselines)} "
          f"(+1 post-search refined)" if refined else f"  baselines       {len(baselines)}")
    print(f"  PNG             {png_path.relative_to(PROJECT_ROOT)}")
    print(f"  CSV             {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
