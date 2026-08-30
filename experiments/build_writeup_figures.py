"""Render the thesis-ready figures from artefacts that already exist.

Reads the CSVs in ``results/writeup_package/`` (built by
``build_writeup_package.py``) plus ``truck_prediction_distribution.csv`` (built
by ``analyse_truck_predictions.py``), and writes PNGs into
``results/writeup_package/figures/``. No model is loaded here, nothing is
measured, and no committed result is touched.

Design decisions, so they are deliberate rather than defaults:

**Palette.** The three series colours are the ones already used by
``plot_pareto_front_class.py`` for the ten committed Pareto figures, so this set
and those read as one system in the same document. Validated for categorical use
on a light surface: lightness band, chroma floor, CVD separation (worst adjacent
pair dE 20.9 protan / 19.1 tritan), normal-vision separation (27.4) and >= 3:1
contrast all pass. Grey is a neutral for connectors and reference lines, never a
series.

**Light surface only.** These are print figures for a dissertation page. The
purple fails the lightness band against a dark surface, and a dark variant is not
produced rather than shipped unchecked.

**Forms.** Two values per class over ten classes is a dumbbell, not paired bars:
dots carry no area, so a zoomed axis is honest, which matters for `ACC_r` where
the whole range is a quarter of a point. Figures 1-3 share one class order so
they can be read across. Mixed-scale metrics are small multiples, never one
grouped chart with two y-scales.

**Every mark is directly labelled**, so identity and value never depend on colour
alone.

Run::

    python experiments/build_writeup_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG = PROJECT_ROOT / "results" / "writeup_package"
FIG = PKG / "figures"

# Validated categorical set (light surface). Identical to the colours used by the
# ten committed Pareto figures.
PURE = "#1b6ca8"
HYBRID = "#c1440e"
THIRD = "#7a1fa2"
NEUTRAL = "#8c9196"
INK = "#22262b"
MUTED = "#5b6472"
GRID = "#dcdfe3"
SURFACE = "#ffffff"

DPI = 300

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})


def read(name: str) -> list[dict[str, str]]:
    with (PKG / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def bare(ax, *, grid_axis: str = "x") -> None:
    """Recessive frame: no box, one faint grid direction, marks in front."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(figure, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


# ------------------------------------------------------------------ dumbbells

def dumbbell(name: str, title: str, subtitle: str, xlabel: str,
             order: list[str], values: dict[str, tuple[float, float]],
             better: str, note: str, legend_loc: str = "lower right",
             pad: float = 0.06) -> Path:
    """One row per class, pure dot to hybrid dot, connected."""
    figure, ax = plt.subplots(figsize=(7.2, 4.6))
    ys = list(range(len(order)))[::-1]

    lo = min(min(v) for v in values.values())
    hi = max(max(v) for v in values.values())
    span = max(hi - lo, 1e-6)
    left, right = lo - span * pad * 3, hi + span * pad * 5

    for y, cls in zip(ys, order):
        p, h = values[cls]
        ax.plot([p, h], [y, y], color=NEUTRAL, linewidth=1.6,
                solid_capstyle="round", zorder=2, alpha=0.55)
        ax.scatter([p], [y], s=64, color=PURE, zorder=3,
                   edgecolors=SURFACE, linewidths=1.4)
        ax.scatter([h], [y], s=64, color=HYBRID, zorder=4,
                   edgecolors=SURFACE, linewidths=1.4)

        # Direct labels, placed outward so the pair never collides.
        if abs(h - p) < span * 0.012:
            ax.annotate(f"{p:.2f}", (max(p, h), y), xytext=(9, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.5, color=MUTED)
        else:
            first, second = (p, h) if p < h else (h, p)
            fc, sc = (PURE, HYBRID) if p < h else (HYBRID, PURE)
            ax.annotate(f"{first:.2f}", (first, y), xytext=(-8, 0),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=7.5, color=fc)
            ax.annotate(f"{second:.2f}", (second, y), xytext=(8, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=7.5, color=sc)

    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=9, color=INK)
    ax.set_xlim(left, right)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel(f"{xlabel}   ({better})", labelpad=8)
    bare(ax)

    ax.set_title(title, loc="left", pad=22)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 12), textcoords="offset points",
                fontsize=8.5, color=MUTED, va="bottom")

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=7,
               markerfacecolor=PURE, markeredgecolor=SURFACE, label="pure MED-US"),
        Line2D([], [], marker="o", linestyle="", markersize=7,
               markerfacecolor=HYBRID, markeredgecolor=SURFACE,
               label="hybrid (BN-frozen refinement)"),
    ], loc=legend_loc, frameon=False, fontsize=8.5, handletextpad=0.4)

    figure.text(0.0, -0.045, note, fontsize=7.5, color=MUTED, ha="left")
    return save(figure, name)


def build_dumbbells(pure, hybrid) -> list[Path]:
    p = {r["class_name"]: r for r in pure}
    h = {r["class_name"]: r for r in hybrid}
    # One order for all three, so the set can be read across: worst forgetting first.
    order = sorted(p, key=lambda c: -float(p[c]["ACC_f"]))

    out = [
        dumbbell(
            "pure_vs_hybrid_acc_f_by_class.png",
            "Forget-class accuracy falls for every refined class",
            "ACC_f on the 1,000 held-out images of the forgotten class, by class",
            "ACC_f  (%)", order,
            {c: (float(p[c]["ACC_f"]), float(h[c]["ACC_f"])) for c in order},
            "lower is better",
            "Classes ordered by pure ACC_f, worst first; the same order is used in all three "
            "pure-vs-hybrid figures. Airplane is a deliberate no-op -- its pure ACC_f is already "
            "0.00, so no refinement was attempted and its two dots coincide.",
        ),
        dumbbell(
            "pure_vs_hybrid_acc_r_by_class.png",
            "Retention is essentially unchanged by refinement",
            "ACC_r on the 9,000 held-out images of the nine kept classes, by class",
            "ACC_r  (%)", order,
            {c: (float(p[c]["ACC_r"]), float(h[c]["ACC_r"])) for c in order},
            "higher is better",
            "Note the axis: the entire ten-class range spans about three points, and no single "
            "class moves by more than 0.25. Dots are used rather than bars precisely so a "
            "non-zero axis is legitimate. Frog is the one class that improves (+0.03), and airplane is "
            "the no-op, so its two dots coincide.",
        ),
        dumbbell(
            "pure_vs_hybrid_composite_by_class.png",
            "Nine of ten classes improve on the anchor composite",
            "composite = ACC_r x (1 - ACC_f), the anchor paper's own metric, by class",
            "composite  (%)", order,
            {c: (float(p[c]["composite"]), float(h[c]["composite"])) for c in order},
            "higher is better",
            "The tenth is the airplane no-op, not a regression. Truck gains the most (+10.24) "
            "and still finishes 19.00 points below the next-worst class. Airplane is the no-op, so "
            "its two dots coincide.",
            legend_loc="lower left",
        ),
    ]
    return out


# -------------------------------------------------------------- operator bars

def build_operator_frequency(pure) -> Path:
    freq: dict[str, int] = {}
    for r in pure:
        for op in r["operators"].split("|"):
            freq[op] = freq.get(op, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]))

    figure, ax = plt.subplots(figsize=(7.0, 3.6))
    ys = list(range(len(ranked)))
    names = [k for k, _ in ranked]
    counts = [v for _, v in ranked]
    colours = [HYBRID if n == "MASK" else NEUTRAL for n in names]

    ax.barh(ys, counts, height=0.62, color=colours, zorder=3)
    for y, c in zip(ys, counts):
        ax.annotate(f"{c}", (c, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8.5,
                    color=HYBRID if c == 10 else MUTED,
                    fontweight="bold" if c == 10 else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9, color=INK, family="monospace")
    ax.set_xlim(0, 11.4)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("number of the ten selected C* candidates containing the operator",
                  labelpad=8)
    bare(ax)

    figure.suptitle("MASK is in every selected candidate; no other operator is",
                    x=0.0, y=1.16, ha="left", fontsize=12, fontweight="bold", color=INK)
    figure.text(0.0, 1.05,
                "Operator frequency across the ten per-class C*, pure gradient-free search",
                fontsize=8.5, color=MUTED, ha="left")

    figure.text(0.0, -0.10,
                "Three classes select MASK alone (deer, dog, horse) and five select it alone or with a "
                "single partner.\nThis is a search outcome, not a design choice: all seven operators were "
                "available at equal cost in all ten runs.",
                fontsize=7.5, color=MUTED, ha="left")
    return save(figure, "operator_frequency_selected_cstar.png")


# ------------------------------------------------------------- benchmark panel

def build_benchmark(bench) -> Path:
    rows = {r["method"]: r for r in bench}
    anchor = rows["Kodge et al. 2024 (anchor)"]
    retrain = rows["Retraining (gold standard)"]
    pure = rows["MED-US pure (this work)"]
    hybrid = rows["MED-US hybrid (this work)"]

    def composite(r) -> float:
        return float(r["ACC_r_mean"]) * (1.0 - float(r["ACC_f_mean"]) / 100.0)

    panels = [
        ("ACC_r  (%)", "higher is better",
         [float(anchor["ACC_r_mean"]), float(pure["ACC_r_mean"]), float(hybrid["ACC_r_mean"])],
         float(retrain["ACC_r_mean"]), (93.4, 95.3)),
        ("ACC_f  (%)", "lower is better",
         [float(anchor["ACC_f_mean"]), float(pure["ACC_f_mean"]), float(hybrid["ACC_f_mean"])],
         float(retrain["ACC_f_mean"]), (-1.4, 15.0)),
        ("composite  (%)", "higher is better",
         [composite(anchor), composite(pure), composite(hybrid)],
         composite(retrain), (79.5, 96.5)),
        ("anchor MIA  (%)", "higher is better",
         [float(anchor["MIA_mean"]), float(pure["MIA_mean"]), float(hybrid["MIA_mean"])],
         float(retrain["MIA_mean"]), (90.5, 101.5)),
    ]
    labels = ["Kodge et al.\n(reported)", "MED-US pure\n(this work)",
              "MED-US hybrid\n(this work)"]
    colours = [THIRD, PURE, HYBRID]

    # Dots, not bars. Three of these four panels need a zoomed axis to show any
    # difference at all, and a bar whose baseline is not zero misstates the ratio
    # between its neighbours. A dot encodes position only, so the zoom is honest.
    figure, axes = plt.subplots(1, 4, figsize=(11.6, 2.5), sharey=True)
    ys = [2, 1, 0]

    for ax, (metric, better, values, gold, xlim) in zip(axes, panels):
        ax.axvline(gold, color=NEUTRAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        # Below the bottom row. The top of each panel is where the value labels
        # sit, and on the ACC_f panel the gold standard and the anchor are 0.03
        # apart -- a label there lands on top of the anchor's.
        ax.annotate(f"retraining {gold:.2f}", xy=(gold, -0.72), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.5, color=MUTED)
        for y, value, colour in zip(ys, values, colours):
            ax.scatter([value], [y], s=78, color=colour, zorder=4,
                       edgecolors=SURFACE, linewidths=1.4)
            ax.annotate(f"{value:.2f}", (value, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
        ax.set_xlim(*xlim)
        ax.set_ylim(-1.15, 2.75)
        ax.set_title(f"{metric}   {better}", loc="left", pad=10, fontsize=8.5)
        bare(ax)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=8, color=INK)

    figure.suptitle("Retention is competitive; forgetting is not",
                    x=0.0, y=1.19, ha="left", fontsize=12, fontweight="bold",
                    color=INK)
    figure.text(0.0, 1.08,
                "CIFAR-10 / ResNet-18, mean over all ten target classes. Each metric is a separate "
                "panel on its own scale -- the panels are not comparable to each other.",
                fontsize=8.5, color=MUTED, ha="left")
    figure.text(0.0, -0.30,
                "Kodge et al. rows are AS REPORTED in that paper's Table 1 and were not re-measured in "
                "this harness; the two MED-US rows are measured here.\nComposite is ACC_r x (1 - ACC_f), "
                "derived for all three rows from the ACC_r and ACC_f beside them. The dashed line is "
                "retraining from scratch, the gold standard.\nDots rather than bars because three of the "
                "four panels are zoomed, and a bar drawn from a non-zero baseline misstates the ratio "
                "between its neighbours.",
                fontsize=7.5, color=MUTED, ha="left")
    figure.tight_layout()
    return save(figure, "benchmark_comparison.png")


# --------------------------------------------------------------- truck panels

def build_truck(rows) -> Path:
    models: dict[str, list[tuple[str, float]]] = {}
    titles = {
        "W_0": "$W_0$  -- the original model",
        "W_ref": "$W_{ref}$  -- retain-only reference",
        "C_star_pure": "$C^*$  -- pure MED-US",
        "C_star_hybrid": "$C^*$  -- hybrid refinement",
    }
    for r in rows:
        models.setdefault(r["model"], []).append(
            (r["predicted_class_name"], float(r["percent"])))

    order = ["W_0", "W_ref", "C_star_pure", "C_star_hybrid"]
    classes = [c for c, _ in models["W_0"]]

    figure, axes = plt.subplots(1, 4, figsize=(11.4, 4.2), sharey=True)
    ys = list(range(len(classes)))[::-1]

    for ax, key in zip(axes, order):
        values = dict(models[key])
        heights = [values[c] for c in classes]
        colours = [HYBRID if c == "truck" else NEUTRAL for c in classes]
        ax.barh(ys, heights, height=0.68, color=colours, zorder=3)
        for y, c, v in zip(ys, classes, heights):
            if v < 0.05:
                continue
            ax.annotate(f"{v:.1f}", (v, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=HYBRID if c == "truck" else MUTED,
                        fontweight="bold" if c == "truck" else "normal")
        ax.set_xlim(0, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(titles[key], loc="left", pad=8, fontsize=9)
        bare(ax)
        still = values["truck"]
        ax.annotate(f"still truck: {still:.1f}%", xy=(1, 0), xycoords="axes fraction",
                    xytext=(0, -34), textcoords="offset points", ha="right",
                    fontsize=8, color=HYBRID if still > 0 else MUTED,
                    fontweight="bold")

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(classes, fontsize=8.5, color=INK)
    for ax in axes:
        ax.set_xlabel("% of the 1,000 truck test images", labelpad=6, fontsize=8)

    figure.suptitle("Unlearning truck moves it toward automobile, and stalls",
                    x=0.0, y=1.09, ha="left", fontsize=12, fontweight="bold",
                    color=INK)
    figure.text(0.0, 1.005,
                "Predicted class of every held-out truck image, under each of the four models. "
                "Inference only -- no model was trained or re-refined.",
                fontsize=8.5, color=MUTED, ha="left")
    figure.text(0.0, -0.135,
                "A model that never saw a truck sends 68.4% of them to automobile. Pure MED-US sends only "
                "16.7% there and leaves 42.1% still classified truck;\nthe refinement moves a further "
                "11.2 points out, and 20.4% now reach automobile. The failure is a partial move along the "
                "truck-automobile axis, not a\nrandom scattering: the reference's destination is the same "
                "destination, reached less far.",
                fontsize=7.5, color=MUTED, ha="left")
    figure.tight_layout()
    return save(figure, "truck_failure_analysis.png")


# ------------------------------------------------------------------------ main

def main() -> int:
    written: list[Path] = []
    pure = read("pure_medus_10_class_table.csv")
    hybrid = read("hybrid_medus_10_class_table.csv")

    written += build_dumbbells(pure, hybrid)
    written.append(build_operator_frequency(pure))
    written.append(build_benchmark(read("benchmark_comparison_table.csv")))

    truck_csv = PKG / "truck_prediction_distribution.csv"
    if truck_csv.is_file():
        written.append(build_truck(read(truck_csv.name)))
    else:
        print("  truck_prediction_distribution.csv absent -- run "
              "experiments/analyse_truck_predictions.py first; skipping figure 6")

    print(f"wrote {len(written)} figures to {FIG.relative_to(PROJECT_ROOT)}")
    for path in written:
        print(f"  {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
