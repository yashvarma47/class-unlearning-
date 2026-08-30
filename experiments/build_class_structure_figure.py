"""The class-structure figure, from the committed measurement only.

Sources, both already in the repository:

    results/analysis/class_structure/summary.json
    results/analysis/class_structure/channel_contrast_all_classes.csv

Nothing is trained, searched, refined or re-measured. The only computation is
arithmetic over those two files: cosine similarity between the per-class
channel-contrast vectors, and two correlation coefficients.

Three panels, because the honest answer needs all three:

**A -- how much forget-specific structure each class has.** 84 to 91 per cent of
channels stand above the noise floor for every class, against 0.55 per cent for
the predecessor project's instance-level forget set, where chance alone gives
1.00. That three-orders-of-magnitude gap is the project's founding result and it
holds for all ten classes.

**B -- whether that structure predicts difficulty. It does not.** Median SNR
against pure ``ACC_f`` is a null scatter, Pearson -0.04 over ten points. Truck is
sixth of ten on structure and worst by far on forgetting; automobile has the
*least* structure of any class and forgets three times better than truck. This
panel exists to stop a reader assuming the obvious explanation, which the data
does not support.

**C -- which classes share their structure with which.** Cosine similarity
between the per-class contrast vectors recovers the semantic grouping without
being told it -- vehicles with vehicles, animals with animals -- which is the
evidence that the measurement means something. Truck's nearest neighbour is
automobile, mutually, and that matches where truck images actually go when the
class is unlearned (``truck_failure_analysis.png``).

What this figure does **not** show, and must not be captioned as showing: that
similarity predicts difficulty. It does not -- Pearson -0.08, and airplane is the
counterexample, with the highest similarity to another class of any of the ten
and a perfect ``ACC_f`` of 0.00.

Also writes ``class_structure_similarity.csv``, the 10x10 matrix behind panel C.

Run::

    python experiments/build_class_structure_figure.py
"""

from __future__ import annotations

import collections
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCT = PROJECT_ROOT / "results" / "analysis" / "class_structure"
PKG = PROJECT_ROOT / "results" / "writeup_package"
FIG = PKG / "figures"

PURE = "#1b6ca8"
HYBRID = "#c1440e"
NEUTRAL = "#8c9196"
INK = "#22262b"
MUTED = "#5b6472"
GRID = "#dcdfe3"
SURFACE = "#ffffff"

#: The predecessor project's instance-level forget set, from the README: 0.55% of
#: channels above the noise floor, where the null control gives 1.00% by
#: construction. Fewer stood out than chance produces.
INSTANCE_LEVEL_PCT = 0.55
NULL_CONTROL_PCT = 1.00

ORDER = ["airplane", "automobile", "bird", "cat", "deer", "dog",
         "frog", "horse", "ship", "truck"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})


def bare(ax, *, grid_axis: str = "x") -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den


def load() -> tuple[dict, dict, dict]:
    summary = json.loads((STRUCT / "summary.json").read_text(encoding="utf-8"))
    stats = {r["class_name"]: r for r in summary["ranked"]}

    with (PKG / "pure_medus_10_class_table.csv").open(newline="", encoding="utf-8") as fh:
        acc_f = {r["class_name"]: float(r["ACC_f"]) for r in csv.DictReader(fh)}

    vectors: dict[str, dict] = collections.defaultdict(dict)
    with (STRUCT / "channel_contrast_all_classes.csv").open(
            newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row["group"], row["module"], row["channel"])
            vectors[row["class_name"]][key] = float(row["contrast_forget_vs_retain"])
    return stats, acc_f, vectors


def similarity(vectors: dict[str, dict]) -> dict[str, dict[str, float]]:
    keys = sorted(vectors[ORDER[0]])
    norms = {c: sum(vectors[c][k] ** 2 for k in keys) ** 0.5 for c in ORDER}
    out: dict[str, dict[str, float]] = {}
    for a in ORDER:
        out[a] = {}
        for b in ORDER:
            dot = sum(vectors[a][k] * vectors[b][k] for k in keys)
            out[a][b] = dot / (norms[a] * norms[b])
    return out


def main() -> int:
    if not (STRUCT / "summary.json").is_file():
        print("class-structure artefacts absent; nothing to plot")
        return 1

    stats, acc_f, vectors = load()
    sim = similarity(vectors)
    FIG.mkdir(parents=True, exist_ok=True)

    with (PKG / "class_structure_similarity.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class_name"] + ORDER)
        for a in ORDER:
            writer.writerow([a] + [f"{sim[a][b]:.6f}" for b in ORDER])

    figure = plt.figure(figsize=(13.0, 4.3))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.25], wspace=0.42)

    # --- A: how much structure ------------------------------------------------
    ax = figure.add_subplot(grid[0, 0])
    ranked = sorted(ORDER, key=lambda c: stats[c]["pct_beyond_noise"])
    ys = range(len(ranked))
    vals = [stats[c]["pct_beyond_noise"] for c in ranked]
    ax.barh(list(ys), vals, height=0.62,
            color=[HYBRID if c == "truck" else NEUTRAL for c in ranked], zorder=3)
    for y, c, v in zip(ys, ranked, vals):
        ax.annotate(f"{v:.1f}", (v, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.5,
                    color=HYBRID if c == "truck" else MUTED,
                    fontweight="bold" if c == "truck" else "normal")
    ax.axvline(INSTANCE_LEVEL_PCT, color=INK, linewidth=1.2,
               linestyle=(0, (3, 2)), zorder=4)
    ax.annotate("instance-level $D_f$: 0.55%\n(chance alone gives 1.00%)",
                xy=(INSTANCE_LEVEL_PCT, len(ranked) - 0.4), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=7, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(ranked, fontsize=8.5, color=INK)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of channels above the noise floor", labelpad=6, fontsize=8)
    ax.set_title("A.  Every class has forget-specific structure", loc="left", pad=10)
    bare(ax)

    # --- B: does it predict difficulty? --------------------------------------
    ax = figure.add_subplot(grid[0, 1])
    xs = [stats[c]["median_snr"] for c in ORDER]
    ys2 = [acc_f[c] for c in ORDER]
    r = pearson(xs, ys2)
    for c, x, y in zip(ORDER, xs, ys2):
        highlight = c in ("truck", "airplane", "automobile")
        ax.scatter([x], [y], s=86 if highlight else 58,
                   color=HYBRID if highlight else PURE, zorder=4,
                   edgecolors=SURFACE, linewidths=1.4)
        if highlight:
            # truck sits high enough that a label above it collides with the
            # correlation annotation; put its label beside the dot instead.
            offset, ha = ((10, -3), "left") if c == "truck" else ((0, 11), "center")
            ax.annotate(c, (x, y), xytext=offset, textcoords="offset points",
                        ha=ha, va="center" if c == "truck" else "baseline",
                        fontsize=8, color=HYBRID, fontweight="bold")
    ax.set_xlim(2.2, 18.5)
    ax.set_ylim(-4.5, 50)
    ax.set_xlabel("median SNR of the class's channel contrast", labelpad=6, fontsize=8)
    ax.set_ylabel("pure MED-US  ACC_f  (%)", labelpad=6, fontsize=8)
    ax.set_title("B.  ...and it does not predict how hard the class is",
                 loc="left", pad=10)
    ax.annotate(f"Pearson r = {r:+.2f}   (n = 10)\nno relationship",
                xy=(0.03, 0.96), xycoords="axes fraction", ha="left", va="top",
                fontsize=8, color=INK)
    bare(ax, grid_axis="both")

    # --- C: who shares structure with whom -----------------------------------
    ax = figure.add_subplot(grid[0, 2])
    ramp = LinearSegmentedColormap.from_list("contrast", ["#f4f7fa", PURE])
    matrix = [[sim[a][b] if a != b else float("nan") for b in ORDER] for a in ORDER]
    finite = [v for row in matrix for v in row if v == v]
    image = ax.imshow(matrix, cmap=ramp, vmin=min(finite), vmax=max(finite))
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.5, color=NEUTRAL)
                continue
            value = sim[a][b]
            strong = value > (min(finite) + max(finite)) / 2
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.2,
                    color=SURFACE if strong else MUTED)
    ti, ai = ORDER.index("truck"), ORDER.index("automobile")
    for y, x in ((ti, ai), (ai, ti)):
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                   edgecolor=HYBRID, linewidth=2.0, zorder=5))
    ax.set_xticks(range(len(ORDER)))
    ax.set_yticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=7.5, color=INK)
    ax.set_yticklabels(ORDER, fontsize=7.5, color=INK)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("C.  Truck shares most of its structure with automobile",
                 loc="left", pad=10)
    bar = figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7)
    bar.set_label("cosine similarity of channel-contrast vectors", fontsize=7,
                  color=MUTED)

    figure.suptitle("Class structure explains the regime, not the ranking",
                    x=0.0, y=1.13, ha="left", fontsize=12.5, fontweight="bold",
                    color=INK)
    figure.text(0.0, 1.035,
                "Per-channel activation contrast between the forget class and the nine "
                "retained classes, measured on W_0 against a null control built from two "
                "disjoint halves of D_r.",
                fontsize=8.5, color=MUTED, ha="left")

    truck_sim = sim["truck"]["automobile"]
    max_sim = max(sim[a][b] for a in ORDER for b in ORDER if a != b)
    figure.text(0.0, -0.235,
                "A: the founding result of the project, holding for all ten classes -- a class-level "
                "forget set has structure an instance-level one does not.\n"
                "B: the obvious follow-on hypothesis, and it fails. Truck is sixth of ten on median "
                "SNR and fifth on channels above the floor, yet worst by far on forgetting; "
                "automobile has the LEAST structure of any class and forgets 3.3x better.\n"
                f"C: the matrix recovers the semantic grouping without being told it, which is why it "
                f"can be trusted. Truck's nearest neighbour is automobile ({truck_sim:.2f}), mutually, "
                f"matching where truck\nimages actually go when the class is unlearned. But similarity "
                f"does not predict difficulty either (r = {pearson([max(sim[a][b] for b in ORDER if b != a) for a in ORDER], [acc_f[c] for c in ORDER]):+.2f}): "
                f"airplane has the highest similarity to another class of\nany of the ten "
                f"({max_sim:.2f}, with ship) and still reaches ACC_f 0.00.",
                fontsize=7.5, color=MUTED, ha="left")

    out = FIG / "class_structure_analysis.png"
    figure.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)

    print(f"wrote {out.relative_to(PROJECT_ROOT)}  "
          f"({out.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {(PKG / 'class_structure_similarity.csv').relative_to(PROJECT_ROOT)}")
    print(f"\n  pct above noise floor   {min(v['pct_beyond_noise'] for v in stats.values()):.1f} "
          f"to {max(v['pct_beyond_noise'] for v in stats.values()):.1f}  "
          f"(instance-level {INSTANCE_LEVEL_PCT}, null control {NULL_CONTROL_PCT})")
    print(f"  r(median SNR, ACC_f)    {r:+.3f}   -- null")
    print(f"  truck nearest neighbour automobile ({truck_sim:.3f}), mutual")
    return 0


if __name__ == "__main__":
    sys.exit(main())
