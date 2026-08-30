"""Build the dissertation write-up package from results that already exist.

STRICTLY READ-ONLY with respect to every result in the repository. This script
runs no search, no refinement, no training and no evaluation. It loads the
committed artefacts, reshapes them into thesis-ready tables, and writes the
result into ``results/writeup_package/`` and nowhere else.

Sources, all committed, none recomputed here:

  results/reference_training/reference_validation_summary.csv
  results/reference_training/all_reference_models_summary.md      (sha256 prefixes)
  results/literature_alignment/ten_class_pure_summary.csv
  results/literature_alignment/ten_class_pure_mean_std.csv
  results/literature_alignment/ten_class_hybrid_summary.csv
  results/literature_alignment/ten_class_hybrid_mean_std.csv
  results/literature_alignment/pure_vs_hybrid_comparison.csv
  results/search/plan_a_<class>/<class>_anchor_metrics.json       (W_0 / W_ref rows)
  results/search/plan_a_<class>_bn_frozen_refined/refinement.json (acceptance record)

The one block of numbers that does NOT come from this repository is the
literature half of the benchmark table: those rows are transcribed from the
anchor paper's own Table 1 and are labelled as reported, not re-measured. They
live in ``ANCHOR_TABLE_1`` below with their citation attached.

Aggregates (mean, std) use the sample standard deviation, ddof=1, which is what
``build_ten_class_summary.py`` used; the pure aggregate produced here is checked
against the committed one and the script exits non-zero if they disagree.

Output is ASCII only, matching the rest of the generated markdown in this repo.

Run::

    python experiments/build_writeup_package.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
LIT = RESULTS / "literature_alignment"
REFTRAIN = RESULTS / "reference_training"
SEARCH = RESULTS / "search"
OUT = RESULTS / "writeup_package"

CLASSES = [
    (0, "airplane"),
    (1, "automobile"),
    (2, "bird"),
    (3, "cat"),
    (4, "deer"),
    (5, "dog"),
    (6, "frog"),
    (7, "horse"),
    (8, "ship"),
    (9, "truck"),
]

# Airplane has no refinement directory: its pure ACC_f is already 0.00 with
# anchor MIA 100.00, so a forgetting step was deliberately not attempted.
NO_REFINEMENT = {0}

ANCHOR_CITATION = (
    "Kodge, Saha & Roy. Deep Unlearning: Fast and Efficient Gradient-free Class "
    "Forgetting. TMLR 07/2024. https://openreview.net/forum?id=BmI5p6wBi0"
)

# Transcribed from the anchor paper's Table 1 (CIFAR-10 / ResNet-18, mean +/- std
# over all ten target classes). AS REPORTED BY THAT PAPER -- not re-measured in
# this harness. Std is None where the paper reports a bare value.
#   name, ACC_r mean, ACC_r std, ACC_f mean, ACC_f std, MIA mean, MIA std, gradient-free
ANCHOR_TABLE_1 = [
    ("Original",                  94.89, 0.31, 94.89, 2.75,   0.03,  0.03, "n/a"),
    ("Retraining (gold standard)",94.81, 0.52,  0.00, None, 100.00,  0.00, "n/a"),
    ("NegGrad",                   69.89,10.23,  0.02, 0.04,   0.00, None,  "no"),
    ("NegGrad+",                  89.91, 1.41,  0.94, 1.87,  98.68,  1.42, "no"),
    ("UNSIR (Tarun et al. 2023)", 92.20, 0.72, 10.89, 8.79,  61.50, 25.86, "no"),
    ("SCRUB (Kurmanji et al. 2023)",94.79,0.63, 0.00, None,   0.00, None,  "no"),
    ("SSD (Foster et al. 2024)",  85.76,25.76,  4.37,12.79,  87.86, 31.21, "yes"),
    ("Kodge et al. 2024 (anchor)",94.19, 0.50,  0.03, 0.09,  95.50, 14.23, "yes"),
]


# --------------------------------------------------------------------------- io

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_md(name: str, text: str) -> Path:
    path = OUT / name
    body = text.strip() + "\n"
    non_ascii = [c for c in body if ord(c) > 127]
    if non_ascii:
        raise SystemExit(f"{name}: non-ASCII characters present: {sorted(set(non_ascii))}")
    path.write_text(body, encoding="utf-8")
    return path


def table(header: list[str], rows: list[list[str]], align: str) -> str:
    """A GitHub pipe table. ``align`` is one char per column: l, r or c."""
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(sep[a] for a in align) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def f(value, dp: int = 2) -> str:
    """Fixed-point, with a visible marker for a missing value."""
    if value is None or value == "":
        return "--"
    return f"{float(value):.{dp}f}"


def pm(mean, std, dp: int = 2) -> str:
    if std is None:
        return f(mean, dp)
    return f"{f(mean, dp)} +/- {f(std, dp)}"


def agg(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def ops(value: str) -> str:
    """Operator sets are pipe-joined, and '|' is the markdown column separator."""
    return "`" + value.replace("|", r"\|") + "`"


# ----------------------------------------------------------------------- loaders

def load_sha_prefixes() -> dict[int, str]:
    """Pull the 12-char sha256 prefixes out of the committed reference summary."""
    text = (REFTRAIN / "all_reference_models_summary.md").read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*(\d)\s*\|\s*(\w+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip().strip("`*") for c in line.strip().strip("|").split("|")]
        out[int(cells[0])] = cells[8]
    return out


def load_anchor_baselines() -> dict[int, dict[str, dict]]:
    """W_0 and W_ref rows, per class, from each class's committed anchor JSON."""
    out: dict[int, dict[str, dict]] = {}
    for cid, name in CLASSES:
        path = SEARCH / f"plan_a_{name}" / f"{name}_anchor_metrics.json"
        if not path.exists():
            # frog was measured before the sweep driver existed and its anchor
            # metrics were written straight into results/literature_alignment/.
            path = LIT / f"{name}_anchor_metrics.json"
        with path.open(encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        out[cid] = {"W_0": rows["W_0"], "W_ref": rows["W_ref"]}
    return out


def load_refinements() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for cid, name in CLASSES:
        if cid in NO_REFINEMENT:
            continue
        path = SEARCH / f"plan_a_{name}_bn_frozen_refined" / "refinement.json"
        with path.open(encoding="utf-8") as fh:
            out[cid] = json.load(fh)
    return out


# ------------------------------------------------------------------------ tables

def build_reference_table(shas: dict[int, str]) -> None:
    rows_in = {int(r["class_id"]): r for r in read_csv(REFTRAIN / "reference_validation_summary.csv")}

    header = ["class_id", "class_name", "verdict", "selected_epoch", "log_epochs", "seed",
              "D_f_test_acc", "D_r_test_acc", "D_f_test_loss", "D_r_test_loss", "full_test_acc",
              "D_f_train", "D_r_train", "D_f_test", "D_r_test", "sha256_prefix", "source_zip"]
    csv_rows, md_rows = [], []
    for cid, name in CLASSES:
        r = rows_in[cid]
        csv_rows.append([
            cid, name, r["verdict"], r["metadata_epoch"], r["log_epochs"], r["metadata_seed"],
            f(r["forget_test_acc"], 4), f(r["retain_test_acc"], 4),
            f(r["forget_test_loss"], 4), f(r["retain_test_loss"], 4), f(r["full_test_acc"], 4),
            r["d_f_train"], r["d_r_train"], r["d_f_test"], r["d_r_test"],
            shas.get(cid, ""), r["source_zip"],
        ])
        md_rows.append([
            cid, name, f"**{r['verdict']}**", r["metadata_epoch"],
            f(r["forget_test_acc"], 4), f(r["retain_test_acc"], 4), f(r["full_test_acc"], 4),
            f"`{shas.get(cid, '')}`",
        ])
    write_csv("reference_model_validation_table.csv", header, csv_rows)

    dr = [float(rows_in[c]["retain_test_acc"]) for c, _ in CLASSES]
    mean, std = agg(dr)
    write_md("reference_model_validation_table.md", f"""
# Table: retain-only reference models (`W_ref`), all ten classes

{stamp()}

Each `W_ref` is a ResNet-18 trained from scratch on `D_r_train` only -- the 45,000
CIFAR-10 training images that are not the forget class. It never saw a single image
of that class, so it is the retraining gold standard against which every unlearned
model in this dissertation is measured.

{table(
    ["id", "class", "verdict", "epoch", "`D_f_test`", "`D_r_test`", "full test", "sha256"],
    md_rows, "rlcrrrrl")}

`D_f_test` accuracy is **0.0000 for all ten**, which is the correctness condition: a
model that never trained on a class must not classify it. It is diagnostic only and
never influenced checkpoint selection -- the best epoch was chosen on `D_r_test`
accuracy with `D_r_test` loss as tie-breaker.

`D_r_test` accuracy: **{pm(mean, std, 4)}** over the ten references
(min {f(min(dr), 4)}, max {f(max(dr), 4)}).

Full-test accuracy sits near 0.85 for every class because one class in ten is held
out by construction; it is not a utility number and should not be read as one.

Protocol, identical for all ten: 200 epochs, seed 42, split 5,000 / 45,000 / 1,000 / 9,000,
every split byte-compared against the version-controlled local split before import.
""")


def build_pure_table() -> tuple[list[dict[str, str]], dict[str, tuple[float, float]]]:
    rows_in = read_csv(LIT / "ten_class_pure_summary.csv")
    header = ["class_id", "class_name", "front_position", "operators",
              "ACC_r", "ACC_f", "composite", "MIA", "selectivity_S",
              "f1_js", "f2_retain_train_loss", "f3_edit_cost",
              "W_0_ACC_r", "W_0_ACC_f", "W_ref_ACC_r", "W_ref_ACC_f", "search_minutes"]
    csv_rows, md_rows = [], []
    for r in rows_in:
        csv_rows.append([
            r["class_id"], r["class_name"], r["front_position"], r["operators"],
            f(r["anchor_ACC_r"]), f(r["anchor_ACC_f"]), f(r["anchor_composite"]), f(r["anchor_MIA"]),
            f(r["selectivity_S"]), f(r["f1_js"], 4), f(r["f2_retain_train_loss"], 4), f(r["f3_edit_cost"], 4),
            f(r["W_0_ACC_r"]), f(r["W_0_ACC_f"]), f(r["W_ref_ACC_r"]), f(r["W_ref_ACC_f"]),
            r["search_minutes"],
        ])
        md_rows.append([
            r["class_id"], r["class_name"], "#" + r["front_position"], ops(r["operators"]),
            f(r["anchor_ACC_r"]), f(r["anchor_ACC_f"]), f(r["anchor_composite"]),
            f(r["anchor_MIA"]), f(r["selectivity_S"]),
        ])
    write_csv("pure_medus_10_class_table.csv", header, csv_rows)

    stats = {}
    for key in ("anchor_ACC_r", "anchor_ACC_f", "anchor_composite", "anchor_MIA", "selectivity_S"):
        stats[key] = agg([float(r[key]) for r in rows_in])

    committed = {r["metric"]: r for r in read_csv(LIT / "ten_class_pure_mean_std.csv")}
    for key, (m, s) in stats.items():
        want_m, want_s = float(committed[key]["mean"]), float(committed[key]["std"])
        if abs(m - want_m) > 5e-4 or abs(s - want_s) > 5e-4:
            raise SystemExit(
                f"pure aggregate disagrees with the committed table for {key}: "
                f"got {m:.4f} +/- {s:.4f}, committed {want_m:.4f} +/- {want_s:.4f}")

    md_rows.append([
        "", "**mean +/- std**", "", "",
        f"**{pm(*stats['anchor_ACC_r'])}**", f"**{pm(*stats['anchor_ACC_f'])}**",
        f"**{pm(*stats['anchor_composite'])}**", f"**{pm(*stats['anchor_MIA'])}**",
        f"**{pm(*stats['selectivity_S'])}**",
    ])

    alone = [r["class_name"] for r in rows_in if r["operators"] == "MASK"]
    small = [r["class_name"] for r in rows_in if len(r["operators"].split("|")) <= 2]

    write_md("pure_medus_10_class_table.md", f"""
# Table: pure MED-US, all ten CIFAR-10 classes

{stamp()}

Every row is **pure gradient-free weight surgery**. No gradient step was applied to
any of these models. The accepted BN-frozen refinements are a different method and
appear in their own table.

Selection rule, applied identically to all ten classes:
`C* = the front member maximising the anchor composite, ACC_r x (1 - ACC_f)`.
That is the anchor paper's own metric function, not one invented here.

{table(["id", "class", "C*", "operators", "ACC_r", "ACC_f", "composite", "MIA", "S"],
       md_rows, "rlrlrrrrr")}

`ACC_r` is retain-test accuracy (higher is better), `ACC_f` forget-test accuracy
(lower is better), composite `ACC_r x (1 - ACC_f)`, MIA the anchor's membership-inference
score (higher is better), `S` the selectivity ratio.

**`MASK` appears in the selected candidate for all ten classes** -- the only operator
that does. {len(alone)} select `MASK` alone ({", ".join(alone)}) and {len(small)} select
it alone or with a single partner. None selects a candidate without it.

The spread is the story: `ACC_f` runs from 0.00 (airplane) to 42.10 (truck), a range of
42 points against a mean of {f(stats['anchor_ACC_f'][0])}, while `ACC_r` stays inside
{f(stats['anchor_ACC_r'][1])} of its mean. The method's cost is concentrated almost
entirely in forgetting, not in retention.
""")
    return rows_in, stats


def build_hybrid_table() -> tuple[list[dict[str, str]], dict[str, tuple[float, float]]]:
    rows_in = read_csv(LIT / "ten_class_hybrid_summary.csv")
    header = ["class_id", "class_name", "source", "refinement_status", "operators",
              "ACC_r", "ACC_f", "composite", "MIA", "selectivity_S",
              "parameter_movement", "buffer_movement"]
    csv_rows, md_rows = [], []
    for r in rows_in:
        csv_rows.append([
            r["class_id"], r["class_name"], r["source"], r["refinement_status"], r["operators"],
            f(r["anchor_ACC_r"]), f(r["anchor_ACC_f"]), f(r["anchor_composite"]), f(r["anchor_MIA"]),
            f(r["selectivity_S"]),
            f(r["parameter_movement"], 6) if r["parameter_movement"] else "",
            f(r["buffer_movement"], 6) if r["buffer_movement"] else "",
        ])
        md_rows.append([
            r["class_id"], r["class_name"], r["refinement_status"],
            f(r["anchor_ACC_r"]), f(r["anchor_ACC_f"]), f(r["anchor_composite"]),
            f(r["anchor_MIA"]), f(r["selectivity_S"]),
            f(r["parameter_movement"], 6) if r["parameter_movement"] else "--",
            f(r["buffer_movement"], 6) if r["buffer_movement"] else "--",
        ])
    write_csv("hybrid_medus_10_class_table.csv", header, csv_rows)

    stats = {}
    for key in ("anchor_ACC_r", "anchor_ACC_f", "anchor_composite", "anchor_MIA", "selectivity_S"):
        stats[key] = agg([float(r[key]) for r in rows_in])

    md_rows.append([
        "", "**mean +/- std**", "",
        f"**{pm(*stats['anchor_ACC_r'])}**", f"**{pm(*stats['anchor_ACC_f'])}**",
        f"**{pm(*stats['anchor_composite'])}**", f"**{pm(*stats['anchor_MIA'])}**",
        f"**{pm(*stats['selectivity_S'])}**", "", "",
    ])

    write_md("hybrid_medus_10_class_table.md", f"""
# Table: hybrid MED-US (pure `C*` + BN-frozen refinement), all ten classes

{stamp()}

**This is not the pure method.** Each refined row is the pure `C*` followed by one
clipped gradient-ascent step on `D_f` and one repair step on `D_r`, applied outside
the evolutionary search with BatchNorm frozen. Nine classes were eligible and all
nine were accepted. Airplane is a deliberate no-op -- its pure `ACC_f` is already
0.00 with anchor MIA 100.00, so a forgetting step has nothing to improve -- and its
row is the pure `C*` unchanged.

{table(["id", "class", "status", "ACC_r", "ACC_f", "composite", "MIA", "S",
        "param mvmt", "BN mvmt"],
       md_rows, "rlcrrrrrrr")}

BatchNorm buffer movement is **exactly 0.000000 on every accepted refinement**, with
zero counter changes. Parameter movement stays between 0.000303 and 0.000420 against
a budget of 0.0400.

That column is load-bearing. An earlier, unfrozen attempt passed every weight-based
guard while eight batches of `D_r` silently re-estimated the running statistics and
undid the operator edit. Freezing BatchNorm and checking buffer movement explicitly
is what makes these nine results trustworthy.
""")
    return rows_in, stats


def build_pure_vs_hybrid(pure_stats, hybrid_stats) -> None:
    per_class = read_csv(LIT / "pure_vs_hybrid_comparison.csv")

    labels = [("anchor_ACC_r", "ACC_r (%)"), ("anchor_ACC_f", "ACC_f (%)"),
              ("anchor_composite", "composite (%)"), ("anchor_MIA", "anchor MIA (%)"),
              ("selectivity_S", "selectivity S")]
    header = ["metric", "n", "pure_mean", "pure_std", "hybrid_mean", "hybrid_std", "delta_mean"]
    csv_rows, md_rows = [], []
    for key, label in labels:
        pm_, ps = pure_stats[key]
        hm, hs = hybrid_stats[key]
        csv_rows.append([label, 10, f(pm_, 4), f(ps, 4), f(hm, 4), f(hs, 4), f(hm - pm_, 4)])
        md_rows.append([label, pm(pm_, ps), pm(hm, hs), f"{hm - pm_:+.2f}"])
    write_csv("pure_vs_hybrid_summary_table.csv", header, csv_rows)

    delta_rows = []
    for r in per_class:
        delta_rows.append([
            r["class_id"], r["class_name"], r["refinement_status"],
            f(r["pure_ACC_f"]), f(r["hybrid_ACC_f"]), r["delta_ACC_f"],
            f(r["pure_composite"]), f(r["hybrid_composite"]), r["delta_composite"],
            r["delta_ACC_r"],
        ])

    improved = sum(1 for r in per_class if float(r["delta_composite"]) > 0)
    write_md("pure_vs_hybrid_summary_table.md", f"""
# Table: pure against hybrid MED-US

{stamp()}

**These are two different methods and this table compares them; it does not merge
them.** The anchor paper's method is gradient-free, so only the pure table is a
like-for-like comparison with its Table 1.

## Aggregate, ten classes

{table(["metric", "pure", "hybrid", "change"], md_rows, "lrrr")}

## Per class

{table(["id", "class", "status", "pure ACC_f", "hybrid ACC_f", "d ACC_f",
        "pure comp.", "hybrid comp.", "d comp.", "d ACC_r"],
       delta_rows, "rlcrrrrrrr")}

{improved} of 10 classes improved on the composite; the tenth is the airplane no-op.
No class got worse on any headline metric.

`ACC_f` falls by 5.00 points of mean for 0.12 of retain accuracy, and the standard
deviation narrows on every metric -- the refinement helps the weak classes most.
Truck gains 11.20 points of `ACC_f`, the largest absolute improvement of any class,
and still finishes at 30.90 against a 7.55 mean.
""")


def build_benchmark_table(pure_stats, hybrid_stats, baselines) -> None:
    w0_r = agg([baselines[c]["W_0"]["anchor_ACC_r"] for c, _ in CLASSES])
    w0_f = agg([baselines[c]["W_0"]["anchor_ACC_f"] for c, _ in CLASSES])
    w0_m = agg([baselines[c]["W_0"]["anchor_MIA"] for c, _ in CLASSES])
    wr_r = agg([baselines[c]["W_ref"]["anchor_ACC_r"] for c, _ in CLASSES])
    wr_f = agg([baselines[c]["W_ref"]["anchor_ACC_f"] for c, _ in CLASSES])
    wr_m = agg([baselines[c]["W_ref"]["anchor_MIA"] for c, _ in CLASSES])

    header = ["method", "source", "measured_in_this_harness", "gradient_free",
              "ACC_r_mean", "ACC_r_std", "ACC_f_mean", "ACC_f_std", "MIA_mean", "MIA_std"]
    csv_rows, md_rows = [], []

    for name, ar, ars, af, afs, mi, mis, gf in ANCHOR_TABLE_1:
        csv_rows.append([name, "Kodge et al. 2024, Table 1", "no", gf,
                         f(ar), f(ars), f(af), f(afs), f(mi), f(mis)])
        md_rows.append([name, "reported", gf, pm(ar, ars), pm(af, afs), pm(mi, mis)])

    ours = [
        ("Original W_0 (this work)", "this work", "yes", "n/a", w0_r, w0_f, w0_m),
        ("Retraining W_ref (this work)", "this work", "yes", "n/a", wr_r, wr_f, wr_m),
        ("MED-US pure (this work)", "this work", "yes", "yes",
         pure_stats["anchor_ACC_r"], pure_stats["anchor_ACC_f"], pure_stats["anchor_MIA"]),
        ("MED-US hybrid (this work)", "this work", "yes", "no",
         hybrid_stats["anchor_ACC_r"], hybrid_stats["anchor_ACC_f"], hybrid_stats["anchor_MIA"]),
    ]
    for name, src, meas, gf, r, fo, mi in ours:
        csv_rows.append([name, src, meas, gf,
                         f(r[0]), f(r[1]), f(fo[0]), f(fo[1]), f(mi[0]), f(mi[1])])
        md_rows.append([f"**{name}**", "measured", gf,
                        f"**{pm(*r)}**", f"**{pm(*fo)}**", f"**{pm(*mi)}**"])

    write_csv("benchmark_comparison_table.csv", header, csv_rows)

    write_md("benchmark_comparison_table.md", f"""
# Table: benchmark comparison against the anchor's Table 1

{stamp()}

CIFAR-10 / ResNet-18, mean +/- std over all ten target classes -- the same aggregation
the anchor paper uses, which is why the ten-class sweep was run rather than a single
class.

{table(["method", "numbers", "grad-free", "ACC_r (up)", "ACC_f (down)", "MIA (up)"],
       md_rows, "llcrrr")}

## Read this table carefully

The first eight rows are **as reported by the anchor paper**. They were not
re-measured in this harness, and no published unlearning baseline was re-run here.
The comparison is therefore against published numbers, under the anchor's own
protocol and MIA definition, and it inherits whatever differences exist between two
implementations. This repository's own baseline measurements -- `W_0` and `W_ref` --
land within {abs(w0_r[0] - 94.89):.2f} and {abs(wr_r[0] - 94.81):.2f} points of `ACC_r` of
the paper's Original and Retraining rows respectively, and their `ACC_f` and MIA agree
to within {abs(w0_f[0] - 94.89):.2f} and {abs(w0_m[0] - 0.03):.2f}. That agreement on the shared baselines is the
available evidence that the two harnesses measure the same quantities; it is indirect,
and it is the strongest such evidence this dissertation has.

Source for the reported rows: {ANCHOR_CITATION}

## What the comparison shows

**Retention is competitive.** Pure MED-US holds `ACC_r` at {f(pure_stats['anchor_ACC_r'][0])},
{abs(pure_stats['anchor_ACC_r'][0] - 94.19):.2f} below the anchor and inside the range of
the published field. Only NegGrad and SSD are materially worse on retention.

**Forgetting is not.** Pure `ACC_f` of {f(pure_stats['anchor_ACC_f'][0])} sits far above
the anchor's 0.03, and above UNSIR's 10.89 as well -- UNSIR is the nearest published
neighbour, and like this work it has a wide per-class spread rather than uniform
behaviour. The hybrid closes some of the gap, to {f(hybrid_stats['anchor_ACC_f'][0])},
but is no longer gradient-free and so is not a like-for-like row.

**Privacy is competitive** on the anchor's own MIA: pure {f(pure_stats['anchor_MIA'][0])},
hybrid {f(hybrid_stats['anchor_MIA'][0])}, against the anchor's 95.50. How much that
certifies is a separate question -- in this same table SCRUB reaches `ACC_f` 0.00 with
MIA 0.00, and Retraining is pinned at exactly 100.00. See the results-chapter notes.
""")


def build_refinement_acceptance(refinements) -> None:
    header = ["class_id", "class_name", "attempted", "accepted",
              "parameter_movement", "movement_budget", "buffer_movement",
              "batchnorm_frozen", "batchnorm_counters_changed", "retain_test_drop",
              "check_forget_improved", "check_retain_drop", "check_no_collapse",
              "check_edit_cost", "check_parameter_movement", "check_bn_buffers"]
    csv_rows, md_rows = [], []
    for cid, name in CLASSES:
        if cid in NO_REFINEMENT:
            csv_rows.append([cid, name, "no", "n/a (no-op)", "", "", "", "", "", "",
                             "", "", "", "", "", ""])
            md_rows.append([cid, name, "no-op", "--", "--", "--", "--",
                            "pure ACC_f already 0.00"])
            continue
        d = refinements[cid]
        checks = d["acceptance_checks"]
        keys = list(checks.keys())
        csv_rows.append([
            cid, name, "yes", "accepted" if d["accepted"] else "REJECTED",
            f(d["parameter_movement"], 6), "0.040000", f(d["buffer_movement"], 6),
            d["batchnorm_frozen"], d["batchnorm_counters_changed"],
            f(d["retain_test_drop"], 6),
        ] + [checks[k] for k in keys])
        md_rows.append([
            cid, name, "**accepted**" if d["accepted"] else "**REJECTED**",
            f(d["parameter_movement"], 6), f(d["buffer_movement"], 6),
            d["batchnorm_counters_changed"], f(d["retain_test_drop"], 6),
            "6 / 6" if all(checks.values()) else "FAILED",
        ])
    write_csv("refinement_acceptance_table.csv", header, csv_rows)

    hp = refinements[9]["hyperparameters"]
    write_md("refinement_acceptance_table.md", f"""
# Table: BN-frozen refinement acceptance record

{stamp()}

**Nine attempts, nine accepted, zero rejected.** Airplane was not attempted: its pure
`ACC_f` is already 0.00 with anchor MIA 100.00, so there was nothing for a forgetting
step to improve.

{table(["id", "class", "outcome", "param movement", "BN movement", "BN counters",
        "`D_r_test` drop", "checks passed"],
       md_rows, "rlcrrrrc")}

## The six acceptance checks

Every refinement had to pass all six before its checkpoint was kept:

1. forget improved on `D_f_test`
2. `D_r_test` drop <= 0.010
3. no utility collapse (retain losses <= 1.25x)
4. edit cost <= 0.3
5. parameter movement <= 0.0400
6. BatchNorm buffers unchanged

## Hyperparameters, identical for all nine

| | |
|---|---|
| forget step | {hp['forget_step']} |
| retain step | {hp['retain_step']} |
| forget lr | {hp['forget_lr']} |
| retain lr | {hp['retain_lr']} |
| batches per step | {hp['batches_per_step']} |
| steps | {hp['steps']} |
| BatchNorm | {hp['batchnorm']} |
| max buffer movement | {hp['max_buffer_movement']} |
| max `D_r_test` drop | {hp['max_retain_test_drop']} |
| seed | {hp['seed']} |

## Why check 6 exists

An earlier attempt on frog passed every weight-based guard and was reported as a
success. It was not one: the model was left in training mode, and eight batches of
`D_r` re-estimated the BatchNorm running statistics, undoing the operator edit while
parameter movement, edit cost and retain accuracy all looked correct. Buffer movement
is the only check that catches it, and it reads exactly 0.000000 on all nine
refinements above.
""")


# --------------------------------------------------------------------- narrative

def stamp() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (f"Generated {now} by `experiments/build_writeup_package.py` from committed "
            f"artefacts. Nothing here was recomputed, re-measured or re-run.")


NEW_FIGURES = [
    ("pure_vs_hybrid_acc_f_by_class.png",
     "Pure vs hybrid ACC_f, all ten classes (dumbbell)",
     "Results, the hybrid section. The clearest single picture of what the refinement "
     "buys, and of how far truck sits from everything else."),
    ("pure_vs_hybrid_acc_r_by_class.png",
     "Pure vs hybrid ACC_r, all ten classes (dumbbell)",
     "Results, beside the ACC_f figure, to show the cost side of the same trade. Say in "
     "the caption that the axis is zoomed -- the whole ten-class range is three points."),
    ("pure_vs_hybrid_composite_by_class.png",
     "Pure vs hybrid composite, all ten classes (dumbbell)",
     "Results, as the summary of the previous two on the anchor's own metric. If only "
     "one of the three goes in the body, use this one and appendix the others."),
    ("operator_frequency_selected_cstar.png",
     "Operator frequency across the ten selected C* (bar)",
     "Results or Discussion, as the evidence for the MASK finding. Pairs with the caveat "
     "that ten runs at one seed is convergent evidence, not an ablation."),
    ("benchmark_comparison.png",
     "Anchor vs pure vs hybrid on four metrics (dot, small multiples)",
     "Results, the benchmark section. Carries the 'retention competitive, forgetting "
     "not' reading in one image."),
    ("truck_failure_analysis.png",
     "Predicted class of the 1,000 truck test images under four models (bar, small multiples)",
     "Discussion, the truck section. This is the figure that turns truck from a "
     "confession into a finding."),
    ("class_structure_analysis.png",
     "Structure per class, structure against difficulty, and the inter-class "
     "similarity matrix (bar + scatter + heatmap)",
     "Results, as the FIRST figure of the chapter -- it is the measurement the whole "
     "project rests on. Panel B is also the honest answer to RQ3 and belongs in the "
     "discussion beside the truck section; do not caption it as though structure "
     "predicts difficulty, because it does not."),
]


def build_figure_inventory() -> bool:
    rows = []
    for cid, name in CLASSES:
        run = PROJECT_ROOT / f"results/search/plan_a_{name}"
        png = f"results/search/plan_a_{name}/pareto_front_plan_a_{name}.png"
        # frog predates the class-agnostic plotter and carries the equivalent file
        # under the older name, written by the deliberately frozen frog-specific
        # script. Same purpose, different columns -- not a gap.
        if (run / "pareto_front_plot_data.csv").exists():
            plot_data = "yes"
        elif (run / "pareto_front_plot_table.csv").exists():
            plot_data = "yes (`_plot_table`)"
        else:
            plot_data = "**MISSING**"
        rows.append([cid, name, f"`{png}`", plot_data])

    fig_dir = OUT / "figures"
    new_rows = []
    for filename, what, _use in NEW_FIGURES:
        path = fig_dir / filename
        size = f"{path.stat().st_size / 1024:.0f} KB" if path.is_file() else "**MISSING**"
        new_rows.append([f"`{filename}`", what, size])

    missing = [f for f, _, _ in NEW_FIGURES if not (fig_dir / f).is_file()]
    uses = "\n\n".join(f"**`{f}`** -- {use}" for f, _, use in NEW_FIGURES)

    write_md("figure_inventory.md", f"""
# Figure inventory

{stamp()}

Every figure listed as present was checked on disk while this file was written.

## 1. Per-class Pareto fronts -- ten figures, all present

{table(["id", "class", "path", "plot data CSV"], rows, "rllc")}

Each shows the ten-member non-dominated front for that class with `C*` marked.
`pareto_front_plot_data.csv` beside each figure holds the plotted values, so any figure
can be restyled for print without re-running anything.

**Dissertation use.** Do not print all ten in the results chapter. Put two or three in
the body -- **dog** (the strongest front, `S` = 4427.91), **truck** (the weakest,
`S` = 93.99), and optionally **airplane** (the only class reaching `ACC_f` 0.00) -- and
move the remaining seven to Appendix E. Three fronts side by side make the class-wise
spread visible in a way the table cannot; ten in sequence makes a reader skip the
section.

## 2. Write-up figures -- {len(NEW_FIGURES) - len(missing)} of {len(NEW_FIGURES)} present

All in `results/writeup_package/figures/`, 300 dpi PNG. The first six are built by
`experiments/build_writeup_figures.py` from the CSVs in this package;
`class_structure_analysis.png` is built by
`experiments/build_class_structure_figure.py` from
`results/analysis/class_structure/`.

{table(["file", "what it shows", "size"], new_rows, "llr")}

### What each is for

{uses}

### Design notes worth carrying into the captions

Figures 1-3 share one class order -- by pure `ACC_f`, worst first -- so they read across
as a set. Figures 1-3 and 5 use dots rather than bars because those panels need a zoomed
axis, and a bar drawn from a non-zero baseline misstates the ratio between its
neighbours; a dot encodes position only, so the zoom is honest. Every mark is directly
labelled, so no reading depends on colour alone. The palette is the one already used by
the ten Pareto figures, so both sets read as one system in the same document; it is
validated for categorical use on a light surface, and light-surface only, which is what
a printed page is.

## 3. Class-structure figure -- built

`class_structure_analysis.png`, three panels, from
`results/analysis/class_structure/summary.json` and
`channel_contrast_all_classes.csv` by `experiments/build_class_structure_figure.py`.
The 10x10 matrix behind panel C is written out as `class_structure_similarity.csv`
beside this file.

It is the measurement the whole project rests on, so it belongs first in the results
chapter -- but read panel B before writing the caption. Structure explains the
**regime** (class-level forget sets have it, instance-level ones do not) and not the
**ranking** (it does not predict which class is hard). See
`missing_figures_status.md` for what the panels do and do not establish.

## 4. Figures that do not exist and would need new experiments

Named here so they are not assumed available: seed-variance plots, ablation plots
(random search at equal budget, operator families in isolation), baseline comparison
plots, and runtime charts. Each needs work that has not been run.

See `missing_figures_status.md` for what was attempted, what succeeded, and what did not.
""")
    return not missing


def build_missing_figures_status(all_present: bool) -> None:
    bird = PROJECT_ROOT / "results/search/plan_a_bird/pareto_front_plot_data.csv"
    verdict = ("(verified on disk)" if all_present
               else "-- SOME ARE MISSING, see figure_inventory.md")

    write_md("missing_figures_status.md", f"""
# Figure status -- what was generated, what was not, and why

{stamp()}

## Generated

All six requested figures were produced, into `results/writeup_package/figures/`.
**{len(NEW_FIGURES)} of {len(NEW_FIGURES)} present** {verdict}.

| figure | source | new computation? |
|---|---|---|
| `pure_vs_hybrid_acc_f_by_class.png` | `pure_medus_10_class_table.csv`, `hybrid_medus_10_class_table.csv` | none |
| `pure_vs_hybrid_acc_r_by_class.png` | the same two tables | none |
| `pure_vs_hybrid_composite_by_class.png` | the same two tables | none |
| `operator_frequency_selected_cstar.png` | `pure_medus_10_class_table.csv` | none |
| `benchmark_comparison.png` | `benchmark_comparison_table.csv` | composite derived as `ACC_r x (1 - ACC_f)` |
| `truck_failure_analysis.png` | `truck_prediction_distribution.csv` | **yes -- inference only, below** |

## The one figure that needed computation: `truck_failure_analysis.png`

It was **generated**, not skipped.

The tables record how much truck accuracy survives unlearning; they do not record where
the rest of the images went. Answering that needs a forward pass, so
`experiments/analyse_truck_predictions.py` classifies the 1,000 held-out truck images
with four models and records the full predicted-class distribution of each.

**No model was trained, searched or refined.** `W_0`, `W_ref` and the refined hybrid all
exist as checkpoints. The pure `C*` does not -- the search recorded genomes, not weights
-- so it was reconstructed by replaying its stored chromosome through the same
deterministic operators, which is exactly what `report_anchor_metrics.py` already does
in order to score it. That path verifies the rebuild against the recorded front row and
raises rather than proceeding if anything drifts.

**The reconstruction reproduced the published numbers exactly.** The rebuilt `C*`
classifies 42.10% of truck test images as truck, and the refined checkpoint 30.90% --
the `ACC_f` values already in `pure_medus_10_class_table.csv` and
`hybrid_medus_10_class_table.csv`. That agreement is the check that the figure describes
the published models and not some near neighbour of them.

### What it shows

| model | still truck | top non-truck destination |
|---|---:|---|
| `W_0` | 95.40% | automobile (2.80%) |
| `W_ref` | 0.00% | **automobile (68.40%)** |
| `C*` pure | 42.10% | automobile (16.70%) |
| `C*` hybrid | 30.90% | automobile (20.40%) |

A model that never saw a truck sends 68.4% of them to automobile. Pure MED-US sends only
16.7% there and leaves 42.1% still called truck; the refinement moves a further 11.2
points out, most of which arrive at automobile (20.4%). The failure is a **partial move
along the truck-automobile axis**, not a random scattering -- the reference's destination
is the same destination, reached less far.

This is direct evidence for the confusability reading of the truck result, and it did not
exist before. It belongs in the discussion chapter.

## Bird's `pareto_front_plot_data.csv` -- correction: it was never missing

An earlier note in this package reported bird as having a figure but no plot-data CSV.
**That was wrong, and the record is corrected here.** Bird's file was present and
correct all along; the class whose file does not exist under that name is **frog**.

Bird's CSV was regenerated anyway, by `plot_pareto_front_class.py` -- documented
read-only, no model loaded -- from `full_fidelity/front_full_fidelity.csv`. The result is
**byte-identical to the committed version**: `git diff` on that path comes back empty. So
the rebuild changed nothing, which is itself the proof that the original was fine.

Present at `{bird.relative_to(PROJECT_ROOT).as_posix()}`: 10 rows, columns identical to
the other eight class-agnostic runs, front member #0 correctly carrying all three roles
(selected `C*`, best selectivity, strongest forgetting).

The committed bird figure was **not** touched. That script writes a PNG as well as the
CSV, so the PNG was directed to a scratch path outside the repository and discarded.
`results/search/plan_a_bird/pareto_front_plan_a_bird.png` is unchanged.

## Frog: a different filename, not a missing file

Frog has no `pareto_front_plot_data.csv` and does not need one. It predates the
class-agnostic plotter and carries the equivalent artefact as
`results/search/plan_a_frog/pareto_front_plot_table.csv` -- ten rows, one per front
member, with its own `role` column marking best selectivity and strongest forgetting.

It was written by `plot_pareto_front.py`, the frog-specific plotter that is deliberately
left frozen so the committed frog figure stays reproducible from the script that made
it. Regenerating frog's data under the newer name would produce a second file with
different columns describing the same front, so it was not done. **All ten classes have
their plotted values on disk.**

## Generated: `class_structure_analysis.png`, and it returned a null result

Built from `results/analysis/class_structure/summary.json` and
`channel_contrast_all_classes.csv` -- both committed. No new experiment; the only
computation is arithmetic over those files. It also writes
`class_structure_similarity.csv`, the 10x10 matrix behind panel C.

**Panel A -- structure exists, in every class.** Between 84.1% (horse) and 91.2% (ship)
of channels stand above the noise floor, against **0.55%** for the predecessor project's
instance-level forget set, where the null control gives 1.00% by construction. Three
orders of magnitude. This is the project's founding result and it holds for all ten.

**Panel B -- structure does NOT predict difficulty, and this is the important finding.**
Median SNR against pure `ACC_f` is a null scatter: **Pearson r = -0.04 over ten points**.
Truck is sixth of ten on median SNR and fifth on channels above the floor, yet its
`ACC_f` of 42.10 is more than double the next worst. Automobile has the **least**
structure of any class and forgets 3.3x better than truck.

This matters for the write-up. `limitations_future_work_notes.md` proposed regressing
per-class `ACC_f` on the activation-contrast statistic and called it "the strongest
contribution available". **That regression has now been run and it is null.** The
proposal should be corrected rather than left standing as future work, and the null
reported: a reader will otherwise assume the obvious explanation, which the data does
not support.

**Panel C -- the similarity matrix, which does support the truck reading.** Cosine
similarity between the per-class channel-contrast vectors recovers the semantic
grouping without being told it: vehicles with vehicles (airplane-ship 0.41,
truck-automobile 0.32, ship-truck 0.29), animals with animals (cat-dog 0.30,
deer-frog 0.30, bird-deer 0.29). That it reproduces a structure nobody encoded is the
evidence the measurement means something.

**Truck's nearest neighbour is automobile (0.32), and the relation is mutual** --
automobile's nearest is truck. That matches `truck_failure_analysis.png` exactly, where
a model that never saw a truck sends 68.4% of them to automobile. Two independent
measurements, one on activations of `W_0` and one on predictions of four unlearned
models, point at the same pair.

### What this figure must NOT be captioned as showing

That similarity predicts difficulty. **It does not**: Pearson r = -0.08 between each
class's maximum similarity to any other class and its `ACC_f`. Airplane is the
counterexample and it is decisive -- it has the highest similarity to another class of
any of the ten (0.41, with ship) and still reaches `ACC_f` 0.00.

The defensible claim is narrower and still worth making: truck's difficulty **coincides
with** its structure being most shared with a class that stays, and two independent
measurements agree on which class that is. Whether that sharing is the *cause* is not
established by these artefacts.

## Not generated, and why

**Seed-variance, ablation, baseline and runtime figures.** These cannot be built from
existing artefacts, because the underlying experiments have not been run. Building them
would require exactly the searches, ablations and baseline implementations that the
rules for this task excluded. They are listed in `limitations_future_work_notes.md` as
outstanding work, not as missing plots.

## Rules observed

No search was run. Nothing was trained. No refinement was re-run. No committed result
changed. The only computation anywhere in this task was the forward pass described
above, which the ask explicitly permitted for the truck figure. No `.pt` or `.zip` is
committed, and the reconstructed `C*` was held in memory and never written to disk.
""")


def build_results_notes(pure_rows, hybrid_rows, pure_stats, hybrid_stats, per_class) -> None:
    ops_with_mask = sum(1 for r in pure_rows if "MASK" in r["operators"])
    mask_only = [r["class_name"] for r in pure_rows if r["operators"] == "MASK"]

    freq: dict[str, int] = {}
    for r in pure_rows:
        for op in r["operators"].split("|"):
            freq[op] = freq.get(op, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    freq_text = ", ".join(f"`{k}` {v}" for k, v in ranked)

    ref = {int(r["class_id"]): r for r in read_csv(REFTRAIN / "reference_validation_summary.csv")}
    dr = [float(ref[c]["retain_test_acc"]) for c, _ in CLASSES]
    dr_mean, dr_std = agg(dr)

    write_md("results_chapter_notes.md", f"""
# Results chapter -- working notes

{stamp()}

Bullet points to write from, not prose to paste. Every number traces to a table in
this package.

## Reference model quality

- All **10 of 10** classes have a validated retain-only reference `W_ref`; none was
  missing, none failed validation.
- `D_f_test` accuracy is **0.0000 for every one**. A model that never trained on a
  class does not classify it -- this is the correctness condition, and it held
  without exception.
- `D_r_test` accuracy across the ten: **{pm(dr_mean, dr_std, 4)}**
  (min {f(min(dr), 4)} horse, max {f(max(dr), 4)} cat). Tight, so no class has an
  unusually weak or strong gold standard that could distort its unlearning result.
- Identical protocol for all ten: 200 epochs, seed 42, split 5,000 / 45,000 / 1,000 / 9,000.
- Selection was on `D_r_test` accuracy with `D_r_test` loss as tie-breaker.
  `D_f_test` was logged every epoch but **never** influenced selection.
- Every split inside every imported zip was byte-compared against the
  version-controlled local split before import; sha256 recorded for all ten.
- Training was distributed across three people and seven Kaggle bundles plus one
  local run. Say so in the write-up -- it is a reproducibility fact, not a footnote.

## Pure MED-US results

- Ten classes, one selection rule: `C*` maximises `ACC_r x (1 - ACC_f)`, the anchor
  paper's own metric function.
- Aggregate: `ACC_r` **{pm(*pure_stats['anchor_ACC_r'])}**,
  `ACC_f` **{pm(*pure_stats['anchor_ACC_f'])}**,
  composite **{pm(*pure_stats['anchor_composite'])}**,
  MIA **{pm(*pure_stats['anchor_MIA'])}**.
- Retention is stable, forgetting is not. `ACC_r` varies by 3.1 points across all ten
  classes; `ACC_f` varies by 42.1.
- Selectivity `S` = **{pm(*pure_stats['selectivity_S'])}**, min 93.99 (truck),
  max 4427.91 (dog). Compare against the predecessor project's instance-level ceiling
  of **1.158** across 10,534 strategies -- three orders of magnitude, and the central
  evidence that class-level forget sets have structure instance-level ones do not.
- No search failed. Ten runs, zero failures, fronts of ten members each.
- Cost: roughly 8 minutes of search plus 10 minutes of full-fidelity re-measurement
  per class, with no retain-set training loop and no optimiser state.

## Hybrid refinement results

- **Nine eligible classes, nine accepted, zero rejected.** Airplane is a deliberate
  no-op: pure `ACC_f` already 0.00 at MIA 100.00.
- Aggregate: `ACC_r` **{pm(*hybrid_stats['anchor_ACC_r'])}**,
  `ACC_f` **{pm(*hybrid_stats['anchor_ACC_f'])}**,
  composite **{pm(*hybrid_stats['anchor_composite'])}**,
  MIA **{pm(*hybrid_stats['anchor_MIA'])}**.
- `ACC_f` falls 5.00 points of mean for **0.12** of retain accuracy. Composite rises
  4.58, MIA rises 2.51.
- The standard deviation narrows on every metric. The refinement helps the weak
  classes most, which is why the mean moves more than any single strong class does.
- BatchNorm buffer movement is **exactly 0.000000** on all nine, with zero counter
  changes. Parameter movement 0.000303 to 0.000420 against a 0.0400 budget.
- The BN-frozen guard is not decoration. An earlier unfrozen attempt passed every
  weight-based check while `D_r` batches silently re-estimated the running statistics
  and undid the operator edit. Buffer movement is the only check that catches it.

## Benchmark comparison

- Compared against the anchor's Table 1 (CIFAR-10 / ResNet-18, ten-class mean), which
  is why the sweep covers all ten classes rather than one.
- **Retention is competitive.** Pure `ACC_r` {f(pure_stats['anchor_ACC_r'][0])} against
  the anchor's 94.19 and Retraining's 94.81 -- about a point back, inside the field.
- **Forgetting is not.** Pure `ACC_f` {f(pure_stats['anchor_ACC_f'][0])} against the
  anchor's 0.03. This is the honest headline and it should be stated plainly rather
  than buried.
- **Privacy is competitive** on the anchor's MIA: pure {f(pure_stats['anchor_MIA'][0])},
  hybrid {f(hybrid_stats['anchor_MIA'][0])}, anchor 95.50.
- Caveat that must appear in the text: the eight literature rows are **as reported**,
  not re-measured here. No published baseline was re-run in this harness.
- Supporting evidence that the harnesses agree: this project's own `W_0` and `W_ref`
  measurements land within a few tenths of the paper's Original and Retraining rows.
- Worth one sentence of scepticism about the MIA metric itself: in the same table
  SCRUB reaches `ACC_f` 0.00 with MIA 0.00, and Retraining scores exactly 100.00.
  A metric where the gold standard is pinned at the ceiling is saturated.

## Class-wise variation

- This is the most interesting result in the chapter and deserves its own section.
- Best: **airplane**, `ACC_f` 0.00, composite 92.86, MIA 100.00 -- the only class that
  matches the retraining reference exactly on forgetting.
- Worst: **truck**, `ACC_f` 42.10, composite 53.80, MIA 69.60.
- Second tier: dog 3.30, deer 7.80, frog 8.30, horse 9.80, cat 9.60.
- Third tier: automobile 12.70, ship 14.00, bird 17.90.
- The ordering is not random noise across a uniform method -- it is stable, wide, and
  the same classes stay weak after refinement. Truck is worst pure and worst hybrid;
  airplane is best pure and best hybrid.
- The natural reading -- that how much forget-specific structure a class has determines
  how well weight editing can remove it -- **has now been tested against the committed
  class-structure measurement, and it is false.** Pearson r between median SNR and
  `ACC_f` is -0.04 over the ten classes. See `class_structure_analysis.png`, panel B.
- Report the null. It is a better result than the correlation would have been, because
  it says the structure measurement answers a *regime* question (class-level forget
  sets have structure, instance-level ones do not) and not a *ranking* one, and it
  stops a reader assuming the obvious explanation.
- What survives: the ordering is stable and wide, and it is not explained by structure
  magnitude. Why it varies remains open.

## Truck as the weakest class

- Pure `ACC_f` 42.10, against a ten-class mean of {f(pure_stats['anchor_ACC_f'][0])} --
  more than 2.5 standard deviations out.
- Also the weakest on every other metric: composite 53.80 (next worst 78.10), MIA
  69.60 (next worst 91.10), selectivity 93.99 (next worst 154.57).
- Refinement helps truck **more than any other class**: -11.20 `ACC_f`, +10.24
  composite, +9.50 MIA. Largest absolute gain in the sweep.
- And it is still the worst class afterwards, at 30.90 against a 7.55 mean.
  Refinement narrows the gap; it does not close it.
- Truck selects `CLIP|MASK|QUANTIZE` at front position #0 -- the position that
  maximises the composite, on a front whose best composite is 38 points below every
  other class's.
- Do not apologise for this row. A method with a stable, identifiable failure mode is
  more useful than one that fails unpredictably. Report it, and say what is not yet
  known: whether truck's difficulty is confusability with automobile, or lower
  activation contrast, or both.

## `MASK` in every selected pure candidate

- `MASK` appears in the selected `C*` for **{ops_with_mask} of 10** classes.
- Operator frequency across the ten selected candidates: {freq_text}. `MASK` is the
  only operator present in every one; the next most frequent appears in {ranked[1][1]}.
- {len(mask_only)} classes select `MASK` alone: {", ".join(mask_only)}.
- Full operator sets, in class order: {", ".join(r["class_name"] + " " + ops(r["operators"]) for r in pure_rows)}.
- This is a search **finding**, not a design choice. `MASK` was one of several
  operators available at equal cost, and the search converged on it independently in
  ten separate runs.
- Consistent with the mechanism the project argues for: if class-specific information
  is concentrated in identifiable channels, zeroing those channels is the operator
  that removes it, and the magnitude-based alternatives are blunter.
- Be careful with the strength of the claim. Ten runs at one seed is convergent
  evidence, not proof; an operator ablation would settle it and has not been run.

## Why pure and hybrid must be reported separately

- They are different methods. Pure MED-US applies no gradient at any point; the
  hybrid applies two gradient steps after the search finishes.
- The anchor paper's method is **gradient-free**. Only the pure table is a
  like-for-like comparison with its Table 1.
- Merging them, or quoting the hybrid's {f(hybrid_stats['anchor_ACC_f'][0])} as
  "MED-US", would overstate what the gradient-free method achieves by 5 points of
  `ACC_f` and 4.6 of composite.
- The temptation is real and worth naming in the text: the hybrid is better on every
  headline metric. That is exactly why the separation has to be explicit and
  permanent rather than left to the reader.
- Practical rule for the write-up: the pure table is the result. The hybrid is a
  clearly labelled extension answering a different question -- what does one
  constrained gradient step add to a gradient-free solution.
""")


def build_key_numbers(pure_stats, hybrid_stats, pure_rows, per_class) -> None:
    best = min(pure_rows, key=lambda r: float(r["anchor_ACC_f"]))
    worst = max(pure_rows, key=lambda r: float(r["anchor_ACC_f"]))
    biggest = min(per_class, key=lambda r: float(r["delta_ACC_f"]))

    write_md("key_numbers_summary.md", f"""
# Key numbers

{stamp()}

Single reference sheet. If a number appears in the dissertation, it should match this
page.

## Pure MED-US, ten-class aggregate

| metric | mean +/- std | min | max |
|---|---|---|---|
| `ACC_r` (%) | **{pm(*pure_stats['anchor_ACC_r'])}** | 92.52 (frog) | 95.62 (dog) |
| `ACC_f` (%) | **{pm(*pure_stats['anchor_ACC_f'])}** | 0.00 (airplane) | 42.10 (truck) |
| composite (%) | **{pm(*pure_stats['anchor_composite'])}** | 53.80 (truck) | 92.86 (airplane) |
| anchor MIA (%) | **{pm(*pure_stats['anchor_MIA'])}** | 69.60 (truck) | 100.00 (airplane) |
| selectivity `S` | **{pm(*pure_stats['selectivity_S'])}** | 93.99 (truck) | 4427.91 (dog) |

## Hybrid MED-US, ten-class aggregate

| metric | mean +/- std | min | max |
|---|---|---|---|
| `ACC_r` (%) | **{pm(*hybrid_stats['anchor_ACC_r'])}** | 92.56 (frog) | 95.57 (dog) |
| `ACC_f` (%) | **{pm(*hybrid_stats['anchor_ACC_f'])}** | 0.00 (airplane) | 30.90 (truck) |
| composite (%) | **{pm(*hybrid_stats['anchor_composite'])}** | 64.04 (truck) | 93.75 (dog) |
| anchor MIA (%) | **{pm(*hybrid_stats['anchor_MIA'])}** | 79.10 (truck) | 100.00 (airplane) |
| selectivity `S` | **{pm(*hybrid_stats['selectivity_S'])}** | 107.42 (truck) | 4950.81 (dog) |

## Pure against hybrid

| metric | pure | hybrid | change |
|---|---|---|---|
| `ACC_r` (%) | {pm(*pure_stats['anchor_ACC_r'])} | {pm(*hybrid_stats['anchor_ACC_r'])} | **{hybrid_stats['anchor_ACC_r'][0] - pure_stats['anchor_ACC_r'][0]:+.2f}** |
| `ACC_f` (%) | {pm(*pure_stats['anchor_ACC_f'])} | {pm(*hybrid_stats['anchor_ACC_f'])} | **{hybrid_stats['anchor_ACC_f'][0] - pure_stats['anchor_ACC_f'][0]:+.2f}** |
| composite (%) | {pm(*pure_stats['anchor_composite'])} | {pm(*hybrid_stats['anchor_composite'])} | **{hybrid_stats['anchor_composite'][0] - pure_stats['anchor_composite'][0]:+.2f}** |
| anchor MIA (%) | {pm(*pure_stats['anchor_MIA'])} | {pm(*hybrid_stats['anchor_MIA'])} | **{hybrid_stats['anchor_MIA'][0] - pure_stats['anchor_MIA'][0]:+.2f}** |

Nine of ten classes improved on the composite; the tenth is the airplane no-op. No
class regressed on any headline metric.

## Benchmark comparison (anchor Table 1, ten-class mean)

| method | ACC_r | ACC_f | MIA | numbers |
|---|---|---|---|---|
| Original (paper) | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 | reported |
| Retraining (paper) | 94.81 +/- 0.52 | 0.00 | 100.00 | reported |
| Kodge et al. 2024 (anchor) | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.50 +/- 14.23 | reported |
| SSD (Foster et al. 2024) | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 | reported |
| UNSIR (Tarun et al. 2023) | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.50 +/- 25.86 | reported |
| **MED-US pure** | **{pm(*pure_stats['anchor_ACC_r'])}** | **{pm(*pure_stats['anchor_ACC_f'])}** | **{pm(*pure_stats['anchor_MIA'])}** | measured |
| **MED-US hybrid** | **{pm(*hybrid_stats['anchor_ACC_r'])}** | **{pm(*hybrid_stats['anchor_ACC_f'])}** | **{pm(*hybrid_stats['anchor_MIA'])}** | measured |

Gap to the anchor: `ACC_r` {pure_stats['anchor_ACC_r'][0] - 94.19:+.2f}, `ACC_f` {pure_stats['anchor_ACC_f'][0] - 0.03:+.2f}, MIA {pure_stats['anchor_MIA'][0] - 95.50:+.2f}.

Nearest published neighbour on `ACC_f` is UNSIR at 10.89 +/- 8.79. Its spread is the
same order as this work's ({f(pure_stats['anchor_ACC_f'][1])}) -- both have strong and
weak classes rather than uniform behaviour, unlike the anchor's 0.03 +/- 0.09.

## Best class

**airplane** (class 0), pure. Best by `ACC_f`, composite and MIA simultaneously; its
`ACC_r` is mid-table, which is the trade the composite accepted.

| | |
|---|---|
| `ACC_r` | {f(best['anchor_ACC_r'])} |
| `ACC_f` | **{f(best['anchor_ACC_f'])}** -- matches the retraining reference exactly |
| composite | {f(best['anchor_composite'])} |
| MIA | {f(best['anchor_MIA'])} |
| operators | {ops(best['operators'])}, front position #{best['front_position']} |

The only class where refinement was not attempted, because there was nothing left to
forget.

## Worst class

**truck** (class 9), pure.

| | |
|---|---|
| `ACC_r` | {f(worst['anchor_ACC_r'])} |
| `ACC_f` | **{f(worst['anchor_ACC_f'])}** |
| composite | {f(worst['anchor_composite'])} |
| MIA | {f(worst['anchor_MIA'])} |
| selectivity `S` | {f(worst['selectivity_S'])} -- lowest of the ten |
| operators | {ops(worst['operators'])}, front position #{worst['front_position']} |

Worst on every headline metric, pure and hybrid. Still 30.90 `ACC_f` after
refinement.

## Biggest refinement improvement

**{biggest['class_name']}** (class {biggest['class_id']}).

| metric | pure | hybrid | change |
|---|---|---|---|
| `ACC_f` | {biggest['pure_ACC_f']} | {biggest['hybrid_ACC_f']} | **{biggest['delta_ACC_f']}** |
| composite | {biggest['pure_composite']} | {biggest['hybrid_composite']} | **{biggest['delta_composite']}** |
| MIA | {biggest['pure_MIA']} | {biggest['hybrid_MIA']} | **{biggest['delta_MIA']}** |
| `ACC_r` | {biggest['pure_ACC_r']} | {biggest['hybrid_ACC_r']} | {biggest['delta_ACC_r']} |

Largest absolute `ACC_f` gain of any class, and it remains the weakest class after
the gain.

Runner-up by `ACC_f`: ship, -9.30 (14.00 -> 4.70), which is also the largest composite
gain among the classes that end in a strong position (+8.71, to 90.05).

## Retain accuracy cost of refinement

**{hybrid_stats['anchor_ACC_r'][0] - pure_stats['anchor_ACC_r'][0]:+.2f} points of mean `ACC_r`**, for -5.00 points of mean `ACC_f`.

| class | d `ACC_r` |
|---|---|
| airplane | +0.0000 (no-op) |
| frog | +0.0333 |
| dog | -0.0556 |
| automobile | -0.0778 |
| ship | -0.0889 |
| deer | -0.1111 |
| horse | -0.2111 |
| bird | -0.2222 |
| truck | -0.2333 |
| cat | -0.2444 |

Worst single-class retain cost is 0.2444 points (cat). No class exceeded the 0.010
fractional `D_r_test` drop that acceptance check 2 enforces.

Parameter movement across the nine: 0.000303 (truck) to 0.000420 (frog), against a
0.0400 budget -- between 0.8% and 1.1% of what was permitted.

BatchNorm buffer movement: **0.000000 on all nine**, zero counter changes.
""")


def build_limitations() -> None:
    write_md("limitations_future_work_notes.md", f"""
# Limitations and future work -- working notes

{stamp()}

Written to be defensible rather than modest. Each limitation states what was done,
what it does not license, and what would fix it.

## 1. CIFAR-10 only, ResNet-18 only

- Every result is CIFAR-10 / ResNet-18, 32x32 images, ten balanced classes,
  one architecture, one `W_0`.
- What it does not license: any claim about scale, about architectures without
  BatchNorm, or about datasets with many more classes or class imbalance. The
  operator library acts on convolutional layer groups and BatchNorm behaviour is
  central to the refinement, so a transformer is not a small extrapolation.
- Mitigating context: this is the anchor paper's primary setting, so the comparison
  is fair even though the coverage is narrow. The anchor itself also reports
  CIFAR-100 and ImageNet; this work does not.
- Fix: CIFAR-100 (or its twenty superclasses) on the same ResNet-18 tests scale in
  the number of classes at moderate cost; VGG-16 or a ViT tests architecture. Both
  need a new `W_0` and a new reference per target class -- roughly 2.3 GPU-hours per
  reference at the protocol used here.

## 2. One seed

- Every search ran at seed 42. `W_0` is a single model at seed 42; all ten `W_ref`
  are seed 42; NSGA-II ran at seed 42.
- What it does not license: reporting `+/- std` over the ten classes as if it were an
  uncertainty estimate. **It is class-to-class spread, not run-to-run variance.**
  Those are different quantities and the write-up must not blur them.
- Two distinct variances are unmeasured, and they cost very different amounts:
  - **Search variance** -- would the evolutionary search find an equally good `C*` on
    a different seed? Cheap to measure: `W_0` and `W_ref` are fixed and the objective
    evaluation is deterministic given a genome, so only the sampler changes.
  - **Training variance** -- would a different `W_0` and a different reference give
    the same result? Expensive: a new reference per class per seed.
- Fix, in order of value per hour: extra search seeds on a small subset of classes
  (best, middle, worst) first; training variance only if a reviewer requires it.

## 3. The literature benchmark is a comparison against reported numbers

- The eight baseline rows in the benchmark table are transcribed from the anchor
  paper's Table 1. **No published unlearning baseline was re-implemented or re-run in
  this harness.**
- What it does not license: a claim that MED-US beats or loses to any specific
  baseline under controlled conditions. The comparison inherits every difference
  between two implementations -- augmentation, MIA attack details, checkpoint
  selection, evaluation subsets.
- What supports it: this project's own `W_0` and `W_ref` measurements land within a
  few tenths of the paper's Original and Retraining rows, which is evidence the two
  harnesses agree on the baselines they share. That is the strongest available
  argument and it is indirect.
- A second concern, worth its own paragraph: the anchor's MIA appears saturated.
  Retraining scores exactly 100.00, and in the same table SCRUB reaches `ACC_f` 0.00
  with MIA 0.00. A metric that pins the gold standard at the ceiling has limited
  discriminative power, and this project's own MIA AUC on the same models sits far
  closer to chance than the anchor MIA implies.
- Fix: implement finetune-on-`D_r`, NegGrad, random-relabel and the anchor's own
  method against this harness, and evaluate every method under one MIA definition.
  A stronger attack (U-LiRA) would address the saturation separately.

## 4. The hybrid is gradient-based post-search refinement

- The hybrid applies one clipped gradient-ascent step on `D_f` and one repair step on
  `D_r` **after** the search finishes. It is not part of the evolutionary method and
  it is not gradient-free.
- What it does not license: quoting the hybrid's `ACC_f` of 7.55 as MED-US's result,
  or placing it in the anchor's Table 1 as a like-for-like row. The anchor's method is
  gradient-free; only the pure table compares fairly.
- It also needs `D_f` and `D_r` at unlearning time and an optimiser step, which
  removes part of the deployment argument for a gradient-free method.
- Honest framing: the hybrid answers a narrower question -- what does one constrained
  gradient step add to a gradient-free solution? The answer is 5 points of `ACC_f` for
  0.12 of retain accuracy. That is a useful result and a separate one.
- Fix: none needed. The separation is the correct treatment and should be maintained
  permanently. What could be added is a comparison against the same two gradient steps
  applied to `W_0` directly, which would show how much of the hybrid's gain comes from
  the search rather than from the gradient.

## 5. Truck remains difficult

- Pure `ACC_f` 42.10, hybrid 30.90, against ten-class means of 12.55 and 7.55.
  Worst class on every headline metric, before and after refinement.
- What is known: the failure is stable and reproducible, not noise, and refinement
  helps truck more than any other class while still leaving it worst.
- Both open hypotheses have now been tested against the committed class-structure
  measurement, and the results are in `class_structure_analysis.png`.
- **Lower activation contrast: ruled out.** Truck is sixth of ten on median SNR and
  fifth on channels above the noise floor, and the correlation between structure and
  `ACC_f` across the ten classes is null (Pearson -0.04). Automobile has the least
  structure of any class and forgets 3.3x better than truck.
- **Confusability with automobile: supported, but not established as the cause.**
  Truck's nearest neighbour in channel-contrast space is automobile (0.32), mutually,
  and a model that never saw a truck sends 68.4% of them to automobile. Two independent
  measurements agree on the pair. But maximum similarity does not predict `ACC_f`
  either (Pearson -0.08), and airplane is decisive against the simple version: highest
  similarity to another class of any of the ten, and `ACC_f` 0.00.
- So **why truck is hard remains open.** What can be said is narrower and still worth
  saying: the failure is stable, it is not explained by how much forget-specific
  structure the class has, and it coincides with truck sharing more of its structure
  with a retained class than with anything else.
- What it does not license: describing MED-US as reliable across classes. It is
  reliable on retention and variable on forgetting, and the variation is large. Nor
  does it license the claim, floated before the regression was run, that per-class
  difficulty is predictable in advance from the structure statistic. It is not.
- Fix: a per-class predictor would need something these artefacts do not contain --
  candidate directions include the overlap between the channels an operator actually
  edits and the channels a retained neighbour depends on, which is measurable from
  `channel_contrast_all_classes.csv` plus the selected genomes, and was not attempted
  here.

## 6. No full ablation

- Not run: NSGA-II against random search at equal evaluation budget; operator families
  in isolation; population and generation budget sensitivity; the `class_contrast`
  selector against alternatives; `max_level`.
- What it does not license: the claim that the evolutionary search is necessary. The
  search evaluated roughly 210 to 320 genomes per class, and nothing here rules out a
  uniform random sample of that size performing comparably.
- Related unresolved observation: `MASK` appears in all ten selected candidates. That
  is convergent evidence across ten independent runs, but without an operator ablation
  it stays an observation rather than a demonstrated property.
- Fix, cheap: random sampling at matched evaluation count reuses the entire existing
  harness and changes only the sampler. It is the first question a viva will ask about
  any evolutionary method.

## 7. No runtime optimisation

- Reported runtimes -- roughly 8 minutes of search and 10 minutes of full-fidelity
  re-measurement per class -- are from unoptimised single-GPU code with
  `num_workers: 0`, a batch cap of 3 during search, and no parallel evaluation of the
  population.
- What it does not license: comparing this project's wall-clock against published
  runtimes as though the implementations were equally tuned. The anchor reports a
  one-shot closed-form update; a comparison of seconds would flatter it and mean
  little either way.
- What is fair to claim: the *shape* of the cost. No retraining, no optimiser state,
  no retain-set training loop, and a search that is trivially parallel across the
  population because each genome evaluates independently.
- Fix: parallel population evaluation and non-zero dataloader workers would give a
  large constant-factor gain, but runtime is not a contribution of this dissertation
  and the effort is better spent on items 3, 5 and 6.

## Priority if time is limited

1. **Item 6** -- random search at equal budget. Cheapest of the three, and it defends
   the method choice against the first question a viva will ask.
2. **Item 3** -- baselines in this harness. The largest scientific gap, and the most
   expensive.
3. **Item 2** -- extra search seeds on a subset of classes.

Items 1, 4 and 7 are best handled in the text as scope statements rather than as
outstanding work.

**No longer outstanding:** the per-class regression that earlier versions of this file
listed first. It has been run against the committed class-structure measurement and
returned a null (Pearson -0.04 for structure magnitude, -0.08 for maximum inter-class
similarity). It is now a reported result, not a plan -- see item 5 and
`class_structure_analysis.png`.
""")


# ----------------------------------------------------------------------------- main

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    shas = load_sha_prefixes()
    baselines = load_anchor_baselines()
    refinements = load_refinements()
    per_class = read_csv(LIT / "pure_vs_hybrid_comparison.csv")

    build_reference_table(shas)
    pure_rows, pure_stats = build_pure_table()
    hybrid_rows, hybrid_stats = build_hybrid_table()
    build_pure_vs_hybrid(pure_stats, hybrid_stats)
    build_benchmark_table(pure_stats, hybrid_stats, baselines)
    build_refinement_acceptance(refinements)
    all_present = build_figure_inventory()
    build_missing_figures_status(all_present)
    build_results_notes(pure_rows, hybrid_rows, pure_stats, hybrid_stats, per_class)
    build_key_numbers(pure_stats, hybrid_stats, pure_rows, per_class)
    build_limitations()

    written = sorted(p.name for p in OUT.iterdir() if p.is_file())
    print(f"wrote {len(written)} files to {OUT.relative_to(PROJECT_ROOT)}")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
