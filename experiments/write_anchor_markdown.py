"""Render ``frog_anchor_metrics.md`` from the JSON that
``experiments/report_anchor_metrics.py`` writes.

Separate from the measuring script on purpose: the table is what a supervisor
reads and it will be reformatted more than once, while re-measuring costs a full
pass over CIFAR-10 for four models plus four SVC fits. Nothing here recomputes a
number -- it only formats what was already measured.

Run::

    python experiments/write_anchor_markdown.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.utils.config import resolve_path  # noqa: E402

#: The anchor's Table 1, CIFAR-10 / ResNet-18, mean +/- std over all 10 target
#: classes. Transcribed from https://arxiv.org/html/2312.00761v4 so our rows can
#: be read against theirs without leaving the file.
ANCHOR_TABLE_1: list[tuple[str, str, str, str]] = [
    ("Original", "94.89 +/- 0.31", "94.89 +/- 2.75", "0.03 +/- 0.03"),
    ("Retraining (gold standard)", "94.81 +/- 0.52", "0", "100 +/- 0"),
    ("NegGrad", "69.89 +/- 10.23", "0.02 +/- 0.04", "0"),
    ("NegGrad+", "89.91 +/- 1.41", "0.94 +/- 1.87", "98.68 +/- 1.42"),
    ("Tarun et al. 2023 (UNSIR)", "92.20 +/- 0.72", "10.89 +/- 8.79", "61.5 +/- 25.86"),
    ("Kurmanji et al. 2023 (SCRUB)", "94.79 +/- 0.63", "0", "0"),
    ("Foster et al. 2024 (SSD)", "85.76 +/- 25.76", "4.37 +/- 12.79", "87.86 +/- 31.21"),
    ("Kodge et al. 2024 (the anchor's own method)", "94.19 +/- 0.50",
     "0.03 +/- 0.09", "95.5 +/- 14.23"),
]

LABELS = {
    "W_0": "`W_0` (original)",
    "W_ref": "`W_ref` (retain-only reference)",
    "C_star": "`C*` (pure MED-US, front #8)",
    "C_star_refined_bn_frozen": "`C*_refined_bn_frozen`",
}

ANCHOR_COLUMNS = [
    ("anchor_ACC_r", "ACC_r (%)", "{:.2f}"),
    ("anchor_ACC_f", "ACC_f (%)", "{:.2f}"),
    ("anchor_composite", "composite (%)", "{:.2f}"),
    ("anchor_MIA", "MIA (%)", "{:.2f}"),
]

OUR_COLUMNS = [
    ("f1_js", "f1 JS to W_ref", "{:.6f}"),
    ("f2_retain_train_loss", "f2 retain train loss", "{:.6f}"),
    ("f3_edit_cost", "f3 edit cost", "{:.6f}"),
    ("selectivity_S", "S", "{:.4f}"),
    ("mia_auc", "our MIA AUC", "{:.4f}"),
]

DIAGNOSTIC_COLUMNS = [
    ("forget_train_acc", "D_f train acc", "{:.4f}"),
    ("forget_train_loss", "D_f train loss", "{:.4f}"),
    ("forget_test_acc", "D_f test acc", "{:.4f}"),
    ("forget_test_loss", "D_f test loss", "{:.4f}"),
    ("retain_train_acc", "D_r train acc", "{:.4f}"),
    ("retain_train_loss", "D_r train loss", "{:.4f}"),
    ("retain_test_acc", "D_r test acc", "{:.4f}"),
    ("retain_test_loss", "D_r test loss", "{:.4f}"),
    ("kl_to_reference", "KL to W_ref (diag)", "{:.4f}"),
]


def cell(row: dict[str, Any], key: str, fmt: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    if number in (float("inf"), float("-inf")):
        return "inf" if number > 0 else "-inf"
    return fmt.format(number)


def table(order: list[str], rows: dict[str, dict[str, Any]],
          columns: list[tuple[str, str, str]]) -> str:
    header = "| model | " + " | ".join(label for _, label, _ in columns) + " |"
    rule = "|---" * (len(columns) + 1) + "|"
    lines = [header, rule]
    for name in order:
        cells = " | ".join(cell(rows[name], key, fmt) for key, _, fmt in columns)
        lines.append(f"| {LABELS.get(name, name)} | {cells} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",
                        default="results/literature_alignment/frog_anchor_metrics.json")
    parser.add_argument("--out",
                        default="results/literature_alignment/frog_anchor_metrics.md")
    args = parser.parse_args()

    payload = json.loads(resolve_path(args.json).read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = payload["rows"]
    order = [n for n in ("W_0", "W_ref", "C_star", "C_star_refined_bn_frozen")
             if n in rows]

    anchor = payload["anchor"]
    cap = payload.get("max_shadow_per_group")

    parts = [
        "# Frog (class 6) under the anchor protocol",
        "",
        f"Anchor: {anchor['citation']}  ",
        f"Paper: {anchor['paper']}  ",
        f"Code: {anchor['code']}",
        "",
        f"Forget class: **{payload['forget_class']} (frog)**. "
        f"Config: `{payload['config']}`. Loader sizes: `{payload['loader_sizes']}`.",
        "",
        "Measured by `experiments/report_anchor_metrics.py`; no model was trained "
        "and no search was run. `C*` was rebuilt from the chromosome stored on "
        "its Pareto-front row and checked against that row's recorded objectives "
        "before being scored.",
        "",
        "## Anchor-protocol metrics",
        "",
        "`ACC_r` and `ACC_f` are **test-set** accuracies in percent. "
        "`composite = ACC_r x (1 - ACC_f)`, their `metric_function`. "
        "`MIA` is their `SVC_MIA`: an RBF SVC on true-class confidence, fit on "
        "`D_r_train` (member) against `D_f_train` (non-member) and scored as the "
        "fraction of `D_f_test` it calls non-member. Higher `ACC_r` is better, "
        "lower `ACC_f` is better, higher `MIA` is better.",
        "",
        table(order, rows, ANCHOR_COLUMNS),
        "",
    ]

    if cap is None:
        parts += ["The MIA used the full shadow sets, as the anchor does "
                  "(no subsampling).", ""]
    else:
        parts += [f"**Deviation:** the MIA shadow groups were capped at {cap} "
                  f"samples each. The anchor does not subsample.", ""]

    parts += [
        "## The anchor's own Table 1, for comparison",
        "",
        "CIFAR-10 / ResNet-18, mean +/- std over all 10 target classes "
        "(https://arxiv.org/html/2312.00761v4). Our rows above are a **single "
        "class**, so they are not yet commensurable with these; see "
        "`protocol_validation_report.md`.",
        "",
        "| method | ACC_r | ACC_f | MIA |",
        "|---|---|---|---|",
    ]
    parts += [f"| {name} | {acc_r} | {acc_f} | {mia} |"
              for name, acc_r, acc_f, mia in ANCHOR_TABLE_1]
    parts += [
        "",
        "## Our objectives and diagnostics (unchanged)",
        "",
        "Kept as extra columns exactly as previously reported. `f3` is undefined "
        "for `W_ref`, which is an independently trained model rather than an edit "
        "of `W_0`; its distance from `W_0` is recorded separately in the CSV as "
        "`distance_from_W0_not_an_edit`.",
        "",
        table(order, rows, OUR_COLUMNS),
        "",
        "### Accuracies and losses on all four splits",
        "",
        table(order, rows, DIAGNOSTIC_COLUMNS),
        "",
    ]

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
