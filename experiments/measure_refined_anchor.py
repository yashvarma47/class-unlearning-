"""Add the refined model's anchor row to a class's existing anchor CSV.

``report_anchor_metrics.py`` re-measures every row it writes -- ``W_0``,
``W_ref``, ``C*``, the best-S member and the refined model -- which costs about
eleven minutes per class. Four of those five rows already exist, are correct,
and are committed. This measures **only** the refined model and merges it in.

The point is not just speed. Re-running the full script would recompute rows
that have already been reported, and a re-measurement that came back even
slightly different -- a different GPU reduction order, a library upgrade -- would
silently change a published number. Carrying the existing rows through as raw
text makes that impossible.

Guarantees
----------
1. Existing rows are copied **as raw CSV text**, never parsed and re-serialised,
   so they cannot be perturbed by float formatting.
2. After writing, the file is re-read and every preserved row is compared
   **byte for byte** against what was there before. Any difference restores the
   backup and exits non-zero.
3. The header is preserved exactly, and the refined row is written under that
   header's column order.
4. Only ``C_star_refined_bn_frozen`` is added or replaced. Nothing else is
   touched.

Run::

    python experiments/measure_refined_anchor.py --config search/plan_a_dog.yaml \\
        --class-name dog
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_anchor_metrics import measure_model  # noqa: E402

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.objectives import selectivity  # noqa: E402
from medus_class.models import build_model, load_checkpoint  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

REFINED_ROW = "C_star_refined_bn_frozen"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--refined", default=None)
    parser.add_argument("--anchor-csv", default=None)
    parser.add_argument("--max-shadow-per-group", type=int, default=None)
    args = parser.parse_args()

    name = args.class_name
    refined_path = resolve_path(
        args.refined or f"results/search/plan_a_{name}_bn_frozen_refined/refined_best.pt")
    anchor_csv = resolve_path(
        args.anchor_csv or f"results/search/plan_a_{name}/{name}_anchor_metrics.csv")

    if not refined_path.is_file():
        raise SystemExit(f"refined checkpoint not found: {refined_path}")
    if not anchor_csv.is_file():
        raise SystemExit(f"anchor CSV not found: {anchor_csv}")

    print("=" * 100)
    print(f"REFINED ANCHOR ROW -- class '{name}'")
    print("=" * 100)

    # --- 1. keep the existing file, verbatim ------------------------------
    original_text = anchor_csv.read_text(encoding="utf-8")
    original_lines = original_text.splitlines()
    header_line, *data_lines = original_lines
    header = next(csv.reader(io.StringIO(header_line)))
    preserved = [line for line in data_lines
                 if not line.startswith(f"{REFINED_ROW},")]
    replaced = len(preserved) != len(data_lines)
    print(f"  existing rows kept verbatim: {len(preserved)}"
          + ("  (an old refined row will be replaced)" if replaced else ""))
    for line in preserved:
        print(f"    {line.split(',')[0]}")

    backup = anchor_csv.with_suffix(".csv.bak")
    shutil.copy2(anchor_csv, backup)

    # --- 2. measure ONLY the refined model --------------------------------
    cfg = load_config(args.config)
    cfg["evaluation"]["forget_subset_size"] = None
    cfg["evaluation"]["retain_subset_size"] = None
    cfg["evaluation"]["num_workers"] = 0
    cfg["evaluation"]["measure_retain_test"] = True
    seed = int(cfg.get("seed", 42))

    evaluator = ClassEvaluator(cfg)
    model = build_model(cfg["model"], num_classes=int(cfg["data"]["num_classes"]))
    load_checkpoint(refined_path, model, map_location="cpu")
    print(f"\n  measuring {refined_path.name} ...")
    measured: dict[str, Any] = measure_model(
        evaluator, model, args.max_shadow_per_group, seed)
    measured["selectivity_S"] = selectivity(
        measured["forget_train_loss"], measured["retain_train_loss"],
        evaluator.original["forget_train_loss"],
        evaluator.original["retain_train_loss"],
    )
    measured["kind"] = ("POST-SEARCH REFINEMENT -- HYBRID, not pure "
                        "gradient-free; not a Pareto-front member")
    measured["operators"] = "pure C* + 1 forget step + 1 retain repair (BN frozen)"

    # --- 3. write: preserved lines untouched, refined appended ------------
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([REFINED_ROW] + [measured.get(c, "") for c in header[1:]])
    refined_line = buffer.getvalue().rstrip("\n")

    anchor_csv.write_text(
        "\n".join([header_line, *preserved, refined_line]) + "\n", encoding="utf-8")

    # --- 4. verify the preserved rows are byte-identical ------------------
    written = anchor_csv.read_text(encoding="utf-8").splitlines()
    expected = [header_line, *preserved]
    problems = [f"line {i}: {expected[i]!r} != {written[i]!r}"
                for i in range(len(expected))
                if i >= len(written) or written[i] != expected[i]]
    if problems:
        shutil.copy2(backup, anchor_csv)
        print("\n  VERIFICATION FAILED -- existing rows changed. Backup restored.")
        for line in problems[:5]:
            print(f"    {line}")
        return 1
    if written[-1] != refined_line:
        shutil.copy2(backup, anchor_csv)
        print("\n  VERIFICATION FAILED -- refined row not written. Backup restored.")
        return 1

    print(f"\n  VERIFIED: {len(expected)} pre-existing lines (header + "
          f"{len(preserved)} rows) are byte-identical; 1 refined row appended.")
    backup.unlink()

    # --- 5. mirror into the JSON, same guarantee --------------------------
    anchor_json = anchor_csv.with_suffix(".json")
    if anchor_json.is_file():
        payload = json.loads(anchor_json.read_text(encoding="utf-8"))
        before = {k: v for k, v in payload.get("rows", {}).items() if k != REFINED_ROW}
        payload.setdefault("rows", {})[REFINED_ROW] = measured
        anchor_json.write_text(json.dumps(payload, indent=1, default=str),
                               encoding="utf-8")
        after = {k: v for k, v in
                 json.loads(anchor_json.read_text(encoding="utf-8"))["rows"].items()
                 if k != REFINED_ROW}
        if before != after:
            print("  WARNING: JSON rows other than the refined one changed")
            return 1
        print(f"  JSON updated; its other {len(before)} rows are unchanged.")

    print(f"\n  {'ACC_r (%)':<22}{measured['anchor_ACC_r']:>12.4f}")
    print(f"  {'ACC_f (%)':<22}{measured['anchor_ACC_f']:>12.4f}")
    print(f"  {'composite (%)':<22}{measured['anchor_composite']:>12.4f}")
    mia = measured.get("anchor_MIA")
    print(f"  {'anchor MIA (%)':<22}"
          + (f"{mia:>12.4f}" if mia is not None else f"{'not computed':>12}"))
    print(f"  {'S':<22}{measured['selectivity_S']:>12.4f}")
    print(f"\n  wrote {anchor_csv.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
