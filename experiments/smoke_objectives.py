"""Smoke-test the objectives against the real W_ref, before spending a search.

Three checks, ordered by how expensive the mistake would be to discover later.

**1. The identity chromosome scores zero on f1 and f3.**
An untouched model must be identical to itself: edit cost exactly 0, and -- when
the reference is the model under test -- JS exactly 0. Against the real W_ref,
f1 is instead the *distance between W_0 and W_ref*, which is the single most
useful number this script prints: it is the gap the search has to close, and
every candidate's f1 should be read against it.

**2. The objectives respond to real edits.**
Random candidates must produce a spread, not a constant. A constant means the
operators are not reaching the weights, or the measurement is not reaching the
operators.

**3. f2 and f3 are not duplicates.**
The point of the exercise. The predecessor project ran four rounds of
experiments with ``f2 = L_r`` and ``f3 = KL(P_ref || P)`` before anyone measured
their correlation and found the two were near-redundant -- a nominally
three-objective search that was really optimising two things. Every candidate f3
metric is measured on every candidate here and its rank correlation against f2 is
reported, so the choice rests on evidence from this run rather than on the
previous one's.

Run::

    python experiments/smoke_objectives.py --config search/plan_a_frog.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation import ClassEvaluator  # noqa: E402
from medus_class.evaluation.objectives import JS_MAX_NATS  # noqa: E402
from medus_class.search import Chromosome, ChromosomeBounds  # noqa: E402
from medus_class.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, without pulling in scipy."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="search/plan_a_frog.yaml")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/analysis/objective_smoke.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 100)
    print("OBJECTIVE SMOKE TEST -- against the real W_ref")
    print("=" * 100)

    evaluator = ClassEvaluator(cfg)
    reference_file = Path(evaluator.reference_path).name

    print(f"  forget class    {evaluator.forget_class}")
    print(f"  W_0             {Path(evaluator.checkpoint_path).name}")
    print(f"  W_ref           {reference_file}")
    print(f"  selector        {evaluator.selection_rule}")
    print(f"  loader sizes    {evaluator.loaders.sizes()}")

    if "_best_dr" not in reference_file:
        print("\n  WARNING: W_ref is not a '_best_dr' checkpoint. For a")
        print("  retain-only reference, full-test accuracy is the wrong")
        print("  selection criterion -- see kaggle/README_KAGGLE.md.")

    print("\n" + "-" * 100)
    print("BASELINES")
    print("-" * 100)
    print(f"  {'':<12}{'D_f acc':>10}{'D_f loss':>11}{'D_r acc':>10}"
          f"{'D_f_test':>11}{'D_r_test':>11}")
    for label, m in (("W_0", evaluator.original),
                     ("W_ref", evaluator.reference_metrics)):
        print(f"  {label:<12}{m['forget_train_acc']:>10.4f}{m['forget_train_loss']:>11.4f}"
              f"{m['retain_train_acc']:>10.4f}{m['forget_test_acc']:>11.4f}"
              f"{m['retain_test_acc']:>11.4f}")

    # --- check 1 ----------------------------------------------------------
    print("\n" + "-" * 100)
    print("CHECK 1  identity chromosome -- W_0 measured against W_ref")
    print("-" * 100)
    bounds = ChromosomeBounds.from_registry(
        n_groups=len(evaluator.registry.names),
        implemented_only=True,
        max_level=cfg["evaluation"].get("max_level"),
    )
    identity = evaluator.evaluate(Chromosome.identity(bounds))

    print(f"  f1  JS(W_ref || W_0)      {identity.obj1_js:.6f}   "
          f"of a possible {JS_MAX_NATS:.4f}")
    print(f"  f2  retain loss           {identity.obj2_retain_loss:.6f}")
    print(f"  f3  edit cost             {identity.obj3_edit_cost:.8f}")
    print(f"      D_f_test acc          {identity.forget_test_acc:.4f}   "
          f"(W_ref {evaluator.reference_metrics['forget_test_acc']:.4f})")

    edit_cost_ok = abs(identity.obj3_edit_cost) < 1e-9
    print(f"\n  edit cost is exactly zero for an untouched model: "
          f"{'PASS' if edit_cost_ok else 'FAIL'}")
    print(f"  f1 = {identity.obj1_js:.6f} is the GAP THE SEARCH MUST CLOSE.")
    print("  Read every candidate's f1 against this, not against zero.")

    # --- checks 2 and 3 ---------------------------------------------------
    print("\n" + "-" * 100)
    print(f"CHECK 2/3  {args.candidates} random candidates")
    print("-" * 100)
    print(f"  {'#':<4}{'f1 JS':>10}{'f2 L_r':>10}{'f3 edit':>10}{'KL':>10}"
          f"{'D_f acc':>10}{'D_f_test':>10}{'D_r_test':>10}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for i in range(args.candidates):
        chromosome = Chromosome.random(
            bounds, rng, p_active=float(cfg["search"].get("p_active", 0.5))
        )
        result = evaluator.evaluate(chromosome)
        rows.append({
            "index": i,
            "status": result.status,
            "f1_js": result.obj1_js,
            "f2_retain_loss": result.obj2_retain_loss,
            "f3_edit_cost": result.obj3_edit_cost,
            "kl_to_reference": result.kl_to_reference,
            "forget_train_acc": result.forget_train_acc,
            "forget_test_acc": result.forget_test_acc,
            "retain_test_acc": result.retain_test_acc,
            "selectivity_S": result.selectivity_S,
            "mia_auc": result.mia_auc,
        })
        print(f"  {i:<4}{result.obj1_js:>10.5f}{result.obj2_retain_loss:>10.4f}"
              f"{result.obj3_edit_cost:>10.5f}{result.kl_to_reference:>10.4f}"
              f"{result.forget_train_acc:>10.4f}{result.forget_test_acc:>10.4f}"
              f"{result.retain_test_acc:>10.4f}")

    ok = [r for r in rows if r["status"] == "ok"]
    if len(ok) < 3:
        print(f"\n  FAIL: only {len(ok)} of {len(rows)} candidates evaluated.")
        return 1

    f2 = [r["f2_retain_loss"] for r in ok]
    corr_edit = spearman(f2, [r["f3_edit_cost"] for r in ok])
    corr_kl = spearman(f2, [r["kl_to_reference"] for r in ok])
    corr_f1 = spearman(f2, [r["f1_js"] for r in ok])

    print("\n" + "-" * 100)
    print("CHECK 3  is f3 independent of f2?")
    print("-" * 100)
    print("  Spearman rank correlation against f2 (retain loss):")
    print(f"    f3 = edit cost           {corr_edit:+.4f}   <- the chosen f3")
    print(f"    (KL to reference)        {corr_kl:+.4f}   <- the predecessor's f3")
    print(f"    (f1 = JS to reference)   {corr_f1:+.4f}")
    print("\n  Near +/-1 means the objective restates f2 and the search is really")
    print("  optimising two things, not three. Edit cost never reads the data, so")
    print("  it should sit well below the KL.")

    spread = {
        "f1_js": (min(r["f1_js"] for r in ok), max(r["f1_js"] for r in ok)),
        "f2_retain_loss": (min(f2), max(f2)),
        "f3_edit_cost": (min(r["f3_edit_cost"] for r in ok),
                         max(r["f3_edit_cost"] for r in ok)),
    }
    print("\n  Value ranges (a constant means the objective is not responding):")
    all_vary = True
    for name, (low, high) in spread.items():
        varies = high - low > 1e-6
        all_vary &= varies
        print(f"    {name:<18}{low:>12.6f} ..{high:>12.6f}   "
              f"{'PASS' if varies else 'FAIL -- constant'}")

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": args.config,
        "reference_checkpoint": evaluator.reference_path,
        "forget_class": evaluator.forget_class,
        "baselines": {"W_0": evaluator.original,
                      "W_ref": evaluator.reference_metrics},
        "identity": {
            "f1_js": identity.obj1_js,
            "f2_retain_loss": identity.obj2_retain_loss,
            "f3_edit_cost": identity.obj3_edit_cost,
            "edit_cost_is_zero": bool(edit_cost_ok),
        },
        "candidates": rows,
        "spearman_vs_f2": {
            "edit_cost": corr_edit,
            "kl_reference": corr_kl,
            "f1_js": corr_f1,
        },
        "ranges": {k: list(v) for k, v in spread.items()},
    }, indent=2), encoding="utf-8")

    passed = edit_cost_ok and all_vary
    print("\n" + "=" * 100)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 100)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
