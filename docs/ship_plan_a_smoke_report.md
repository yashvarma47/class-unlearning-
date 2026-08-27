# Ship (class 8) Plan A smoke test

Mechanical, not scientific. Seven evaluations say nothing about selectivity. They say the
loop runs end to end on class 8, no objective returns `nan`, the `class_contrast` selector
finds connections on the ship split, the **ship** reference is what `f1` is measured
against, and the output files are written in the expected shape.

Run on 2026-08-27, GTX 1650, from a clean working tree. **The full ship search was not
run.**

---

## Commands

```bash
# 1. baseline wiring check -- no search, just the two fixed models
python experiments/check_class_baselines.py \
    --config search/plan_a_ship_smoke.yaml --expect-class 8 \
    --expect-reference-forget-test-acc 0.0 \
    --expect-reference-retain-test-acc 0.9502222222222222 \
    --out results/reference_training/ship_reference_validation.md

# 2. the smoke search: population 4, 1 generation
python experiments/run_plan_a.py --config search/plan_a_ship_smoke.yaml
```

## 1. Baseline wiring check — PASS

Before a search is worth running at all, the two fixed models have to be the right ones.
Point `W_ref` at the wrong file and the search still runs, still produces a Pareto front,
and still writes plausible numbers — about a different experiment.

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9650 | 0.1343 | 0.9460 | 0.2055 |
| `W_ref` (ship, retain-only) | **0.0000** | 10.3602 | **0.9502** | 0.1986 |

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot ship | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.90 | 0.9502 | PASS |
| `W_0` knows ship | ≥ 0.80 | 0.9650 | PASS |

`W_ref` reproduces the values recorded at import exactly (`D_f_test` 0.0000 / 10.3602,
`D_r_test` 0.9502 / 0.1986), to a tolerance of 1e-4. Report:
`results/reference_training/ship_reference_validation.md`.

## 2. Smoke search — PASS

| | |
|---|---|
| config | `search/plan_a_ship_smoke.yaml` |
| population / generations | 4 / 1 |
| evaluated | 7 (+1 cache hit, 8 evaluator calls) |
| **failures** | **0** |
| wall clock | 0.2 min (mean 1.96 s per individual, min 1.57, max 2.48) |
| Pareto front | 4 members |
| search space | 3.874e14 over 30 genes |
| output directory | `results/search/plan_a_ship_smoke/` |

Per-generation best objectives:

| gen | evaluated | cached | failed | best `f1` | best `f2` | best `f3` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 0 | 0 | 0.49710 | 0.00088 | 0.00000 |
| 1 | 7 | 1 | 0 | 0.49710 | 0.00301 | 0.07776 |

Outputs written: `pareto_front.csv` (1,289 B), `evaluation_history.csv` (1,831 B),
`summary.json` (1,137 B). All three are under `results/search/`, which is git-ignored —
a smoke run is not a result, and the frog smoke run is treated the same way.

## 3. Contamination and numerical checks

| question | answer |
|---|---|
| did the config use class 8? | **Yes.** `summary.json` records `forget_class: 8`; the evaluator reported `forget class 8` |
| did it use `class8_ship_reference_best_dr.pt`? | **Yes.** `summary.json` `reference_checkpoint: results/checkpoints/class8_ship_reference_best_dr.pt` |
| did it use `cifar10_class8_ship.json`? | **Yes.** Split sizes 5000 / 45000 / 1000 / 9000 with the ship split file |
| did any frog path appear? | **No.** `grep -ril "frog\|class6\|class_6"` over the smoke outputs, the console log and the wiring report found nothing. The resolved config JSON contains neither `"frog"` nor `"class6"` |
| any failures or `nan`? | **No.** All 8 evaluator calls returned `status: ok`, `error` empty, and no objective or metric cell is non-finite |

One clarification, because a naive grep flags it: `pareto_front.csv` contains `inf` in the
**`crowding_distance`** column on three rows. That is standard NSGA-II — boundary solutions
are given infinite crowding distance so selection always preserves them. Every objective
value is finite.

## 4. What was deliberately not done

* The full ship search (population 10 × 50 generations) — **not run**.
* Full-fidelity re-measurement of the smoke front — pointless on 4 screening-fidelity
  members, and it is what `evaluate_class_front.py` exists for after a real run.
* Any baseline method, any further reference training, any change to the objectives, and
  any change to the frog results or the ship `W_ref`.
* The ship `.pt` was **not** committed to Git LFS. See §6 of `docs/artifact_manifest.md`.

## 5. Readiness

The wiring is correct and the loop runs clean on class 8. The full ship search is a
configuration change away — `search/plan_a_ship.yaml` already carries population 10,
50 generations and `results_dir: results/search/plan_a_ship`, with every other search
setting deliberately identical to `plan_a_frog.yaml` so the two classes stay comparable.

Estimated cost, from the frog run's measured per-individual times at the same fidelity:
roughly 510 evaluations, in the region of 1.5–2 h on the GTX 1650.
