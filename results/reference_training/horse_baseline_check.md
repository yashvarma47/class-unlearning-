# Baseline wiring check — class 7 (horse)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_horse.yaml` at 2026-08-29T21:03:44+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 7 (horse) |
| split file | `results/splits/cifar10_class7_horse.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class7_horse_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9550 | 0.1336 | 0.9471 | 0.2056 |
| `W_ref` (retain-only) | 0.0000 | 9.9951 | 0.9423 | 0.2382 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot horse | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9423 | PASS |
| `W_0` knows horse | ≥ 0.8 | 0.9550 | PASS |
