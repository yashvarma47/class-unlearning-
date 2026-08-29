# Baseline wiring check — class 3 (cat)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_cat.yaml` at 2026-08-29T19:18:15+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 3 (cat) |
| split file | `results/splits/cifar10_class3_cat.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class3_cat_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9040 | 0.3846 | 0.9528 | 0.1777 |
| `W_ref` (retain-only) | 0.0000 | 10.4254 | 0.9612 | 0.1530 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot cat | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9612 | PASS |
| `W_0` knows cat | ≥ 0.8 | 0.9040 | PASS |
