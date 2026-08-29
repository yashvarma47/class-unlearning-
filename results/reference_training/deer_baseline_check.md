# Baseline wiring check — class 4 (deer)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_deer.yaml` at 2026-08-29T19:53:26+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 4 (deer) |
| split file | `results/splits/cifar10_class4_deer.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class4_deer_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9570 | 0.1720 | 0.9469 | 0.2013 |
| `W_ref` (retain-only) | 0.0000 | 9.3447 | 0.9457 | 0.2203 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot deer | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9457 | PASS |
| `W_0` knows deer | ≥ 0.8 | 0.9570 | PASS |
