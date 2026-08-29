# Baseline wiring check — class 0 (airplane)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_airplane_smoke.yaml` at 2026-08-29T16:22:46+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 0 (airplane) |
| split file | `results/splits/cifar10_class0_airplane.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class0_airplane_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9580 | 0.1475 | 0.9468 | 0.2040 |
| `W_ref` (retain-only) | 0.0000 | 9.9663 | 0.9516 | 0.1973 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot airplane | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9516 | PASS |
| `W_0` knows airplane | ≥ 0.8 | 0.9580 | PASS |
