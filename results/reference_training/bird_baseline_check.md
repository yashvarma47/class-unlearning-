# Baseline wiring check — class 2 (bird)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_bird.yaml` at 2026-08-29T18:48:07+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 2 (bird) |
| split file | `results/splits/cifar10_class2_bird.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class2_bird_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9260 | 0.2658 | 0.9503 | 0.1909 |
| `W_ref` (retain-only) | 0.0000 | 10.2204 | 0.9526 | 0.1951 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot bird | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9526 | PASS |
| `W_0` knows bird | ≥ 0.8 | 0.9260 | PASS |
