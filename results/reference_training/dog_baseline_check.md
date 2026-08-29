# Baseline wiring check — class 5 (dog)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_dog.yaml` at 2026-08-29T20:24:37+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 5 (dog) |
| split file | `results/splits/cifar10_class5_dog.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class5_dog_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9090 | 0.3755 | 0.9522 | 0.1787 |
| `W_ref` (retain-only) | 0.0000 | 11.0671 | 0.9576 | 0.1664 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot dog | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9576 | PASS |
| `W_0` knows dog | ≥ 0.8 | 0.9090 | PASS |
