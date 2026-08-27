# Baseline wiring check — class 8 (ship)

Produced by `experiments/check_class_baselines.py` from `search/plan_a_ship_smoke.yaml` at 2026-08-27T19:43:25+00:00.

**Result: PASS**

| | |
|---|---|
| forget class | 8 (ship) |
| split file | `results/splits/cifar10_class8_ship.json` |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class8_ship_reference_best_dr.pt` |
| loader sizes | `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}` |
| device | cuda:0 |

| model | `D_f_test` acc | `D_f_test` loss | `D_r_test` acc | `D_r_test` loss |
|---|---:|---:|---:|---:|
| `W_0` (original) | 0.9650 | 0.1343 | 0.9460 | 0.2055 |
| `W_ref` (retain-only) | 0.0000 | 10.3602 | 0.9502 | 0.1986 |

## Checks

| check | threshold | measured | verdict |
|---|---|---:|---|
| `W_ref` forgot ship | ≤ 0.02 | 0.0000 | PASS |
| `W_ref` still useful | ≥ 0.9 | 0.9502 | PASS |
| `W_0` knows ship | ≥ 0.8 | 0.9650 | PASS |
