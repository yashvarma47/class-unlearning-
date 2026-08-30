# Pure vs hybrid, class by class

Generated 2026-08-30T00:41:30+00:00 by `experiments/build_hybrid_summary.py`.

**Pure** = gradient-free MED-US alone. **Hybrid** = that same `C*` plus one forget step and one retain repair, BatchNorm frozen, applied outside the search. They are different methods and are reported separately on purpose.

| id | class | status | ACC_r pure -> hybrid | ACC_f pure -> hybrid | composite pure -> hybrid | MIA pure -> hybrid |
|---:|---|---|---|---|---|---|
| 0 | airplane | no-op | 92.86 -> 92.86 (+0.00) | 0.00 -> 0.00 (+0.00) | 92.86 -> 92.86 (+0.00) | 100.00 -> 100.00 (+0.00) |
| 1 | automobile | accepted | 92.68 -> 92.60 (-0.08) | 12.70 -> 5.50 (-7.20) | 80.91 -> 87.51 (+6.60) | 91.70 -> 95.80 (+4.10) |
| 2 | bird | accepted | 95.12 -> 94.90 (-0.22) | 17.90 -> 12.50 (-5.40) | 78.10 -> 83.04 (+4.94) | 91.70 -> 93.70 (+2.00) |
| 3 | cat | accepted | 94.80 -> 94.56 (-0.24) | 9.60 -> 6.00 (-3.60) | 85.70 -> 88.88 (+3.18) | 95.10 -> 96.70 (+1.60) |
| 4 | deer | accepted | 94.20 -> 94.09 (-0.11) | 7.80 -> 4.30 (-3.50) | 86.85 -> 90.04 (+3.19) | 96.00 -> 97.50 (+1.50) |
| 5 | dog | accepted | 95.62 -> 95.57 (-0.06) | 3.30 -> 1.90 (-1.40) | 92.47 -> 93.75 (+1.28) | 99.00 -> 99.50 (+0.50) |
| 6 | frog | accepted | 92.52 -> 92.56 (+0.03) | 8.30 -> 2.70 (-5.60) | 84.84 -> 90.06 (+5.21) | 94.20 -> 96.30 (+2.10) |
| 7 | horse | accepted | 93.14 -> 92.93 (-0.21) | 9.80 -> 7.00 (-2.80) | 84.02 -> 86.43 (+2.41) | 91.10 -> 92.80 (+1.70) |
| 8 | ship | accepted | 94.58 -> 94.49 (-0.09) | 14.00 -> 4.70 (-9.30) | 81.34 -> 90.05 (+8.71) | 97.00 -> 99.10 (+2.10) |
| 9 | truck | accepted | 92.91 -> 92.68 (-0.23) | 42.10 -> 30.90 (-11.20) | 53.80 -> 64.04 (+10.24) | 69.60 -> 79.10 (+9.50) |

## Aggregate

| metric | pure | hybrid | change |
|---|---|---|---|
| `ACC_r` (%) | 93.84 +/- 1.15 | 93.72 +/- 1.12 | -0.12 |
| `ACC_f` (%) | 12.55 +/- 11.57 | 7.55 +/- 8.87 | -5.00 |
| composite (%) | 82.09 +/- 11.00 | 86.66 +/- 8.52 | +4.58 |
| anchor MIA (%) | 92.54 +/- 8.62 | 95.05 +/- 6.08 | +2.51 |
| `S` | 1053.9 +/- 1413.5 | 1151.4 +/- 1592.4 | +97.6 |

**9 of 10 classes improved on the composite.** The 1 no-op class is airplane, whose pure `ACC_f` is already 0.00. 0 refinements were rejected.

