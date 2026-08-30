# Ten-class HYBRID result (pure C* + BN-frozen refinement)

Generated 2026-08-30T00:41:30+00:00 by `experiments/build_hybrid_summary.py`. Nothing is recomputed here.

**This is not pure gradient-free MED-US.** Every row marked *refined (hybrid)* had one clipped gradient-ascent step on `D_f` and one repair step on `D_r` applied after the search, with BatchNorm frozen. The anchor paper's own method is gradient-free, so the like-for-like comparison with it remains `ten_class_pure_summary.md`.

| id | class | source | ACC_r | ACC_f | composite | MIA | S | dW | dBN |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | pure C* | 92.86 | 0.00 | 92.86 | 100.00 | 221.25 | -- | -- |
| 1 | automobile | refined (hybrid) | 92.60 | 5.50 | 87.51 | 95.80 | 178.42 | 0.000368 | 0.000000 |
| 2 | bird | refined (hybrid) | 94.90 | 12.50 | 83.04 | 93.70 | 1305.45 | 0.000343 | 0.000000 |
| 3 | cat | refined (hybrid) | 94.56 | 6.00 | 88.88 | 96.70 | 172.64 | 0.000303 | 0.000000 |
| 4 | deer | refined (hybrid) | 94.09 | 4.30 | 90.04 | 97.50 | 1262.49 | 0.000373 | 0.000000 |
| 5 | dog | refined (hybrid) | 95.57 | 1.90 | 93.75 | 99.50 | 4950.81 | 0.000357 | 0.000000 |
| 6 | frog | refined (hybrid) | 92.56 | 2.70 | 90.06 | 96.30 | 297.25 | 0.000420 | 0.000000 |
| 7 | horse | refined (hybrid) | 92.93 | 7.00 | 86.43 | 92.80 | 185.79 | 0.000394 | 0.000000 |
| 8 | ship | refined (hybrid) | 94.49 | 4.70 | 90.05 | 99.10 | 2832.90 | 0.000409 | 0.000000 |
| 9 | truck | refined (hybrid) | 92.68 | 30.90 | 64.04 | 79.10 | 107.42 | 0.000307 | 0.000000 |

## Refinement attempts

| id | class | status | note |
|---:|---|---|---|
| 0 | airplane | **no-op** | pure ACC_f is already 0.00 with anchor MIA 100.00 -- a forgetting step has nothing to improve |
| 1 | automobile | **accepted** | -- |
| 2 | bird | **accepted** | -- |
| 3 | cat | **accepted** | -- |
| 4 | deer | **accepted** | -- |
| 5 | dog | **accepted** | -- |
| 6 | frog | **accepted** | -- |
| 7 | horse | **accepted** | -- |
| 8 | ship | **accepted** | -- |
| 9 | truck | **accepted** | -- |

Accepted: 9. Rejected: 0. No-op: 1.

Every accepted refinement was required to hold BatchNorm buffer movement at **exactly zero**, and every one did. That is the condition the first frog attempt failed silently: eight batches of `D_r` re-estimated the running statistics and undid the operator edit while every weight-based guard reported success.

## Mean +/- std over the ten classes

| metric | hybrid | pure |
|---|---|---|
| `ACC_r` (%) | 93.72 +/- 1.12 | 93.84 +/- 1.15 |
| `ACC_f` (%) | **7.55 +/- 8.87** | 12.55 +/- 11.57 |
| composite (%) | **86.66 +/- 8.52** | 82.09 +/- 11.00 |
| anchor MIA (%) | **95.05 +/- 6.08** | 92.54 +/- 8.62 |
| `S` | 1151.4 +/- 1592.4 | 1053.9 +/- 1413.5 |

