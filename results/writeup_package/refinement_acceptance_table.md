# Table: BN-frozen refinement acceptance record

Generated 2026-08-30T21:57:42+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

**Nine attempts, nine accepted, zero rejected.** Airplane was not attempted: its pure
`ACC_f` is already 0.00 with anchor MIA 100.00, so there was nothing for a forgetting
step to improve.

| id | class | outcome | param movement | BN movement | BN counters | `D_r_test` drop | checks passed |
|---:|:---|:---:|---:|---:|---:|---:|:---:|
| 0 | airplane | no-op | -- | -- | -- | -- | pure ACC_f already 0.00 |
| 1 | automobile | **accepted** | 0.000368 | 0.000000 | 0 | 0.000778 | 6 / 6 |
| 2 | bird | **accepted** | 0.000343 | 0.000000 | 0 | 0.002222 | 6 / 6 |
| 3 | cat | **accepted** | 0.000303 | 0.000000 | 0 | 0.002444 | 6 / 6 |
| 4 | deer | **accepted** | 0.000373 | 0.000000 | 0 | 0.001111 | 6 / 6 |
| 5 | dog | **accepted** | 0.000357 | 0.000000 | 0 | 0.000556 | 6 / 6 |
| 6 | frog | **accepted** | 0.000420 | 0.000000 | 0 | -0.000333 | 6 / 6 |
| 7 | horse | **accepted** | 0.000394 | 0.000000 | 0 | 0.002111 | 6 / 6 |
| 8 | ship | **accepted** | 0.000409 | 0.000000 | 0 | 0.000889 | 6 / 6 |
| 9 | truck | **accepted** | 0.000307 | 0.000000 | 0 | 0.002333 | 6 / 6 |

## The six acceptance checks

Every refinement had to pass all six before its checkpoint was kept:

1. forget improved on `D_f_test`
2. `D_r_test` drop <= 0.010
3. no utility collapse (retain losses <= 1.25x)
4. edit cost <= 0.3
5. parameter movement <= 0.0400
6. BatchNorm buffers unchanged

## Hyperparameters, identical for all nine

| | |
|---|---|
| forget step | SGD gradient ASCENT on cross-entropy over D_f |
| retain step | SGD gradient DESCENT on cross-entropy over D_r |
| forget lr | 0.0001 |
| retain lr | 0.0001 |
| batches per step | 8 |
| steps | one optimiser step per stage, on the mean gradient |
| BatchNorm | FROZEN -- model held in eval mode for both steps; running_mean/running_var/num_batches_tracked cannot update |
| max buffer movement | 1e-09 |
| max `D_r_test` drop | 0.01 |
| seed | 42 |

## Why check 6 exists

An earlier attempt on frog passed every weight-based guard and was reported as a
success. It was not one: the model was left in training mode, and eight batches of
`D_r` re-estimated the BatchNorm running statistics, undoing the operator edit while
parameter movement, edit cost and retain accuracy all looked correct. Buffer movement
is the only check that catches it, and it reads exactly 0.000000 on all nine
refinements above.
