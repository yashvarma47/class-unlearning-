# Table: hybrid MED-US (pure `C*` + BN-frozen refinement), all ten classes

Generated 2026-08-30T22:55:44+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

**This is not the pure method.** Each refined row is the pure `C*` followed by one
clipped gradient-ascent step on `D_f` and one repair step on `D_r`, applied outside
the evolutionary search with BatchNorm frozen. Nine classes were eligible and all
nine were accepted. Airplane is a deliberate no-op -- its pure `ACC_f` is already
0.00 with anchor MIA 100.00, so a forgetting step has nothing to improve -- and its
row is the pure `C*` unchanged.

| id | class | status | ACC_r | ACC_f | composite | MIA | S | param mvmt | BN mvmt |
|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | no-op | 92.86 | 0.00 | 92.86 | 100.00 | 221.25 | -- | -- |
| 1 | automobile | accepted | 92.60 | 5.50 | 87.51 | 95.80 | 178.42 | 0.000368 | 0.000000 |
| 2 | bird | accepted | 94.90 | 12.50 | 83.04 | 93.70 | 1305.45 | 0.000343 | 0.000000 |
| 3 | cat | accepted | 94.56 | 6.00 | 88.88 | 96.70 | 172.64 | 0.000303 | 0.000000 |
| 4 | deer | accepted | 94.09 | 4.30 | 90.04 | 97.50 | 1262.49 | 0.000373 | 0.000000 |
| 5 | dog | accepted | 95.57 | 1.90 | 93.75 | 99.50 | 4950.81 | 0.000357 | 0.000000 |
| 6 | frog | accepted | 92.56 | 2.70 | 90.06 | 96.30 | 297.25 | 0.000420 | 0.000000 |
| 7 | horse | accepted | 92.93 | 7.00 | 86.43 | 92.80 | 185.79 | 0.000394 | 0.000000 |
| 8 | ship | accepted | 94.49 | 4.70 | 90.05 | 99.10 | 2832.90 | 0.000409 | 0.000000 |
| 9 | truck | accepted | 92.68 | 30.90 | 64.04 | 79.10 | 107.42 | 0.000307 | 0.000000 |
|  | **mean +/- std** |  | **93.72 +/- 1.12** | **7.55 +/- 8.87** | **86.66 +/- 8.52** | **95.05 +/- 6.08** | **1151.44 +/- 1592.37** |  |  |

BatchNorm buffer movement is **exactly 0.000000 on every accepted refinement**, with
zero counter changes. Parameter movement stays between 0.000303 and 0.000420 against
a budget of 0.0400.

That column is load-bearing. An earlier, unfrozen attempt passed every weight-based
guard while eight batches of `D_r` silently re-estimated the running statistics and
undid the operator edit. Freezing BatchNorm and checking buffer movement explicitly
is what makes these nine results trustworthy.
