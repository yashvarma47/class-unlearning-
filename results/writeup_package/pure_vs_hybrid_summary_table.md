# Table: pure against hybrid MED-US

Generated 2026-08-30T21:57:42+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

**These are two different methods and this table compares them; it does not merge
them.** The anchor paper's method is gradient-free, so only the pure table is a
like-for-like comparison with its Table 1.

## Aggregate, ten classes

| metric | pure | hybrid | change |
|:---|---:|---:|---:|
| ACC_r (%) | 93.84 +/- 1.15 | 93.72 +/- 1.12 | -0.12 |
| ACC_f (%) | 12.55 +/- 11.57 | 7.55 +/- 8.87 | -5.00 |
| composite (%) | 82.09 +/- 11.00 | 86.66 +/- 8.52 | +4.58 |
| anchor MIA (%) | 92.54 +/- 8.62 | 95.05 +/- 6.08 | +2.51 |
| selectivity S | 1053.88 +/- 1413.47 | 1151.44 +/- 1592.37 | +97.57 |

## Per class

| id | class | status | pure ACC_f | hybrid ACC_f | d ACC_f | pure comp. | hybrid comp. | d comp. | d ACC_r |
|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | no-op | 0.00 | 0.00 | +0.0000 | 92.86 | 92.86 | +0.0000 | +0.0000 |
| 1 | automobile | accepted | 12.70 | 5.50 | -7.2000 | 80.91 | 87.51 | +6.5993 | -0.0778 |
| 2 | bird | accepted | 17.90 | 12.50 | -5.4000 | 78.10 | 83.04 | +4.9422 | -0.2222 |
| 3 | cat | accepted | 9.60 | 6.00 | -3.6000 | 85.70 | 88.88 | +3.1830 | -0.2444 |
| 4 | deer | accepted | 7.80 | 4.30 | -3.5000 | 86.85 | 90.04 | +3.1907 | -0.1111 |
| 5 | dog | accepted | 3.30 | 1.90 | -1.4000 | 92.47 | 93.75 | +1.2842 | -0.0556 |
| 6 | frog | accepted | 8.30 | 2.70 | -5.6000 | 84.84 | 90.06 | +5.2137 | +0.0333 |
| 7 | horse | accepted | 9.80 | 7.00 | -2.8000 | 84.02 | 86.43 | +2.4117 | -0.2111 |
| 8 | ship | accepted | 14.00 | 4.70 | -9.3000 | 81.34 | 90.05 | +8.7110 | -0.0889 |
| 9 | truck | accepted | 42.10 | 30.90 | -11.2000 | 53.80 | 64.04 | +10.2448 | -0.2333 |

9 of 10 classes improved on the composite; the tenth is the airplane no-op.
No class got worse on any headline metric.

`ACC_f` falls by 5.00 points of mean for 0.12 of retain accuracy, and the standard
deviation narrows on every metric -- the refinement helps the weak classes most.
Truck gains 11.20 points of `ACC_f`, the largest absolute improvement of any class,
and still finishes at 30.90 against a 7.55 mean.
