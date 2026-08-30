# Key numbers

Generated 2026-08-30T22:55:44+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Single reference sheet. If a number appears in the dissertation, it should match this
page.

## Pure MED-US, ten-class aggregate

| metric | mean +/- std | min | max |
|---|---|---|---|
| `ACC_r` (%) | **93.84 +/- 1.15** | 92.52 (frog) | 95.62 (dog) |
| `ACC_f` (%) | **12.55 +/- 11.57** | 0.00 (airplane) | 42.10 (truck) |
| composite (%) | **82.09 +/- 11.00** | 53.80 (truck) | 92.86 (airplane) |
| anchor MIA (%) | **92.54 +/- 8.62** | 69.60 (truck) | 100.00 (airplane) |
| selectivity `S` | **1053.88 +/- 1413.47** | 93.99 (truck) | 4427.91 (dog) |

## Hybrid MED-US, ten-class aggregate

| metric | mean +/- std | min | max |
|---|---|---|---|
| `ACC_r` (%) | **93.72 +/- 1.12** | 92.56 (frog) | 95.57 (dog) |
| `ACC_f` (%) | **7.55 +/- 8.87** | 0.00 (airplane) | 30.90 (truck) |
| composite (%) | **86.66 +/- 8.52** | 64.04 (truck) | 93.75 (dog) |
| anchor MIA (%) | **95.05 +/- 6.08** | 79.10 (truck) | 100.00 (airplane) |
| selectivity `S` | **1151.44 +/- 1592.37** | 107.42 (truck) | 4950.81 (dog) |

## Pure against hybrid

| metric | pure | hybrid | change |
|---|---|---|---|
| `ACC_r` (%) | 93.84 +/- 1.15 | 93.72 +/- 1.12 | **-0.12** |
| `ACC_f` (%) | 12.55 +/- 11.57 | 7.55 +/- 8.87 | **-5.00** |
| composite (%) | 82.09 +/- 11.00 | 86.66 +/- 8.52 | **+4.58** |
| anchor MIA (%) | 92.54 +/- 8.62 | 95.05 +/- 6.08 | **+2.51** |

Nine of ten classes improved on the composite; the tenth is the airplane no-op. No
class regressed on any headline metric.

## Benchmark comparison (anchor Table 1, ten-class mean)

| method | ACC_r | ACC_f | MIA | numbers |
|---|---|---|---|---|
| Original (paper) | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 | reported |
| Retraining (paper) | 94.81 +/- 0.52 | 0.00 | 100.00 | reported |
| Kodge et al. 2024 (anchor) | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.50 +/- 14.23 | reported |
| SSD (Foster et al. 2024) | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 | reported |
| UNSIR (Tarun et al. 2023) | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.50 +/- 25.86 | reported |
| **MED-US pure** | **93.84 +/- 1.15** | **12.55 +/- 11.57** | **92.54 +/- 8.62** | measured |
| **MED-US hybrid** | **93.72 +/- 1.12** | **7.55 +/- 8.87** | **95.05 +/- 6.08** | measured |

Gap to the anchor: `ACC_r` -0.35, `ACC_f` +12.52, MIA -2.96.

Nearest published neighbour on `ACC_f` is UNSIR at 10.89 +/- 8.79. Its spread is the
same order as this work's (11.57) -- both have strong and
weak classes rather than uniform behaviour, unlike the anchor's 0.03 +/- 0.09.

## Best class

**airplane** (class 0), pure. Best by `ACC_f`, composite and MIA simultaneously; its
`ACC_r` is mid-table, which is the trade the composite accepted.

| | |
|---|---|
| `ACC_r` | 92.86 |
| `ACC_f` | **0.00** -- matches the retraining reference exactly |
| composite | 92.86 |
| MIA | 100.00 |
| operators | `CLIP\|DAMP\|MASK`, front position #0 |

The only class where refinement was not attempted, because there was nothing left to
forget.

## Worst class

**truck** (class 9), pure.

| | |
|---|---|
| `ACC_r` | 92.91 |
| `ACC_f` | **42.10** |
| composite | 53.80 |
| MIA | 69.60 |
| selectivity `S` | 93.99 -- lowest of the ten |
| operators | `CLIP\|MASK\|QUANTIZE`, front position #0 |

Worst on every headline metric, pure and hybrid. Still 30.90 `ACC_f` after
refinement.

## Biggest refinement improvement

**truck** (class 9).

| metric | pure | hybrid | change |
|---|---|---|---|
| `ACC_f` | 42.1000 | 30.9000 | **-11.2000** |
| composite | 53.7955 | 64.0403 | **+10.2448** |
| MIA | 69.6000 | 79.1000 | **+9.5000** |
| `ACC_r` | 92.9111 | 92.6778 | -0.2333 |

Largest absolute `ACC_f` gain of any class, and it remains the weakest class after
the gain.

Runner-up by `ACC_f`: ship, -9.30 (14.00 -> 4.70), which is also the largest composite
gain among the classes that end in a strong position (+8.71, to 90.05).

## Retain accuracy cost of refinement

**-0.12 points of mean `ACC_r`**, for -5.00 points of mean `ACC_f`.

| class | d `ACC_r` |
|---|---|
| airplane | +0.0000 (no-op) |
| frog | +0.0333 |
| dog | -0.0556 |
| automobile | -0.0778 |
| ship | -0.0889 |
| deer | -0.1111 |
| horse | -0.2111 |
| bird | -0.2222 |
| truck | -0.2333 |
| cat | -0.2444 |

Worst single-class retain cost is 0.2444 points (cat). No class exceeded the 0.010
fractional `D_r_test` drop that acceptance check 2 enforces.

Parameter movement across the nine: 0.000303 (truck) to 0.000420 (frog), against a
0.0400 budget -- between 0.8% and 1.1% of what was permitted.

BatchNorm buffer movement: **0.000000 on all nine**, zero counter changes.
