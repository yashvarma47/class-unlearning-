# Table: pure MED-US, all ten CIFAR-10 classes

Generated 2026-08-30T22:55:44+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Every row is **pure gradient-free weight surgery**. No gradient step was applied to
any of these models. The accepted BN-frozen refinements are a different method and
appear in their own table.

Selection rule, applied identically to all ten classes:
`C* = the front member maximising the anchor composite, ACC_r x (1 - ACC_f)`.
That is the anchor paper's own metric function, not one invented here.

| id | class | C* | operators | ACC_r | ACC_f | composite | MIA | S |
|---:|:---|---:|:---|---:|---:|---:|---:|---:|
| 0 | airplane | #0 | `CLIP\|DAMP\|MASK` | 92.86 | 0.00 | 92.86 | 100.00 | 221.25 |
| 1 | automobile | #5 | `CLIP\|MASK\|PRUNE` | 92.68 | 12.70 | 80.91 | 91.70 | 154.57 |
| 2 | bird | #0 | `CLIP\|MASK\|RANDOM_PRUNE` | 95.12 | 17.90 | 78.10 | 91.70 | 1320.40 |
| 3 | cat | #0 | `DAMP\|MASK\|PRUNE\|RANDOM_PRUNE\|RESET` | 94.80 | 9.60 | 85.70 | 95.10 | 168.69 |
| 4 | deer | #2 | `MASK` | 94.20 | 7.80 | 86.85 | 96.00 | 1257.14 |
| 5 | dog | #3 | `MASK` | 95.62 | 3.30 | 92.47 | 99.00 | 4427.91 |
| 6 | frog | #8 | `DAMP\|MASK` | 92.52 | 8.30 | 84.84 | 94.20 | 281.76 |
| 7 | horse | #7 | `MASK` | 93.14 | 9.80 | 84.02 | 91.10 | 176.56 |
| 8 | ship | #6 | `MASK\|RESET` | 94.58 | 14.00 | 81.34 | 97.00 | 2436.50 |
| 9 | truck | #0 | `CLIP\|MASK\|QUANTIZE` | 92.91 | 42.10 | 53.80 | 69.60 | 93.99 |
|  | **mean +/- std** |  |  | **93.84 +/- 1.15** | **12.55 +/- 11.57** | **82.09 +/- 11.00** | **92.54 +/- 8.62** | **1053.88 +/- 1413.47** |

`ACC_r` is retain-test accuracy (higher is better), `ACC_f` forget-test accuracy
(lower is better), composite `ACC_r x (1 - ACC_f)`, MIA the anchor's membership-inference
score (higher is better), `S` the selectivity ratio.

**`MASK` appears in the selected candidate for all ten classes** -- the only operator
that does. 3 select `MASK` alone (deer, dog, horse) and 5 select
it alone or with a single partner. None selects a candidate without it.

The spread is the story: `ACC_f` runs from 0.00 (airplane) to 42.10 (truck), a range of
42 points against a mean of 12.55, while `ACC_r` stays inside
1.15 of its mean. The method's cost is concentrated almost
entirely in forgetting, not in retention.
