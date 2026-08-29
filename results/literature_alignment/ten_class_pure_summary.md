# Ten-class pure gradient-free MED-US result

Generated 2026-08-29T23:08:06+00:00 by `experiments/build_ten_class_summary.py` from each class's own artefacts. Nothing here is recomputed.

**10 of 10 classes.**

Every row is **pure gradient-free weight surgery** -- no gradient step was applied to any of these models. The accepted BN-frozen refinements for frog and ship are hybrids and are deliberately excluded.

## Selection rule

`C* = the front member maximising the anchor composite, ACC_r x (1 - ACC_f)`.

One rule for all ten, and it is the anchor paper's own `metric_function` rather than something invented here. It reproduces the three classes that were selected by hand before the sweep existed -- frog #8, ship #6, airplane #0 -- so no previously reported number changed. Applying one rule uniformly is what makes the spread below a property of the method rather than of whoever read the fronts.

## Per class

| id | class | C* | operators | ACC_r | ACC_f | composite | MIA | f1 | f2 | f3 | S | min |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | #0 | `CLIP|DAMP|MASK` | 92.86 | 0.00 | 92.86 | 100.00 | 0.3892 | 0.0392 | 0.1795 | 221.25 | 15.5 |
| 1 | automobile | #5 | `CLIP|MASK|PRUNE` | 92.68 | 12.70 | 80.91 | 91.70 | 0.3647 | 0.0271 | 0.1210 | 154.57 | 11.9 |
| 2 | bird | #0 | `CLIP|MASK|RANDOM_PRUNE` | 95.12 | 17.90 | 78.10 | 91.70 | 0.2894 | 0.0042 | 0.1819 | 1320.40 | 8.3 |
| 3 | cat | #0 | `DAMP|MASK|PRUNE|RANDOM_PRUNE|RESET` | 94.80 | 9.60 | 85.70 | 95.10 | 0.4477 | 0.0207 | 0.1350 | 168.69 | 10.0 |
| 4 | deer | #2 | `MASK` | 94.20 | 7.80 | 86.85 | 96.00 | 0.3623 | 0.0047 | 0.1176 | 1257.14 | 6.4 |
| 5 | dog | #3 | `MASK` | 95.62 | 3.30 | 92.47 | 99.00 | 0.1403 | 0.0023 | 0.1126 | 4427.91 | 7.0 |
| 6 | frog | #8 | `DAMP|MASK` | 92.52 | 8.30 | 84.84 | 94.20 | 0.3637 | 0.0189 | 0.1451 | 281.76 | 8.2 |
| 7 | horse | #7 | `MASK` | 93.14 | 9.80 | 84.02 | 91.10 | 0.4889 | 0.0302 | 0.1220 | 176.56 | 7.3 |
| 8 | ship | #6 | `MASK|RESET` | 94.58 | 14.00 | 81.34 | 97.00 | 0.2090 | 0.0027 | 0.1128 | 2436.50 | 7.8 |
| 9 | truck | #0 | `CLIP|MASK|QUANTIZE` | 92.91 | 42.10 | 53.80 | 69.60 | 0.4205 | 0.0257 | 0.1166 | 93.99 | 8.2 |

## Mean +/- std over the ten classes

| metric | mean +/- std | min | max |
|---|---|---|---|
| `ACC_r` (%) | **93.84 +/- 1.15** | 92.52 | 95.62 |
| `ACC_f` (%) | **12.55 +/- 11.57** | 0.00 | 42.10 |
| composite (%) | **82.09 +/- 11.00** | 53.80 | 92.86 |
| anchor MIA (%) | **92.54 +/- 8.62** | 69.60 | 100.00 |
| `S` selectivity | **1053.9 +/- 1413.5** | 94.0 | 4427.9 |

Standard deviation is the sample std over classes, which is how the anchor reports its own table.

## Against the anchor paper

Kodge, Saha & Roy, TMLR 07/2024 -- CIFAR-10 / ResNet-18, means +/- std over all ten classes. Our row is now aggregated the same way, so this is the first like-for-like comparison in the project.

| method | `ACC_r` | `ACC_f` | MIA |
|---|---|---|---|
| Original | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 |
| Retraining (gold standard) | 94.81 +/- 0.52 | 0 | 100 +/- 0 |
| NegGrad | 69.89 +/- 10.23 | 0.02 +/- 0.04 | 0 |
| NegGrad+ | 89.91 +/- 1.41 | 0.94 +/- 1.87 | 98.68 +/- 1.42 |
| Tarun et al. 2023 (UNSIR) | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.5 +/- 25.86 |
| Kurmanji et al. 2023 (SCRUB) | 94.79 +/- 0.63 | 0 | 0 |
| Foster et al. 2024 (SSD) | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 |
| Kodge et al. 2024 (the anchor) | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.5 +/- 14.23 |
| **MED-US pure (ours, 10 classes)** | **93.84 +/- 1.15** | **12.55 +/- 11.57** | **92.54 +/- 8.62** |

Every class clears the instance-level selectivity ceiling of **1.158**, measured over 10,534 strategies in the predecessor project. The lowest `S` here is 94.0 (truck).

