# Table: benchmark comparison against the anchor's Table 1

Generated 2026-08-30T20:45:08+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

CIFAR-10 / ResNet-18, mean +/- std over all ten target classes -- the same aggregation
the anchor paper uses, which is why the ten-class sweep was run rather than a single
class.

| method | numbers | grad-free | ACC_r (up) | ACC_f (down) | MIA (up) |
|:---|:---|:---:|---:|---:|---:|
| Original | reported | n/a | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 |
| Retraining (gold standard) | reported | n/a | 94.81 +/- 0.52 | 0.00 | 100.00 +/- 0.00 |
| NegGrad | reported | no | 69.89 +/- 10.23 | 0.02 +/- 0.04 | 0.00 |
| NegGrad+ | reported | no | 89.91 +/- 1.41 | 0.94 +/- 1.87 | 98.68 +/- 1.42 |
| UNSIR (Tarun et al. 2023) | reported | no | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.50 +/- 25.86 |
| SCRUB (Kurmanji et al. 2023) | reported | no | 94.79 +/- 0.63 | 0.00 | 0.00 |
| SSD (Foster et al. 2024) | reported | yes | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 |
| Kodge et al. 2024 (anchor) | reported | yes | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.50 +/- 14.23 |
| **Original W_0 (this work)** | measured | n/a | **94.79 +/- 0.29** | **94.79 +/- 2.59** | **0.00 +/- 0.00** |
| **Retraining W_ref (this work)** | measured | n/a | **95.06 +/- 0.57** | **0.00 +/- 0.00** | **100.00 +/- 0.00** |
| **MED-US pure (this work)** | measured | yes | **93.84 +/- 1.15** | **12.55 +/- 11.57** | **92.54 +/- 8.62** |
| **MED-US hybrid (this work)** | measured | no | **93.72 +/- 1.12** | **7.55 +/- 8.87** | **95.05 +/- 6.08** |

## Read this table carefully

The first eight rows are **as reported by the anchor paper**. They were not
re-measured in this harness, and no published unlearning baseline was re-run here.
The comparison is therefore against published numbers, under the anchor's own
protocol and MIA definition, and it inherits whatever differences exist between two
implementations. This repository's own baseline measurements -- `W_0` and `W_ref` --
land within 0.10 and 0.25 points of `ACC_r` of
the paper's Original and Retraining rows respectively, and their `ACC_f` and MIA agree
to within 0.10 and 0.03. That agreement on the shared baselines is the
available evidence that the two harnesses measure the same quantities; it is indirect,
and it is the strongest such evidence this dissertation has.

Source for the reported rows: Kodge, Saha & Roy. Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting. TMLR 07/2024. https://openreview.net/forum?id=BmI5p6wBi0

## What the comparison shows

**Retention is competitive.** Pure MED-US holds `ACC_r` at 93.84,
0.35 below the anchor and inside the range of
the published field. Only NegGrad and SSD are materially worse on retention.

**Forgetting is not.** Pure `ACC_f` of 12.55 sits far above
the anchor's 0.03, and above UNSIR's 10.89 as well -- UNSIR is the nearest published
neighbour, and like this work it has a wide per-class spread rather than uniform
behaviour. The hybrid closes some of the gap, to 7.55,
but is no longer gradient-free and so is not a like-for-like row.

**Privacy is competitive** on the anchor's own MIA: pure 92.54,
hybrid 95.05, against the anchor's 95.50. How much that
certifies is a separate question -- in this same table SCRUB reaches `ACC_f` 0.00 with
MIA 0.00, and Retraining is pinned at exactly 100.00. See the results-chapter notes.
