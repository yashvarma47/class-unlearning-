# Ship (class 8) BN-frozen refinement -- ACCEPTED

**This is a post-search HYBRID result, not pure gradient-free.** One clipped gradient-ascent
step on `D_f` followed by one repair step on `D_r`, applied to `C*_ship` *outside* the
evolutionary search. The pure result in `results/search/plan_a_ship/` is untouched and remains
the gradient-free claim.

Run 2026-08-29, GTX 1650. Same corrected procedure and identical defaults to the accepted frog
refinement: forget_lr 1e-4, retain_lr 1e-4, 8 batches per step, one optimiser step per stage on
the mean gradient, movement budget 0.02, seed 42.

## BatchNorm was frozen, and it is verified rather than assumed

20 BatchNorm modules asserted in eval mode before either step, and **buffer movement measured at
exactly 0.000000** with 0 counter changes. This is the defect that sank the first frog attempt:
the model was left in `train()` mode, eight batches of `D_r` re-estimated `running_mean` and
`running_var`, and that quietly undid the operator edit -- frog `D_f_test` went 0.0830 -> 0.6700
while the weights moved by 3e-5 and every weight-based guard reported success. The acceptance
rule now refuses any run where buffers moved at all.

## Before and after

| stage | `D_f_train` | `D_f_test` | `D_r_train` | `D_r_test` | S | dW | dBN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C*_ship` (pure) | 0.1552 | 0.1400 | 0.9997 | 0.9458 | 2436.50 | 0.000000 | 0.000000 |
| + forget step | 0.0590 | 0.0470 | 0.9997 | 0.9449 | 2828.63 | 0.000409 | 0.000000 |
| **+ retain repair (final)** | **0.0590** | **0.0470** | **0.9997** | **0.9449** | **2832.90** | **0.000409** | **0.000000** |
| `W_ref_ship` (target) | 0.0000 | 0.0000 | 1.0000 | 0.9502 | -- | -- | -- |

Losses, same order: `D_f_train` 3.8408 -> 5.2394, `D_f_test` 4.4004 -> 5.7614,
`D_r_train` 0.0027 -> 0.0030, `D_r_test` 0.2060 -> 0.2071. The retain losses are essentially
flat, which is the check that matters -- retain accuracy can hold while the model quietly
becomes badly calibrated on `D_r`, and here it did not.

## Full metric comparison

| metric | pure `C*_ship` | **refined (hybrid)** | change |
|---|---:|---:|---|
| `ACC_r` (%) | 94.5778 | **94.4889** | -0.089 |
| `ACC_f` (%) | 14.0000 | **4.7000** | **-9.30** |
| composite (%) | 81.3369 | **90.0479** | **+8.71** |
| anchor MIA (%) | 97.0000 | **99.1000** | **+2.10** |
| `f1` JS to `W_ref` | 0.2090 | **0.1631** | -0.046 |
| `f2` retain train loss | 0.0027 | 0.0030 | +0.0003 |
| `f3` edit cost | 0.1128 | 0.1128 | unchanged |
| `S` selectivity | 2436.50 | **2832.90** | +396 |
| our MIA AUC | 0.5407 | 0.5408 | +0.0001 |
| parameter movement vs `C*` | -- | 0.000409 | budget 0.04 |
| **BatchNorm buffer movement** | -- | **0.000000** | must be 0 |

`f3` is unchanged because it measures the *operator* edit against `W_0`; the refinement's own
movement is reported separately as parameter movement, and at 0.000409 it is 1% of the budget.

## Acceptance rule -- all six conditions

| condition | verdict |
|---|---|
| forget improved on `D_f_test` | PASS (0.1400 -> 0.0470) |
| `D_r_test` drop <= 0.010 | PASS (0.0009) |
| no utility collapse (retain losses <= 1.25x) | PASS |
| edit cost <= 0.30 | PASS (0.1128) |
| parameter movement <= 0.0400 | PASS (0.000409) |
| BatchNorm buffers unchanged | PASS (0.000000, 0 counters) |

## Against frog, and against the anchor

| | frog pure | frog refined | ship pure | **ship refined** |
|---|---:|---:|---:|---:|
| `ACC_r` (%) | 92.52 | 92.56 | 94.58 | **94.49** |
| `ACC_f` (%) | 8.30 | 2.70 | 14.00 | **4.70** |
| composite (%) | 84.84 | 90.06 | 81.34 | **90.05** |
| anchor MIA (%) | 94.20 | 96.30 | 97.00 | **99.10** |
| `S` | 281.76 | 297.25 | 2436.50 | **2832.90** |

Ship's refinement is proportionally the better trade: 9.3 points of forgetting for 0.089 points
of retain accuracy, against frog's 5.6 for 0.03 in the other direction on a lower base. The two
refined models land at almost the same composite (90.05 vs 90.06) from very different pure
starting points.

Against the anchor's Table 1 (CIFAR-10 / ResNet-18, 10-class means):

* `ACC_r` **94.49** is above Kodge et al.'s 94.19 and every baseline except SCRUB (94.79).
* MIA **99.10** is the highest number in the table -- above NegGrad+ (98.68 +/- 1.42) and
  Kodge (95.5 +/- 14.23).
* `ACC_f` **4.70** is still the weak axis against Kodge's 0.03, but it is now in the same range
  as SSD (4.37 +/- 12.79) and clearly better than UNSIR (10.89 +/- 8.79). Pure gradient-free is
  14.00; this number is only reachable with the hybrid step.

The honest framing for the write-up: the **pure** claim is `ACC_f` 14.00 with MIA 97.00, and the
**hybrid** claim is `ACC_f` 4.70 with MIA 99.10. They are different methods and must be reported
as such -- the anchor's own method is gradient-free, so only the pure row is a like-for-like
comparison with it.

## Files

`refined_best.pt` (43 MB, git-ignored -- LFS is not approved beyond the frog chain),
`refined_best.json`, `refinement.json`, this summary.
