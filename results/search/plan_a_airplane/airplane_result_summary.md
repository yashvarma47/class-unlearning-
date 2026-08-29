# Airplane (class 0) Plan A -- pure gradient-free result

Run 2026-08-29 on a GTX 1650. Search 15.5 min, full-fidelity re-measurement 10.7 min, anchor
metrics ~11 min. **No refinement was run.** Every number here is pure gradient-free weight
surgery: no gradient step was ever applied.

Third class of the sweep, and the first one where pure MED-US reaches the retrain gold standard.

---

## 1. The headline

`C*_airplane` is Pareto-front member **#0**, operators **`CLIP|DAMP|MASK`**.

| | `W_0` | `W_ref_airplane` | **`C*_airplane`** | best-S (#6) | front #7 |
|---|---:|---:|---:|---:|---:|
| operators | -- | -- | **`CLIP\|DAMP\|MASK`** | `DAMP\|MASK` | `DAMP\|MASK\|RESET` |
| **`ACC_r`** (%) up | 94.68 | 95.16 | **92.86** | 94.70 | 94.39 |
| **`ACC_f`** (%) down | 95.80 | 0.00 | **0.00** | 25.70 | 10.90 |
| **composite** (%) up | 3.98 | 95.16 | **92.86** | 70.36 | 84.10 |
| **anchor MIA** (%) up | 0.00 | 100.00 | **100.00** | 93.30 | 98.00 |
| `f1` JS to `W_ref` down | 0.6898 | 0.0000 | 0.3892 | 0.3800 | 0.3500 |
| `f2` retain train loss down | 0.0011 | 0.0009 | 0.0392 | 0.0045 | 0.0096 |
| `f3` edit cost down | 0.0000 | n/a | 0.1795 | 0.1138 | 0.1462 |
| **`S`** selectivity up | n/a | n/a | 221.25 | **802.84** | 439.69 |
| our MIA AUC | 0.5961 | 0.5202 | 0.5637 | 0.5736 | 0.5727 |
| `D_f_train` acc / loss | 1.0000 / 0.0010 | 0.0000 / 9.8136 | **0.0000 / 8.4305** | 0.3008 / 2.6842 | 0.1298 / 3.7503 |
| `D_f_test` acc / loss | 0.9580 / 0.1475 | 0.0000 / 9.9663 | **0.0000 / 8.6945** | 0.2570 / 3.4995 | 0.1090 / 4.4464 |
| `D_r_train` acc / loss | 1.0000 / 0.0011 | 1.0000 / 0.0009 | 0.9926 / 0.0392 | 0.9998 / 0.0045 | 0.9991 / 0.0096 |
| `D_r_test` acc / loss | 0.9468 / 0.2040 | 0.9516 / 0.1973 | 0.9286 / 0.2465 | 0.9470 / 0.1811 | 0.9439 / 0.2000 |

**`ACC_f` = 0.0000 and anchor MIA = 100.00.** Both are exactly the retrain gold standard's
values. On the anchor protocol's two forgetting metrics, this pure gradient-free model is
indistinguishable from a model retrained from scratch without airplanes -- and it forgets on
`D_f_train` too (0.0000), not just on held-out data.

Unlike ship, no judgement call was needed: `evaluate_class_front.py`'s automatic rule and the
composite both select #0, so the reported `C*` is the automatic one. Members #8 and #9 duplicate
#6 and #2 respectively -- different chromosomes, identical metrics.

The cost is retain accuracy: 92.86, which is 2.30 points below its own reference. That is the
worst retain figure of the three classes, and it is the honest price of driving `ACC_f` to zero.
Front #7 is the alternative trade -- `ACC_r` 94.39 for `ACC_f` 10.90 -- and it is measured in
`airplane_anchor_metrics.csv` so the shape of the trade-off is on record rather than asserted.

---

## 2. Three classes, three different answers -- all pure

| | frog `C*` (#8) | ship `C*` (#6) | **airplane `C*` (#0)** |
|---|---:|---:|---:|
| operators | `DAMP\|MASK` | `MASK\|RESET` | `CLIP\|DAMP\|MASK` |
| `ACC_r` (%) | 92.52 | **94.58** | 92.86 |
| `ACC_f` (%) | 8.30 | 14.00 | **0.00** |
| composite (%) | 84.84 | 81.34 | **92.86** |
| anchor MIA (%) | 94.20 | 97.00 | **100.00** |
| `f1` JS to own `W_ref` | 0.3637 | **0.2090** | 0.3892 |
| `f2` retain train loss | 0.0189 | **0.0027** | 0.0392 |
| `f3` edit cost | 0.1451 | **0.1128** | 0.1795 |
| `S` | 281.76 | **2436.50** | 221.25 |
| retain gap to own `W_ref` | -2.07 | **-0.44** | -2.30 |
| best `S` on the front | 4447.42 | 2436.50 | 802.84 |
| median `S` on the front | -- | 15.75 | **305.44** |

Three classes, three genuinely different operator combinations, and every one of them clears the
instance-level ceiling of **1.158** measured over 10,534 strategies. Frog and ship confirmed the
class-structure explanation; airplane shows the trade-off curve extends all the way to the gold
standard when the search is allowed to pay for it in retain accuracy.

Airplane's front is also the healthiest overall: median `S` of 305 against ship's 16, meaning most
members are selective rather than only the best one.

---

## 3. Against the anchor paper

Kodge, Saha & Roy, TMLR 07/2024 -- CIFAR-10 / ResNet-18, Table 1 means +/- std over ten classes.

| method | `ACC_r` up | `ACC_f` down | MIA up |
|---|---|---|---|
| Original | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 |
| Retraining (gold) | 94.81 +/- 0.52 | **0** | **100 +/- 0** |
| NegGrad | 69.89 +/- 10.23 | 0.02 +/- 0.04 | 0 |
| NegGrad+ | 89.91 +/- 1.41 | 0.94 +/- 1.87 | 98.68 +/- 1.42 |
| UNSIR (Tarun 2023) | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.5 +/- 25.86 |
| SCRUB (Kurmanji 2023) | 94.79 +/- 0.63 | 0 | 0 |
| SSD (Foster 2024) | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 |
| Kodge et al. (theirs) | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.5 +/- 14.23 |
| **airplane `C*` (ours, n=1)** | **92.86** | **0.00** | **100.00** |
| *ship `C*` (ours, n=1)* | *94.58* | *14.00* | *97.00* |
| *frog `C*` (ours, n=1)* | *92.52* | *8.30* | *94.20* |

For the first time our `ACC_f` and MIA columns match the gold standard exactly, and beat every
method in the table on MIA including Kodge's own. `ACC_r` 92.86 is the weak axis: above UNSIR
(92.20) and SSD (85.76), below Kodge (94.19) and SCRUB (94.79).

Two caveats that do not go away. Their rows are ten-class means and ours is a single class with
n = 1 -- three classes is still not a mean, and the spread across our three (`ACC_f` 0.00 to
14.00) is exactly why. And a single class where `ACC_f` hits zero is not evidence the method
does so reliably; the sweep is what would settle that.

---

## 4. Run record

| | |
|---|---|
| config | `configs/search/plan_a_airplane.yaml` (population 10 x 50, seed 42) |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class0_airplane_reference_best_dr.pt` |
| split | `results/splits/cifar10_class0_airplane.json` (5000 / 45000 / 1000 / 9000) |
| baseline sanity | PASS -- `W_0` `D_f_test` 0.9580, `W_ref` 0.0000 / 0.9516 |
| evaluation slots | 510 |
| real evaluations | 233 |
| cache hits | 277 |
| failures | **0** |
| non-finite objectives | **0** |
| search wall clock | 15.5 min (mean 4.00 s/individual, min 1.56, max 12.87) |
| full-fidelity wall clock | 10.7 min (mean 64.5 s/member) |
| Pareto front | 10 members |
| best / median `S` | 802.84 / 305.44 |

`S` is `nan` for `W_0` (a ratio against itself) and meaningless for `W_ref` (not an edit of
`W_0`). `W_ref`'s `f3` is blank for the same reason; its distance from `W_0` is recorded
separately in `airplane_anchor_metrics.csv`.

Files: `pareto_front.csv`, `full_fidelity/front_full_fidelity.csv`, `full_fidelity/baselines.json`,
`airplane_anchor_metrics.csv`, `airplane_anchor_metrics.json`, `pareto_front_plan_a_airplane.png`,
`pareto_front_plot_data.csv`, `summary.json`, this report.

---

## 5. What this does and does not license

It does **not** license running the remaining seven classes without thought: airplane is the best
case so far and three classes disagree with each other by 14 points of `ACC_f`. It does mean the
pipeline is proven on three independent classes with zero failures, and that the sweep is now a
matter of GPU time rather than open questions.

A BN-frozen refinement has **not** been run for airplane and is not obviously worth it -- `ACC_f`
is already 0.00, so the refinement's usual job is done. If anything, airplane is the class where
a refinement might be pointed at `ACC_r` instead, which is a different procedure and not one that
exists yet.
