# Ship (class 8) Plan A — pure gradient-free result, and what it says about frog

Run 2026-08-27 on a GTX 1650. Search 7.8 min, full-fidelity re-measurement 10.6 min, anchor
metrics ~11 min. Every ship number in sections 1-5 is *pure* gradient-free
weight surgery: no gradient step was ever applied. A BN-frozen refinement was run later, on
2026-08-29 and is reported separately in section 6 as a HYBRID; it changes nothing in
sections 1-5.

---

## 1. The headline

`C*_ship` is Pareto-front member **#6**, operators **`MASK|RESET`** — a MASK on the stem
(smoother, level 2) and a RESET on `layer4` (editor, level 1).

| | `W_0` | `W_ref_ship` | **`C*_ship`** | front #1 |
|---|---:|---:|---:|---:|
| operators | — | — | **`MASK\|RESET`** | `CLIP\|MASK\|RANDOM_PRUNE` |
| **`ACC_r`** (%) ↑ | 94.60 | 95.02 | **94.58** | 88.73 |
| **`ACC_f`** (%) ↓ | 96.50 | 0.00 | **14.00** | 12.50 |
| **composite** (%) ↑ | 3.31 | 95.02 | **81.34** | 77.64 |
| **anchor MIA** (%) ↑ | 0.00 | 100.00 | **97.00** | 79.30 |
| `f1` JS to `W_ref` ↓ | 0.6906 | 0.0000 | **0.2090** | 0.3053 |
| `f2` retain train loss ↓ | 0.0011 | 0.0011 | **0.0027** | 0.1950 |
| `f3` edit cost ↓ | 0.0000 | n/a | **0.1128** | 0.1860 |
| **`S`** selectivity ↑ | n/a | n/a | **2436.50** | 22.46 |
| our MIA AUC | 0.5336 | 0.5075 | 0.5407 | 0.5316 |
| `D_f_train` acc / loss | 1.0000 / 0.0008 | 0.0000 / 10.3019 | 0.1552 / 3.8408 | 0.1244 / 4.3557 |
| `D_f_test` acc / loss | 0.9650 / 0.1343 | 0.0000 / 10.3602 | 0.1400 / 4.4004 | 0.1250 / 4.7451 |
| `D_r_train` acc / loss | 1.0000 / 0.0011 | 1.0000 / 0.0011 | 0.9997 / 0.0027 | 0.9462 / 0.1950 |
| `D_r_test` acc / loss | 0.9460 / 0.2055 | 0.9502 / 0.1986 | 0.9458 / 0.2060 | 0.8873 / 0.4702 |

`W_ref`'s `f3` is blank because it is an independently trained model, not an edit of `W_0`;
its distance from `W_0` (1.3363) is recorded separately in `ship_anchor_metrics.csv`. `S` is
undefined for `W_0` (a ratio against itself) and meaningless for `W_ref`.

### Why #6 and not #1

`evaluate_class_front.py`'s automatic rule picked **#1**, because that rule minimises the
`D_f_test` gap subject only to `D_r_test > 0.80` — it does not price retain damage above that
floor. #1 buys 1.5 points of extra forgetting for **5.8 points of retain accuracy**, and loses
on every other axis: `f1` 0.3053 vs 0.2090, `f2` 0.1950 vs 0.0027, `f3` 0.1860 vs 0.1128,
`S` 22 vs 2437, anchor MIA 79.3 vs 97.0, composite 77.6 vs 81.3.

#6 is therefore reported as `C*_ship`, and #1 is measured alongside it in
`ship_anchor_metrics.csv` so the choice is auditable rather than silent. Front member **#9**
(`MASK|PRUNE|RESET`) is a duplicate of #6 — a different chromosome, identical metrics to every
decimal place; its extra `PRUNE` on the stem changes nothing the `MASK` had not already done.

---

## 2. Frog vs ship — does the class-level result generalise?

**Yes, and on the axis that matters most it is stronger.** Both are pure gradient-free.

| | frog `C*` (#8) | **ship `C*` (#6)** |
|---|---:|---:|
| operators | `DAMP\|MASK` | `MASK\|RESET` |
| `ACC_r` (%) | 92.52 | **94.58** |
| `ACC_f` (%) | **8.30** | 14.00 |
| composite (%) | 84.84 | 81.34 |
| anchor MIA (%) | 94.20 | **97.00** |
| `f1` JS to own `W_ref` | 0.3637 | **0.2090** |
| `f2` retain train loss | 0.0189 | **0.0027** |
| `f3` edit cost | 0.1451 | **0.1128** |
| `S` | 281.76 | **2436.50** |
| **retain gap to own `W_ref`** | **−2.07 pts** | **−0.44 pts** |

The two runs disagree about *which* trade-off is reachable, and agree about the thing the
project is testing.

* **Selectivity generalises, and improves.** Ship reaches `S` = 2436 against frog's 282, both
  against an instance-level ceiling of **1.158** measured over 10,534 strategies. Three orders
  of magnitude, on a second class, with a different operator pair. The class-structure
  explanation predicted this; it now has two independent confirmations rather than one.
* **Ship preserves utility far better.** Its retain accuracy sits **0.44 points** below its own
  reference, against frog's 2.07 — and `f2`, the retain *loss*, is 7× lower. `ACC_r` 94.58 is
  effectively at the anchor paper's own method (94.19 ± 0.50).
* **Frog forgets harder.** `ACC_f` 8.30 against 14.00. Frog's composite is correspondingly
  higher (84.84 vs 81.34), because the composite is dominated by `ACC_f` once `ACC_r` is
  similar.
* **Best-`S` and `C*` coincided for ship, not for frog.** Frog's highest-`S` member (#1,
  S = 4447) only reached `D_f_test` 0.302 — it was selective but did not forget much. Ship's #6
  is simultaneously the most selective and among the strongest-forgetting members. That is a
  qualitatively better front, not just a better point.
* **Different operators, same conclusion.** `DAMP|MASK` for frog, `MASK|RESET` for ship. The
  result is not an artefact of one operator pair happening to suit one class.

### The references themselves

| | `W_ref_frog` | `W_ref_ship` |
|---|---:|---:|
| `D_f_test` acc | 0.0000 | 0.0000 |
| `D_r_test` acc | 0.9459 | 0.9502 |
| `ACC_r` (%) | 94.59 | 95.02 |
| anchor MIA (%) | 100.00 | 100.00 |
| selected epoch | 163 | 181 |
| trained on | GTX 1650, torch 2.5.1+cu121 | Tesla T4, torch 2.10.0+cu128 |

Both behave exactly as the anchor protocol's gold-standard row must: `ACC_f` = 0 and MIA = 100.
Ship's is 0.43 points stronger on `ACC_r`, which is why ship's `C*` can sit higher without
being *closer* to its target. Note the two were trained on different software stacks with an
identical recipe — worth one line in the methods chapter.

### The refined frog result, clearly labelled

`C*_refined_bn_frozen` (frog: `ACC_r` 92.56, `ACC_f` **2.70**, composite 90.06, MIA 96.30) is a
**hybrid**, not a pure result: `DAMP|MASK` followed by two gradient steps with BatchNorm frozen,
applied *outside* the evolutionary search. It is the best forgetting the project has produced,
and it is not comparable with either pure `C*` on the "gradient-free" claim. The ship refinement has since been run and
accepted -- see section 6. The honest comparison is refined-frog against refined-ship, and
pure against pure.

---

## 3. Against the anchor paper

Kodge, Saha & Roy, TMLR 07/2024 — CIFAR-10 / ResNet-18, their Table 1 means ± std over all ten
classes.

| method | `ACC_r` ↑ | `ACC_f` ↓ | MIA ↑ |
|---|---|---|---|
| Original | 94.89 ± 0.31 | 94.89 ± 2.75 | 0.03 ± 0.03 |
| Retraining (gold) | 94.81 ± 0.52 | 0 | 100 ± 0 |
| NegGrad | 69.89 ± 10.23 | 0.02 ± 0.04 | 0 |
| NegGrad+ | 89.91 ± 1.41 | 0.94 ± 1.87 | 98.68 ± 1.42 |
| UNSIR (Tarun 2023) | 92.20 ± 0.72 | 10.89 ± 8.79 | 61.5 ± 25.86 |
| SCRUB (Kurmanji 2023) | 94.79 ± 0.63 | 0 | 0 |
| SSD (Foster 2024) | 85.76 ± 25.76 | 4.37 ± 12.79 | 87.86 ± 31.21 |
| Kodge et al. (theirs) | 94.19 ± 0.50 | 0.03 ± 0.09 | 95.5 ± 14.23 |
| **ship `C*` (ours, n=1)** | **94.58** | **14.00** | **97.00** |
| *frog `C*` (ours, n=1)* | *92.52* | *8.30* | *94.20* |

Where we stand, honestly:

* **`ACC_r` is competitive.** 94.58 is between SCRUB (94.79) and Kodge (94.19), and above every
  other baseline in the table.
* **MIA is competitive, arguably better.** 97.00 beats Kodge's mean (95.5), SSD (87.86), UNSIR
  (61.5) and SCRUB (0), and sits just under NegGrad+ (98.68).
* **`ACC_f` is where we lose, badly.** 14.00 against Kodge's 0.03 — three orders of magnitude —
  and worse than UNSIR's 10.89, the weakest method in their table on this metric. Frog's 8.30
  is better but still far off. This is the gap that decides whether the method is publishable.

And the standing caveat: their rows are 10-class means; ours are single classes with n = 1.
Two classes is not a mean. Nothing here can enter their table until the sweep is done.

---

## 4. Run record

| | |
|---|---|
| config | `configs/search/plan_a_ship.yaml` (population 10 × 50 generations, seed 42) |
| `W_0` | `results/checkpoints/cifar10_resnet18_seed42_best.pt` |
| `W_ref` | `results/checkpoints/class8_ship_reference_best_dr.pt` |
| split | `results/splits/cifar10_class8_ship.json` (5000 / 45000 / 1000 / 9000) |
| evaluation slots | 510 |
| real evaluations | 230 |
| cache hits | 280 |
| failures | **0** |
| non-finite objectives | **0** |
| search wall clock | 7.8 min (mean 2.03 s / individual) |
| full-fidelity wall clock | 10.6 min (mean 63.5 s / member) |
| Pareto front | 10 members |

The two `nan` values in the `S` column of the full-fidelity table are front members #3 and #4,
which are identity edits (`f3` = 0.00000, metrics identical to `W_0`): `S` is a ratio of deltas
against `W_0`, so it is 0/0 for a candidate that moved nothing. Not a failure.

Files: `pareto_front.csv`, `full_fidelity/front_full_fidelity.csv`, `full_fidelity/baselines.json`,
`ship_anchor_metrics.csv`, `ship_anchor_metrics.json`, `pareto_front_plan_a_ship.png`,
`pareto_front_plot_data.csv`, `summary.json`.

---

## 5. Recommendation

**Proceed to the remaining classes.** Ship is not a weaker second data point — it clears the
instance-level ceiling by a wider margin than frog, with better retain preservation and a
different operator pair. Two classes, two confirmations, no contradiction. The eight remaining
references are the bottleneck (~18 h of Kaggle time), and they are independent of any search,
so they can start immediately.

**Refinement is worth running on ship, and has not been.** The frog refinement moved `ACC_f`
from 8.30 to 2.70 for 0.03 points of `ACC_r` — the single largest improvement in the project.
Ship's `ACC_f` of 14.00 is the weakest number in the ship result and the one the anchor
comparison turns on. The BatchNorm-frozen procedure already exists and is already corrected.
It needs explicit approval, and the result must be labelled **hybrid**, not gradient-free.

---

## 6. Addendum, 2026-08-29 -- BN-frozen refinement was run and ACCEPTED

Section 5 recommended it; it has now been run. **The pure result above is unchanged** -- nothing
in sections 1-5 was rewritten. The refinement is a separate, clearly labelled **hybrid**.

| | pure `C*_ship` | refined (hybrid) |
|---|---:|---:|
| `ACC_r` (%) | 94.58 | 94.49 |
| `ACC_f` (%) | 14.00 | **4.70** |
| composite (%) | 81.34 | **90.05** |
| anchor MIA (%) | 97.00 | **99.10** |
| `S` | 2436.50 | 2832.90 |
| BatchNorm buffer movement | -- | 0.000000 (verified) |

Full detail, acceptance conditions and the frog comparison:
`results/search/plan_a_ship_bn_frozen_refined/refinement_summary.md`.

The `pure` column remains the gradient-free claim. The anchor's own method is gradient-free, so
only that column is a like-for-like comparison with their Table 1.
