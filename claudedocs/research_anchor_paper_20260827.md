# Anchor-Paper Research: Class-Level Machine Unlearning (CIFAR-10 / ResNet-18)

Date: 2026-08-27. Research only — no code changed, no experiments run.

## Recommendation (one paragraph)

Anchor on **Kodge, Saha & Roy, "Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting", TMLR 07/2024**
(https://openreview.net/forum?id=BmI5p6wBi0, arXiv https://arxiv.org/abs/2312.00761, code
https://github.com/sangamesh-kodge/class_forgetting). It is the only paper in this shortlist whose *primary* setting
is exactly ours — CIFAR-10 + ResNet-18 + **single-class** forgetting, gradient-free, no retraining, deployable post-hoc —
and its Table 1 already reports Original, **Retrain (gold standard)**, NegGrad, NegGrad+, UNSIR (Tarun et al. 2023),
SCRUB (Kurmanji et al. 2023) and SSD (Foster et al. 2024) in one place, aggregated as mean±std **over all 10 target
classes**. Adopting it costs us almost nothing conceptually (our f1/f2/f3 objectives and S/MIA diagnostics stay) but
forces two protocol changes: (i) report `ACC_r`, `ACC_f`, `MIA` with their definitions plus their composite
`ACC_r x (1 - ACC_f)` (verified from their code -- MIA is NOT a factor in it), and (ii) **sweep all 10 classes rather than only frog(6)**, because their numbers are
10-class averages and a frog-only result cannot be dropped into their table. Backup anchor: **SalUn (ICLR 2024
Spotlight)** for the widely-used UA/RA/TA/MIA/RTE protocol and a larger gradient-based baseline suite; second backup
**Boundary Unlearning (CVPR 2023)**, which is class-only on CIFAR-10 but has a narrower baseline set. Flag now, before
running anything: their retrain reference gets `ACC_f = 0` and the best methods ~`0.03%`, whereas our corrected
candidate is at **2.70%** forget-class test accuracy — roughly two orders of magnitude away — so "0.027" is a weak
number *in their units*, and closing that gap is the single most important experimental target.

## Comparison table

| | Deep Unlearning (Kodge et al.) | SSD (Foster et al.) | SalUn (Fan et al.) | Bad Teaching (Chundawat et al.) | SCRUB (Kurmanji et al.) | Boundary Unlearning (Chen et al.) | UNSIR (Tarun et al.) |
|---|---|---|---|---|---|---|---|
| Citation | Kodge, Saha, Roy. *Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting*. TMLR, 07/2024 | Foster, Schoepf, Brintrup. *Fast Machine Unlearning without Retraining through Selective Synaptic Dampening*. AAAI 2024, 12043-12051 | Fan, Liu, Zhang, Wong, Wei, Liu. *SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency...*. ICLR 2024 (Spotlight) | Chundawat, Tarun, Mandal, Kankanhalli. *Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher*. AAAI 2023 | Kurmanji, Triantafillou, Hayes, Triantafillou. *Towards Unbounded Machine Unlearning*. NeurIPS 2023 | Chen, Gao, Liu, Peng, Wang. *Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision Boundary*. CVPR 2023 | Tarun, Chundawat, Mandal, Kankanhalli. *Fast Yet Effective Machine Unlearning*. arXiv:2111.08947 |
| Paper link | https://openreview.net/forum?id=BmI5p6wBi0 / https://arxiv.org/abs/2312.00761 | https://ojs.aaai.org/index.php/AAAI/article/view/29092 / https://arxiv.org/abs/2308.07707 | https://arxiv.org/abs/2310.12508 | https://arxiv.org/abs/2205.08096 | https://arxiv.org/abs/2302.09880 | https://arxiv.org/abs/2303.11570 | https://arxiv.org/abs/2111.08947 |
| Code | https://github.com/sangamesh-kodge/class_forgetting | https://github.com/if-loops/selective-synaptic-dampening | https://github.com/OPTML-Group/Unlearn-Saliency | https://github.com/vikram2000b/bad-teaching-unlearning | https://github.com/Meghdad92/SCRUB | not linked in the sources retrieved | https://github.com/vikram2000b/Fast-Machine-Unlearning |
| Datasets | CIFAR-10, CIFAR-100, ImageNet | CIFAR-10 (random-sample only), CIFAR-20, CIFAR-100, PinsFaceRecognition | CIFAR-10 (primary), CIFAR-100, SVHN, Tiny-ImageNet | CIFAR-10, CIFAR-100 (+ other domains) | CIFAR-10/100, Lacuna-10/100, CIFAR-5/Lacuna-5 | CIFAR-10, VGGFace2 | CIFAR-10, CIFAR-100, VGGFace-100 |
| Architectures | **ResNet-18**, VGG11-BN, ViT-B/14 | **ResNet-18**, ViT | **ResNet-18**, VGG-16, Swin-T | ResNet-18 (+ AllCNN etc.) | **ResNet-18**, All-CNN | CNN / ResNet on CIFAR-10 | ResNet-18, AllCNN, MobileNetV2, ViT |
| Forget setting | **Class-level** (single + multi-class, incl. sequential) | Sub-class + full-class (CIFAR-20/100, faces); CIFAR-10 is *random-sample* | Both: class-wise **and** random 10%/50% | Both: class + random subset | Both: class (10% = one full class) + selective 0.25-2% | **Class-level only** | Class-level (1, 2, 4, 7 classes) |
| Forget classes stated | **All 10 CIFAR-10 classes, each in turn**; multi-class = 5 superclass-related classes; sequential removal of classes 0-9 | CIFAR-20 subclasses "rocket", "sea"; CIFAR-100 "rocket", "mushroom" | Class-wise results deferred to Appendix B.2; class selection/averaging not visible in the HTML we could fetch | Class forgetting on CIFAR-10/ResNet-18; specific index not stated in retrieved sources | **Class 5** on CIFAR-10 | Not stated in retrieved sources | **Class 0** single-class; classes 1-2 / 3-6 / 3-9 multi-class |
| Baselines | Original, **Retrain**, NegGrad, NegGrad+, UNSIR, SCRUB, SSD | Baseline, Finetune, **Retrain**, Bad Teacher, Amnesiac, UNSIR, Fisher forgetting | **Retrain**, FT, RL, GA, IU, BS, BE, l1-sparse | Retrain, Amnesiac, UNSIR, Fisher, finetune | **Retrain**, Original, Finetune, NegGrad+, Fisher, NTK, CF-k, EU-k, Bad-T | Retrain, finetune, NegGrad, random-label, Fisher | Retrain, finetune, NegGrad, Fisher |
| Metrics | `ACC_r`, `ACC_f`, `MIA` (SVM on target-class confidences), composite `metric = ACC_r x (1 - ACC_f)`; **U-LiRA** shadow-model attack as a stronger check | Retain acc, forget acc, MIA (logistic regression), wall-clock seconds | **UA** (=1-forget acc), **RA**, **TA**, **MIA**, **RTE** (minutes) | Forget/retain/test acc, MIA, **ZRF** (zero-retrain-forgetting) | Forget/retain/test error, IC-Err, Fgt-Err, MIA accuracy | Forget/retain acc, MIA, speed-up (~17x vs retrain) | Forget/retain acc, **relearn time**, layer-wise weight distance, prediction distributions |
| Retrain-from-scratch | Baseline / gold standard, **not** inside the method | Gold standard reference only | Gold standard reference only | Explicitly avoided (ZRF is retrain-free) but retrain still reported | Oracle reference | Reference; method imitates the retrained decision boundary | Gold standard reference |
| Gradient-free method? | **Yes** — SVD on layerwise activations + closed-form weight update | Effectively yes (Fisher importance + dampening, no gradient-descent loop) | No — gradient-based saliency mask + RL fine-tuning | No — student/teacher distillation | No — SGD min/max distillation | No — gradient-based boundary shifting | No — impair/repair fine-tuning with error-maximising noise |
| Deployable w/o retraining? | Yes, one-shot; <4% of training data (ImageNet: ~1.5k samples) | Yes, one-shot post-hoc, no retain-set training | No — needs fine-tuning epochs | No | No | No | No |
| Closeness to our setup | **Very high** — same dataset, arch, single-class task, same "gradient-free, no retraining" positioning | Medium-high — same arch, but its CIFAR-10 experiment is random-sample; class forgetting is CIFAR-20/100 | Medium — same dataset/arch, but headline results are random forgetting and the method needs fine-tuning | Medium — same dataset/arch class forgetting, different method family | Medium — one class (5) at 10% forget rate; our f1 (JS to W_ref) is spiritually close to SCRUB's KL distillation | Medium-high on task, low on protocol breadth | Medium — class 0 only |
| Exact experiment to compare fairly | Rerun MED-US for **all 10 CIFAR-10 classes**, >=3 seeds, report mean+/-std of `ACC_r`/`ACC_f`/`MIA` under their definitions; drop our row into their Table 1 | Run their repo's ResNet-18 full-class script on **CIFAR-10** (config not shipped — we add it), alpha=10, lambda=1 | Run SalUn's class-wise CIFAR-10 config; report UA/RA/TA/MIA/RTE alongside ours | Run their CIFAR-10 ResNet-18 class-forgetting script; add ZRF to our diagnostics | Run SCRUB with forget-class = each of the 10 classes; report forget/retain/test error + MIA | Reimplement Boundary Shrink/Expanding on CIFAR-10 ResNet-18; report acc + MIA + speed-up | Run their repo on class 0 and on frog(6) |

### The anchor's Table 1 (CIFAR-10, ResNet-18, mean+/-std over all 10 target classes)

Source: https://arxiv.org/html/2312.00761v4

| Method | ACC_r (up) | ACC_f (down) | MIA (up) |
|---|---|---|---|
| Original | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 |
| Retraining (gold) | 94.81 +/- 0.52 | 0 | 100 +/- 0 |
| NegGrad | 69.89 +/- 10.23 | 0.02 +/- 0.04 | 0 |
| NegGrad+ | 89.91 +/- 1.41 | 0.94 +/- 1.87 | 98.68 +/- 1.42 |
| Tarun et al. 2023 (UNSIR) | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.5 +/- 25.86 |
| Kurmanji et al. 2023 (SCRUB) | 94.79 +/- 0.63 | 0 | 0 |
| Foster et al. 2024 (SSD) | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 |
| Kodge et al. (theirs) | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.5 +/- 14.23 |
| **Our corrected candidate (frog only, n=1)** | **92.56** | **2.70** | not yet in their MIA units |

Two things this table makes immediately clear: our `ACC_r` of 92.56 is ~2 points below the field, and our `ACC_f` of 2.70 is
~90x the anchor's 0.03. Both gaps are the story of the next experiments.

### Supporting / evaluation-critique papers worth citing

- **CMIA / "Unlearning Isn't Forgetting"** — Ebrahimpour-Boroojeny, Wang, Sundaram, https://arxiv.org/html/2506.20893.
  Shows class-unlearning evaluations that look only at the forget-class logit miss leakage into *semantically
  neighbouring* retained classes (retrained models misclassify the forgotten class in a structured way). Evaluates 12
  methods (FT, RL, GA, SalUn, BU, l1, SVD, SCRUB, SCAR, l2ul, UAM, + proposed TREW) on MNIST/CIFAR-10/CIFAR-100/
  Tiny-ImageNet with ResNet-18 and VGG-19. **Directly relevant to our `S` selectivity diagnostic** — we should report
  where unlearned frogs get routed.
- **DAMP — "Class Unlearning via Depth-Aware Removal of Forget-Specific Directions"** — Hatami, Aalishah, Monosov,
  https://arxiv.org/abs/2604.15166, CVPR 2026 Workshop on Machine Unlearning for Vision (oral). One-shot, closed-form
  **weight surgery**, gradient-free, MNIST/CIFAR-10/CIFAR-100/Tiny-ImageNet. Closest contemporary competitor to MED-US's
  operator family; warns that apparent forgetting can come from classifier-head suppression, not representational removal.
- **Classification-Head Bias in Class-Level Machine Unlearning**, https://arxiv.org/pdf/2605.08730 — same head-suppression
  concern; worth a paragraph in the evaluation chapter.

## Step-by-step experiment plan (smallest useful next run)

**Phase 0 — protocol freeze (no GPU).**
1. Implement Kodge et al.'s three metrics exactly: `ACC_r` (retain **test** accuracy), `ACC_f` (forget-class **test**
   accuracy), `MIA` = RBF-SVC on true-class confidence, fit on retain-train (member) vs forget-train
   (non-member) and scored on forget-test, plus the composite `ACC_r x (1 - ACC_f)`. Keep our f1/f2/f3 and S/MIA-AUC as *additional*
   columns, not replacements.
2. Sanity-check the two reference rows on our own frog(6) models: `W0` should land near `ACC_r 94.89 / ACC_f 94.89 /
   MIA ~0`; `W_ref` near `ACC_r 94.81 / ACC_f 0 / MIA 100`. If `W_ref` does not reproduce `MIA ~ 100`, our MIA
   implementation differs from theirs and every downstream comparison is invalid.

**Phase 1 — the 10-class sweep (the actual deliverable).**
3. Train the 9 remaining retain-only references `W_ref^(c)` for c != 6. Expensive but unavoidable: their table is a
   10-class average and our f1 is defined against `W_ref`.
4. Run MED-US per class at the *current* search budget (do **not** enlarge to 50x100 yet), 3 seeds each -> 30 runs.
   Report mean+/-std in their format.
5. Apply the corrected BatchNorm-frozen refinement (commit 2da4f23) uniformly across all 10 classes so the row is
   self-consistent.

**Phase 2 — baselines, cheapest first.**
6. **NegGrad** and **NegGrad+** — ~30 lines each, no external repo, and they bracket the trade-off
   (NegGrad collapses retain to 69.89; NegGrad+ reaches 89.91 / 0.94 / 98.68). Do these first.
7. **SSD** — one-shot, gradient-free, closest in spirit to weight surgery. Clone
   `if-loops/selective-synaptic-dampening`, add a CIFAR-10 full-class config (their shipped CIFAR-10 script is
   random-sample), alpha=10, lambda=1. Its published CIFAR-10 row has huge variance (85.76+/-25.76 / 4.37+/-12.79) —
   that instability is a point in our favour if MED-US turns out stable.
8. **The anchor's own SVD method** — `sh ./scripts/our_cifar10.sh` from `sangamesh-kodge/class_forgetting`. This is the
   head-to-head number the professor will ask for.
9. **SCRUB** (`Meghdad92/SCRUB`) and **UNSIR** (`vikram2000b/Fast-Machine-Unlearning`) only if time allows — both are
   already tabulated in the anchor, so citing their published numbers is defensible provided our reproduced
   `Original`/`Retrain` rows match theirs.

**Phase 3 — robustness (thesis-strengthening, optional).**
10. Add **U-LiRA** (used by the anchor) or **CMIA** (arXiv:2506.20893) as a stronger attack, and report where forgotten
    frogs are *routed*. This turns our `S` selectivity diagnostic from a diagnostic into a contribution.

### Answers to the sub-questions

1. **Main anchor:** Deep Unlearning (Kodge, Saha & Roy, TMLR 2024).
2. **Backup anchor:** SalUn (ICLR 2024 Spotlight); Boundary Unlearning (CVPR 2023) as a class-only second backup.
3. **Smallest next experiment:** Phase 0 + Phase 1 — freeze their metrics, verify the `W0`/`W_ref` reference rows, then
   the 10-class x 3-seed MED-US sweep at the current search budget.
4. **Fastest baselines:** NegGrad and NegGrad+ (hours), then SSD and the anchor's SVD method (both one-shot, public code).
5. **Priority:** **other forget classes + baseline comparison first; the 50x100 search last.** A frog-only pair of
   numbers cannot enter any published table, and a bigger search that moves `ACC_f` from 2.70 to ~1.5 is still ~50x worse
   than the 0.03 the anchor reports — so budget is better spent making the result *comparable* than marginally better.
   Enlarge the search only once the 10-class sweep tells us whether 2.70 is a frog-specific artefact or a systematic
   MED-US ceiling.

## Confidence and gaps

- **High confidence:** the anchor's identity, venue, protocol, Table 1 numbers, and baseline list (read from the paper's
  HTML rendering); SSD's, SCRUB's, UNSIR's and Bad Teaching's venues, code links and general setups.
- **Medium confidence:** SSD has *no* full-class CIFAR-10 experiment (its CIFAR-10 run is 100 random samples; class
  forgetting is on CIFAR-20/CIFAR-100) — confirmed from the paper HTML and consistent with the repo's experiment scripts,
  but worth a 5-minute check of `cifar10_*_exps.sh` before we assert it in writing.
- **Not retrieved:** SalUn's exact class-wise CIFAR-10 table (Appendix B.2 did not render); the specific CIFAR-10 class
  index used by Bad Teaching; a code link for Boundary Unlearning. Get these from the PDFs before citing specifics.

## Correction (2026-08-27, after reading the anchor's released code)

An earlier draft of this report stated the anchor's composite score as `ACC_r x (100 - ACC_f) x MIA`.
That is wrong. Verified in `utils.py` of https://github.com/sangamesh-kodge/class_forgetting (branch
`master`): `metric_function(x, y) = x * (1 - y)` with `x = retain_acc`, `y = forget_acc` in [0, 1], logged as
`100 * metric`. `demo.py` writes the same thing on a percentage scale as `retain_acc*(100-forget_acc)/10000`.
**MIA is not a term in the composite.** Their MIA (`SVC_MIA`) is reported separately.
