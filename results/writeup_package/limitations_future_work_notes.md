# Limitations and future work -- working notes

Generated 2026-08-30T20:45:08+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Written to be defensible rather than modest. Each limitation states what was done,
what it does not license, and what would fix it.

## 1. CIFAR-10 only, ResNet-18 only

- Every result is CIFAR-10 / ResNet-18, 32x32 images, ten balanced classes,
  one architecture, one `W_0`.
- What it does not license: any claim about scale, about architectures without
  BatchNorm, or about datasets with many more classes or class imbalance. The
  operator library acts on convolutional layer groups and BatchNorm behaviour is
  central to the refinement, so a transformer is not a small extrapolation.
- Mitigating context: this is the anchor paper's primary setting, so the comparison
  is fair even though the coverage is narrow. The anchor itself also reports
  CIFAR-100 and ImageNet; this work does not.
- Fix: CIFAR-100 (or its twenty superclasses) on the same ResNet-18 tests scale in
  the number of classes at moderate cost; VGG-16 or a ViT tests architecture. Both
  need a new `W_0` and a new reference per target class -- roughly 2.3 GPU-hours per
  reference at the protocol used here.

## 2. One seed

- Every search ran at seed 42. `W_0` is a single model at seed 42; all ten `W_ref`
  are seed 42; NSGA-II ran at seed 42.
- What it does not license: reporting `+/- std` over the ten classes as if it were an
  uncertainty estimate. **It is class-to-class spread, not run-to-run variance.**
  Those are different quantities and the write-up must not blur them.
- Two distinct variances are unmeasured, and they cost very different amounts:
  - **Search variance** -- would the evolutionary search find an equally good `C*` on
    a different seed? Cheap to measure: `W_0` and `W_ref` are fixed and the objective
    evaluation is deterministic given a genome, so only the sampler changes.
  - **Training variance** -- would a different `W_0` and a different reference give
    the same result? Expensive: a new reference per class per seed.
- Fix, in order of value per hour: extra search seeds on a small subset of classes
  (best, middle, worst) first; training variance only if a reviewer requires it.

## 3. The literature benchmark is a comparison against reported numbers

- The eight baseline rows in the benchmark table are transcribed from the anchor
  paper's Table 1. **No published unlearning baseline was re-implemented or re-run in
  this harness.**
- What it does not license: a claim that MED-US beats or loses to any specific
  baseline under controlled conditions. The comparison inherits every difference
  between two implementations -- augmentation, MIA attack details, checkpoint
  selection, evaluation subsets.
- What supports it: this project's own `W_0` and `W_ref` measurements land within a
  few tenths of the paper's Original and Retraining rows, which is evidence the two
  harnesses agree on the baselines they share. That is the strongest available
  argument and it is indirect.
- A second concern, worth its own paragraph: the anchor's MIA appears saturated.
  Retraining scores exactly 100.00, and in the same table SCRUB reaches `ACC_f` 0.00
  with MIA 0.00. A metric that pins the gold standard at the ceiling has limited
  discriminative power, and this project's own MIA AUC on the same models sits far
  closer to chance than the anchor MIA implies.
- Fix: implement finetune-on-`D_r`, NegGrad, random-relabel and the anchor's own
  method against this harness, and evaluate every method under one MIA definition.
  A stronger attack (U-LiRA) would address the saturation separately.

## 4. The hybrid is gradient-based post-search refinement

- The hybrid applies one clipped gradient-ascent step on `D_f` and one repair step on
  `D_r` **after** the search finishes. It is not part of the evolutionary method and
  it is not gradient-free.
- What it does not license: quoting the hybrid's `ACC_f` of 7.55 as MED-US's result,
  or placing it in the anchor's Table 1 as a like-for-like row. The anchor's method is
  gradient-free; only the pure table compares fairly.
- It also needs `D_f` and `D_r` at unlearning time and an optimiser step, which
  removes part of the deployment argument for a gradient-free method.
- Honest framing: the hybrid answers a narrower question -- what does one constrained
  gradient step add to a gradient-free solution? The answer is 5 points of `ACC_f` for
  0.12 of retain accuracy. That is a useful result and a separate one.
- Fix: none needed. The separation is the correct treatment and should be maintained
  permanently. What could be added is a comparison against the same two gradient steps
  applied to `W_0` directly, which would show how much of the hybrid's gain comes from
  the search rather than from the gradient.

## 5. Truck remains difficult

- Pure `ACC_f` 42.10, hybrid 30.90, against ten-class means of 12.55 and 7.55.
  Worst class on every headline metric, before and after refinement.
- What is known: the failure is stable and reproducible, not noise, and refinement
  helps truck more than any other class while still leaving it worst.
- What is not known: **why**. Two hypotheses are open -- confusability with the
  automobile class, and lower forget-specific activation contrast -- and neither has
  been tested. The instrument for testing the second already exists in
  `results/analysis/class_structure/`.
- What it does not license: describing MED-US as reliable across classes. It is
  reliable on retention and variable on forgetting, and the variation is large.
- Fix, and the cheapest high-value work outstanding: regress per-class `ACC_f` against
  the activation-contrast statistic and against inter-class confusability. If the ten
  points fall on a line, the failure mode becomes predictable before any unlearning is
  attempted, which turns the weakest result in the dissertation into its most useful
  claim. No new training required.

## 6. No full ablation

- Not run: NSGA-II against random search at equal evaluation budget; operator families
  in isolation; population and generation budget sensitivity; the `class_contrast`
  selector against alternatives; `max_level`.
- What it does not license: the claim that the evolutionary search is necessary. The
  search evaluated roughly 210 to 320 genomes per class, and nothing here rules out a
  uniform random sample of that size performing comparably.
- Related unresolved observation: `MASK` appears in all ten selected candidates. That
  is convergent evidence across ten independent runs, but without an operator ablation
  it stays an observation rather than a demonstrated property.
- Fix, cheap: random sampling at matched evaluation count reuses the entire existing
  harness and changes only the sampler. It is the first question a viva will ask about
  any evolutionary method.

## 7. No runtime optimisation

- Reported runtimes -- roughly 8 minutes of search and 10 minutes of full-fidelity
  re-measurement per class -- are from unoptimised single-GPU code with
  `num_workers: 0`, a batch cap of 3 during search, and no parallel evaluation of the
  population.
- What it does not license: comparing this project's wall-clock against published
  runtimes as though the implementations were equally tuned. The anchor reports a
  one-shot closed-form update; a comparison of seconds would flatter it and mean
  little either way.
- What is fair to claim: the *shape* of the cost. No retraining, no optimiser state,
  no retain-set training loop, and a search that is trivially parallel across the
  population because each genome evaluates independently.
- Fix: parallel population evaluation and non-zero dataloader workers would give a
  large constant-factor gain, but runtime is not a contribution of this dissertation
  and the effort is better spent on items 3, 5 and 6.

## Priority if time is limited

1. **Item 5** -- the per-class regression. No new training, and it converts the
   weakest result into the strongest claim.
2. **Item 6** -- random search at equal budget. Defends the method choice.
3. **Item 3** -- baselines in this harness. The largest scientific gap, and the most
   expensive of the three.
4. **Item 2** -- extra search seeds on a subset of classes.

Items 1, 4 and 7 are best handled in the text as scope statements rather than as
outstanding work.
