# Results chapter -- working notes

Generated 2026-08-30T20:45:08+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Bullet points to write from, not prose to paste. Every number traces to a table in
this package.

## Reference model quality

- All **10 of 10** classes have a validated retain-only reference `W_ref`; none was
  missing, none failed validation.
- `D_f_test` accuracy is **0.0000 for every one**. A model that never trained on a
  class does not classify it -- this is the correctness condition, and it held
  without exception.
- `D_r_test` accuracy across the ten: **0.9506 +/- 0.0057**
  (min 0.9423 horse, max 0.9612 cat). Tight, so no class has an
  unusually weak or strong gold standard that could distort its unlearning result.
- Identical protocol for all ten: 200 epochs, seed 42, split 5,000 / 45,000 / 1,000 / 9,000.
- Selection was on `D_r_test` accuracy with `D_r_test` loss as tie-breaker.
  `D_f_test` was logged every epoch but **never** influenced selection.
- Every split inside every imported zip was byte-compared against the
  version-controlled local split before import; sha256 recorded for all ten.
- Training was distributed across three people and seven Kaggle bundles plus one
  local run. Say so in the write-up -- it is a reproducibility fact, not a footnote.

## Pure MED-US results

- Ten classes, one selection rule: `C*` maximises `ACC_r x (1 - ACC_f)`, the anchor
  paper's own metric function.
- Aggregate: `ACC_r` **93.84 +/- 1.15**,
  `ACC_f` **12.55 +/- 11.57**,
  composite **82.09 +/- 11.00**,
  MIA **92.54 +/- 8.62**.
- Retention is stable, forgetting is not. `ACC_r` varies by 3.1 points across all ten
  classes; `ACC_f` varies by 42.1.
- Selectivity `S` = **1053.88 +/- 1413.47**, min 93.99 (truck),
  max 4427.91 (dog). Compare against the predecessor project's instance-level ceiling
  of **1.158** across 10,534 strategies -- three orders of magnitude, and the central
  evidence that class-level forget sets have structure instance-level ones do not.
- No search failed. Ten runs, zero failures, fronts of ten members each.
- Cost: roughly 8 minutes of search plus 10 minutes of full-fidelity re-measurement
  per class, with no retain-set training loop and no optimiser state.

## Hybrid refinement results

- **Nine eligible classes, nine accepted, zero rejected.** Airplane is a deliberate
  no-op: pure `ACC_f` already 0.00 at MIA 100.00.
- Aggregate: `ACC_r` **93.72 +/- 1.12**,
  `ACC_f` **7.55 +/- 8.87**,
  composite **86.66 +/- 8.52**,
  MIA **95.05 +/- 6.08**.
- `ACC_f` falls 5.00 points of mean for **0.12** of retain accuracy. Composite rises
  4.58, MIA rises 2.51.
- The standard deviation narrows on every metric. The refinement helps the weak
  classes most, which is why the mean moves more than any single strong class does.
- BatchNorm buffer movement is **exactly 0.000000** on all nine, with zero counter
  changes. Parameter movement 0.000303 to 0.000420 against a 0.0400 budget.
- The BN-frozen guard is not decoration. An earlier unfrozen attempt passed every
  weight-based check while `D_r` batches silently re-estimated the running statistics
  and undid the operator edit. Buffer movement is the only check that catches it.

## Benchmark comparison

- Compared against the anchor's Table 1 (CIFAR-10 / ResNet-18, ten-class mean), which
  is why the sweep covers all ten classes rather than one.
- **Retention is competitive.** Pure `ACC_r` 93.84 against
  the anchor's 94.19 and Retraining's 94.81 -- about a point back, inside the field.
- **Forgetting is not.** Pure `ACC_f` 12.55 against the
  anchor's 0.03. This is the honest headline and it should be stated plainly rather
  than buried.
- **Privacy is competitive** on the anchor's MIA: pure 92.54,
  hybrid 95.05, anchor 95.50.
- Caveat that must appear in the text: the eight literature rows are **as reported**,
  not re-measured here. No published baseline was re-run in this harness.
- Supporting evidence that the harnesses agree: this project's own `W_0` and `W_ref`
  measurements land within a few tenths of the paper's Original and Retraining rows.
- Worth one sentence of scepticism about the MIA metric itself: in the same table
  SCRUB reaches `ACC_f` 0.00 with MIA 0.00, and Retraining scores exactly 100.00.
  A metric where the gold standard is pinned at the ceiling is saturated.

## Class-wise variation

- This is the most interesting result in the chapter and deserves its own section.
- Best: **airplane**, `ACC_f` 0.00, composite 92.86, MIA 100.00 -- the only class that
  matches the retraining reference exactly on forgetting.
- Worst: **truck**, `ACC_f` 42.10, composite 53.80, MIA 69.60.
- Second tier: dog 3.30, deer 7.80, frog 8.30, horse 9.80, cat 9.60.
- Third tier: automobile 12.70, ship 14.00, bird 17.90.
- The ordering is not random noise across a uniform method -- it is stable, wide, and
  the same classes stay weak after refinement. Truck is worst pure and worst hybrid;
  airplane is best pure and best hybrid.
- The natural reading: how much forget-specific structure a class has determines how
  well weight editing can remove it. The class-structure measurement in
  `results/analysis/class_structure/` is the instrument for testing that, and the
  regression against per-class `ACC_f` has not been run.
- State it as an observation supported by ordering, not as a demonstrated
  correlation, until that regression exists.

## Truck as the weakest class

- Pure `ACC_f` 42.10, against a ten-class mean of 12.55 --
  more than 2.5 standard deviations out.
- Also the weakest on every other metric: composite 53.80 (next worst 78.10), MIA
  69.60 (next worst 91.10), selectivity 93.99 (next worst 154.57).
- Refinement helps truck **more than any other class**: -11.20 `ACC_f`, +10.24
  composite, +9.50 MIA. Largest absolute gain in the sweep.
- And it is still the worst class afterwards, at 30.90 against a 7.55 mean.
  Refinement narrows the gap; it does not close it.
- Truck selects `CLIP|MASK|QUANTIZE` at front position #0 -- the position that
  maximises the composite, on a front whose best composite is 38 points below every
  other class's.
- Do not apologise for this row. A method with a stable, identifiable failure mode is
  more useful than one that fails unpredictably. Report it, and say what is not yet
  known: whether truck's difficulty is confusability with automobile, or lower
  activation contrast, or both.

## `MASK` in every selected pure candidate

- `MASK` appears in the selected `C*` for **10 of 10** classes.
- Operator frequency across the ten selected candidates: `MASK` 10, `CLIP` 4, `DAMP` 3, `PRUNE` 2, `RANDOM_PRUNE` 2, `RESET` 2, `QUANTIZE` 1. `MASK` is the
  only operator present in every one; the next most frequent appears in 4.
- 3 classes select `MASK` alone: deer, dog, horse.
- Full operator sets, in class order: airplane `CLIP\|DAMP\|MASK`, automobile `CLIP\|MASK\|PRUNE`, bird `CLIP\|MASK\|RANDOM_PRUNE`, cat `DAMP\|MASK\|PRUNE\|RANDOM_PRUNE\|RESET`, deer `MASK`, dog `MASK`, frog `DAMP\|MASK`, horse `MASK`, ship `MASK\|RESET`, truck `CLIP\|MASK\|QUANTIZE`.
- This is a search **finding**, not a design choice. `MASK` was one of several
  operators available at equal cost, and the search converged on it independently in
  ten separate runs.
- Consistent with the mechanism the project argues for: if class-specific information
  is concentrated in identifiable channels, zeroing those channels is the operator
  that removes it, and the magnitude-based alternatives are blunter.
- Be careful with the strength of the claim. Ten runs at one seed is convergent
  evidence, not proof; an operator ablation would settle it and has not been run.

## Why pure and hybrid must be reported separately

- They are different methods. Pure MED-US applies no gradient at any point; the
  hybrid applies two gradient steps after the search finishes.
- The anchor paper's method is **gradient-free**. Only the pure table is a
  like-for-like comparison with its Table 1.
- Merging them, or quoting the hybrid's 7.55 as
  "MED-US", would overstate what the gradient-free method achieves by 5 points of
  `ACC_f` and 4.6 of composite.
- The temptation is real and worth naming in the text: the hybrid is better on every
  headline metric. That is exactly why the separation has to be explicit and
  permanent rather than left to the reader.
- Practical rule for the write-up: the pure table is the result. The hybrid is a
  clearly labelled extension answering a different question -- what does one
  constrained gradient step add to a gradient-free solution.
