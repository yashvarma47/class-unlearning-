# C* refinement, BatchNorm frozen — outcome: **ACCEPTED**

Corrected retry of the post-search refinement, run once on C\* only, outside the
evolutionary search. The Plan A search, objectives, operator set, selector and
Pareto front are unchanged.

This supersedes the earlier attempt in
`results/search/plan_a_frog_refined/`, which was **rejected** and whose entire
effect turned out to be BatchNorm recalibration rather than gradient updates.

---

## 1. Original Plan A result (C\*)

Front position **8**, operators **DAMP | MASK**.

| metric | C\* | W_ref |
|---|---|---|
| `D_f_test` acc | 0.0830 | 0.0000 |
| `D_r_test` acc | 0.9252 | 0.9459 |
| `D_f_train` acc | 0.0724 | 0.0000 |
| `D_r_train` acc | 0.9943 | 1.0000 |
| edit cost | 0.1451 | — |
| MIA AUC | 0.5237 | — |

---

## 2. Refined result (C\*_refined_bn_frozen)

| metric | **C\* (before)** | + forget step | **+ retain repair (after)** | W_ref |
|---|---|---|---|---|
| `D_f_train` acc | 0.0724 | 0.0272 | **0.0276** | 0.0000 |
| `D_f_train` loss | 5.0153 | 6.3465 | **6.3395** | 10.1997 |
| `D_f_test` acc | 0.0830 | 0.0270 | **0.0270** | 0.0000 |
| `D_f_test` loss | 5.2857 | 6.5738 | **6.5671** | 10.1853 |
| `D_r_train` acc | 0.9943 | 0.9932 | **0.9933** | 1.0000 |
| `D_r_train` loss | 0.0189 | 0.0227 | **0.0224** | 0.0009 |
| `D_r_test` acc | 0.9252 | 0.9254 | **0.9256** | 0.9459 |
| `D_r_test` loss | 0.2597 | 0.2674 | **0.2666** | 0.2191 |
| full test acc | 0.8410 | 0.8356 | **0.8357** | — |
| selectivity S | 281.76 | 293.37 | **297.25** | — |
| edit cost (vs W_0) | 0.14505 | 0.14507 | **0.14507** | — |
| parameter movement (vs C\*) | 0.000000 | 0.000422 | **0.000420** | — |
| **BatchNorm buffer movement** | 0.000000 | 0.000000 | **0.000000** | — |
| BatchNorm counters changed | 0 | 0 | **0** | — |
| MIA AUC | 0.5237 | 0.5235 | **0.5235** | — |

**Gaps to W_ref after refinement:** `D_f_test` **+0.0270**, `D_r_test` −0.0203,
`D_f_train` +0.0276, `D_r_train` −0.0067.

### Refinement hyperparameters

| | |
|---|---|
| forget step | SGD gradient **ascent** on cross-entropy over `D_f` |
| retain step | SGD gradient **descent** on cross-entropy over `D_r` |
| learning rate | 1e-4 (both steps) |
| batches | 8 per step, accumulated into **one** optimiser step on the mean gradient |
| epochs | none — one step per stage, not run to convergence |
| gradient-norm clipping | **none** |
| movement budget | 0.02 relative weight norm per step (never engaged — actual 0.00042) |
| **BatchNorm** | **FROZEN** — model held in eval mode for both steps; 20 modules asserted before each step |
| buffer tolerance | 1e-9, and `num_batches_tracked` must not change |
| acceptance thresholds | `D_r_test` drop ≤ 0.01; retain losses ≤ 1.25×; edit cost ≤ 0.30 |
| random seed | 42 |

---

## 3. Decision: **ACCEPTED**

| condition | result |
|---|---|
| 1. `D_f_test` improves below 0.0830 | **PASS** — 0.0830 → 0.0270 |
| 2. `D_r_test` drop ≤ 1 pt from 0.9252 | **PASS** — *rose* to 0.9256 |
| 3. no utility collapse in retain losses | **PASS** — +18% and +2.7%, within the 1.25× bound |
| 4. edit cost reasonable | **PASS** — 0.14505 → 0.14507 |
| 5. parameter movement within budget | **PASS** — 0.00042 against a 0.04 bound |
| 6. BatchNorm buffers unchanged | **PASS** — exactly 0.000000, 0 counters changed |

Checkpoint written to `refined_best.pt` in this directory.

---

## 4. Reason for the decision

With BatchNorm frozen, the refinement does what it was always supposed to do.

**The forget step now works.** `D_f_test` fell 0.0830 → 0.0270 and forget loss
rose 5.0153 → 6.5738, moving toward W_ref's 10.19. The gap to the reference on
held-out forget data narrowed from +0.0830 to **+0.0270** — a two-thirds
reduction in the one number that matters most.

**The retain repair is a genuine no-op, and that is the correct outcome.** It
changed `D_f_test` by 0.0000 and `D_r_test` by +0.0002. C\*'s retain loss was
already 0.0189, so the descent gradient has almost nothing to correct. Critically,
it did **not** cancel the forget improvement — which is exactly what happened in
the rejected run, and is why that failure mode is a stated rejection condition.

**The movement numbers confirm the mechanism.** Parameter movement is 0.00042 and
BatchNorm buffer movement is exactly 0.000000, with no counter changes. The
entire effect came from a small, bounded, measured weight update. In the rejected
run the weights moved 3e-5 and produced a 59-point swing in `D_f_test`; here they
move 14× further and produce a 5.6-point improvement in the right direction. That
contrast is the clearest evidence that the first run was measuring recalibration.

Costs are small and real: full test accuracy fell 0.8410 → 0.8357 (arithmetic —
`D_f_test` is 10% of the test set and dropping frog accuracy lowers the pooled
number), `D_r_train` accuracy fell 0.0010, and retain losses rose slightly. All
sit well inside the acceptance bounds.

MIA is unchanged at 0.5235 (from 0.5237) — near chance before and after, so the
refinement does not make membership more detectable.

**A caveat on S.** Selectivity rose 281.8 → 297.3, and this time it moves in the
same direction as the real result — but it did so in the rejected run too, while
that model was plainly worse. S is computed from training-set losses and is not
trustworthy as the deciding metric in this regime. `D_f_test` and `D_r_test`
carried the decision here; S is reported for continuity only.

---

## 5. Final recommended model for dissertation reporting

**C\*_refined_bn_frozen** — `results/search/plan_a_frog_bn_frozen_refined/refined_best.pt`

- derived from Plan A front position 8 (`DAMP | MASK`) plus one clipped
  forget-ascent step and one retain-repair step, BatchNorm frozen throughout
- W_ref: `results/checkpoints/class6_frog_reference_best_dr.pt` (epoch 163)
- headline: `D_f_test` 0.9730 (W_0) → **0.0270**, against W_ref's 0.0000 —
  a gap of **+0.0270** — while `D_r_test` holds at **0.9256** (W_0: 0.9451)

The refinement must be reported as a **post-search step outside the evolutionary
search**, not as part of the searched strategy. The chromosome describes a
gradient-free weight edit; these two gradient steps are a separate finishing pass.

If a purely gradient-free result is wanted — which is the cleaner claim for the
search contribution — **the original unrefined C\*** stands: `D_f_test` 0.0830 at
`D_r_test` 0.9252, with no gradient steps involved at all.

For the strongest *trade-off* rather than the closest match to W_ref, front
position 1 remains the better citation: `D_f_test` 0.3020 at `D_r_test` 0.9436,
**S = 4447**.
