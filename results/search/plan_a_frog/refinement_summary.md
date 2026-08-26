# C* refinement — outcome: **REJECTED**

Controlled post-search refinement of the Plan A winner, run once, outside the
evolutionary search. The original C\* remains the main result.

---

## 1. Original Plan A result (C\*)

Front position **8**, operators **DAMP | MASK**, from the completed 10×50 MicroGA
search re-measured at full fidelity.

| metric | C\* | W_ref | gap |
|---|---|---|---|
| `D_f_train` acc / loss | 0.0724 / 5.0153 | 0.0000 / 10.1997 | +0.0724 |
| `D_f_test` acc / loss | **0.0830** / 5.2857 | 0.0000 / 10.1853 | **+0.0830** |
| `D_r_train` acc / loss | 0.9943 / 0.0189 | 1.0000 / 0.0009 | −0.0057 |
| `D_r_test` acc / loss | **0.9252** / 0.2597 | 0.9459 / 0.2191 | **−0.0207** |
| full test acc | 0.8410 | — | — |
| selectivity S | 281.76 | — | — |
| edit cost | 0.1451 | — | — |
| MIA AUC | 0.5237 | — | — |

Search context: best S on the front was **4447.4**, against the instance-level
ceiling of **1.158** across 10,534 strategies.

---

## 2. Refined result

C\* → one clipped forget-ascent step on `D_f` → one retain-repair step on `D_r`.

| metric | C\* (before) | + forget step | **+ retain repair (after)** | W_ref |
|---|---|---|---|---|
| `D_f_train` acc | 0.0724 | 0.1382 | **0.7498** | 0.0000 |
| `D_f_train` loss | 5.0153 | 4.7342 | **0.7374** | 10.1997 |
| `D_f_test` acc | 0.0830 | 0.1550 | **0.6700** | 0.0000 |
| `D_f_test` loss | 5.2857 | 4.9784 | **1.3179** | 10.1853 |
| `D_r_train` acc | 0.9943 | 0.9911 | 0.9995 | 1.0000 |
| `D_r_train` loss | 0.0189 | 0.0277 | 0.0036 | 0.0009 |
| `D_r_test` acc | 0.9252 | 0.9242 | 0.9409 | 0.9459 |
| `D_r_test` loss | 0.2597 | 0.2815 | 0.2177 | 0.2191 |
| full test acc | 0.8410 | 0.8473 | 0.9138 | — |
| selectivity S | 281.76 | 178.32 | 291.81 | — |
| edit cost | 0.1451 | 0.1451 | 0.1451 | — |
| MIA AUC | 0.5237 | 0.5209 | 0.5305 | — |

Gaps to W_ref after refinement: `D_f_test` **+0.6700**, `D_r_test` −0.0050,
`D_f_train` +0.7498, `D_r_train` −0.0005.

### Refinement hyperparameters

| | |
|---|---|
| forget step | SGD gradient **ascent** on cross-entropy over `D_f` |
| retain step | SGD gradient **descent** on cross-entropy over `D_r` |
| learning rate | 1e-4 (both steps) |
| batches | 8 per step, accumulated into **one** optimiser step on the mean gradient |
| epochs | none — one step per stage, not run to convergence |
| gradient-norm clipping | **none** |
| movement budget | 0.02 relative weight norm per step (never engaged; see §4) |
| measured movement | forget step 0.00003, retain step 0.00000 |
| acceptance thresholds | `D_r_test` drop ≤ 0.01; retain losses ≤ 1.25×; edit cost ≤ 0.30 |
| random seed | 42 |

---

## 3. Decision: **REJECTED**

| condition | result |
|---|---|
| 1. `D_f_test` improves below 0.0830 | **FAIL** — rose to 0.6700 |
| 2. `D_r_test` drop ≤ 1 pt from 0.9252 | PASS — rose to 0.9409 |
| 3. no utility collapse in retain losses | PASS — both fell |
| 4. edit cost reasonable | PASS — unchanged at 0.1451 |

The refined checkpoint was **not written to disk**. `refinement.json` records the
full run.

---

## 4. Reason — and a defect in the refinement procedure

The rejection matches the stated failure mode *"retain repair cancels the forget
improvement"*, but the mechanism is **not** the gradient steps.

**`edit_cost` is 0.1451 at all three stages, identical to four decimal places.**
The weights did not meaningfully change, yet `D_f_test` accuracy went 0.0830 →
0.6700 and forget loss collapsed 5.0153 → 0.7374. The model recovered most of its
frog knowledge without its weights moving.

The cause is **BatchNorm running statistics**. `one_step` calls `model.train()`,
which puts BatchNorm into training mode, so every forward pass updates
`running_mean` / `running_var`. Eight batches of `D_r` in train mode re-estimated
those statistics from retain data and substantially undid what the DAMP|MASK edit
had achieved on the forget class.

Two consequences, both of which make this run's *procedure* unsound even though
its *verdict* is correct:

1. **The movement budget is blind to it.** `relative_parameter_delta` measures
   `*.weight` tensors only — deliberately, since BN buffers are not edits. So it
   reported movement 0.00003 and 0.00000, `clip_to_budget` had nothing to clip,
   and the guard that was supposed to bound the refinement never engaged.
2. **The gradient steps were never really tested.** At lr 1e-4 their weight
   effect is ~3e-5 relative — negligible. What this run measured is BatchNorm
   drift, not gradient refinement.

The fix is to freeze BatchNorm in eval mode for the duration of both steps (and,
separately, to include buffers in the movement measurement so the budget can see
them). That has **not** been applied here: it changes the refinement procedure,
and the Plan A result does not depend on it.

A secondary observation worth recording: selectivity S is a poor guide in this
regime. The refined model scores S = 291.8, *higher* than C\*'s 281.8, while being
plainly worse at forgetting — S is computed from training-set losses and rewards
the collapse in retain loss. `D_f_test` is the metric that caught this.

---

## 5. Final recommended model for dissertation reporting

**C\* — Plan A front position 8, `DAMP | MASK`, unrefined.**

- checkpoint: reproduce by applying the position-8 chromosome from
  `results/search/plan_a_frog/pareto_front.csv` to
  `results/checkpoints/cifar10_resnet18_seed42_best.pt`
- W_ref: `results/checkpoints/class6_frog_reference_best_dr.pt` (epoch 163)
- headline: `D_f_test` 0.9730 → **0.0830** against W_ref's 0.0000, with
  `D_r_test` 0.9451 → **0.9252**

If a single number is wanted for the strongest *trade-off* rather than the
closest match to W_ref, front position 1 is the better citation: `D_f_test`
0.9730 → 0.3020 for `D_r_test` 0.9451 → 0.9436, i.e. two thirds of forget-class
generalisation removed for 0.15 points of utility, at **S = 4447**.

The refinement adds nothing to report beyond this negative result and the
BatchNorm finding in §4.
