# MED-US Class Unlearning

Multi-objective evolutionary search for **layer-wise class unlearning** on
CIFAR-10 / ResNet-18. MSc Computer Science with AI dissertation, University of
Nottingham.

`D_f` is one whole CIFAR-10 class. `D_r` is the other nine. **No other split mode
is supported**, and that is a deliberate constraint rather than an omission —
see [Why this project exists](#why-this-project-exists).

---

## Why this project exists

The predecessor project (`Experimental_Studies_2`) searched **instance-level**
unlearning, where `D_f` is a random subset of training images. It searched
exhaustively and found nothing:

| | |
|---|---|
| evaluated strategies | **10,534** |
| operator families | 3 |
| selectors compared | 5 |
| objective formulations | 4 |
| best selectivity `S` ever measured | **1.158** |
| `S` for retraining without `D_f` | **~932** |

`S ≈ 1` means damage to `D_f` and damage to `D_r` were empirically identical —
no strategy was selective, whatever the operator, selector, objective or search
algorithm.

A direct measurement of the network explained why. Per-channel activation
contrast between `D_f` and `D_r`, against a null control built from two disjoint
halves of `D_r`:

| forget set | channels above the noise floor |
|---|---|
| instance-level `D_f` | **0.55%** |
| pure noise (by construction) | 1.00% |

Fewer channels stood out than chance alone produces. `D_f` and `D_r` were the
same ten classes, so the network used the same features — and the same weights —
for both. There was no forget-specific structure for any weight edit to remove.

**That explanation makes a falsifiable prediction:** make `D_f` genuinely
different and the structure should appear. It does. Measured on the original
model across all ten classes:

| class | median SNR | % channels above noise | layer1 | layer2 | layer3 | layer4 | fc |
|---|---|---|---|---|---|---|---|
| ship | 16.09 | 91.22 | 16.09 | 1.77 | 5.06 | 35.12 | 47.54 |
| **frog** | **13.08** | **91.11** | 1.42 | 10.08 | 13.08 | 40.05 | 48.84 |
| airplane | 10.29 | 90.07 | 1.59 | 10.29 | 3.91 | 35.94 | 56.07 |
| … | | | | | | | |
| automobile | 3.90 | 87.87 | 1.49 | 1.39 | 3.90 | 47.00 | 74.25 |

Every class shows 84–91% of channels above the noise floor, against 0.55% for
instance-level. **The structure exists.** This project tests whether the
operators can exploit it.

### Why frog

Not the top class by every metric — ship has a marginally higher median SNR.
Frog was chosen because its structure is the most usable by a *layer-wise*
search: it rises monotonically with depth (1.42 → 10.08 → 13.08 → 40.05 → 48.84)
and it is the only class with **four** layer groups above SNR 10. Ship's median
is carried by `layer1` with a dead `layer2`, which on a water-background class is
most likely low-level colour statistics — a shortcut cue that would make a strong
result easy to dismiss.

---

## The objectives

```
f1 = JS( P_ref(D_f) || P_cand(D_f) )      minimise    bounded by ln 2
f2 = L_r  (retain loss)                    minimise    unbounded
f3 = ||θ − θ₀||₂ / ||θ₀||₂                 minimise    relative edit cost
```

Two deliberate departures from the predecessor project:

**f1 matches a distribution, not a loss.** Its final objective was
`|L_f − L_f(ref)|`, a scalar target on the cross-entropy. Many very different
models share one loss value: a model that confidently relabels every frog as
"cat" can hit the reference's loss exactly while behaving nothing like it. JS is
symmetric (we want agreement, not one-sided coverage), bounded by `ln 2` (one
destroyed candidate cannot flatten every other individual's `f1` under min-max
normalisation), and finite where the KL diverges.

**f3 is an edit cost, not a second reference term.** The predecessor used
`f2 = L_r` alongside `f3 = KL(P_ref(D_f) || P(D_f))`, and the two behaved as
near-duplicates — both punish a damaged model, so a nominally three-objective
search was really a two-objective one. Measured rank correlation against `f2` on
random candidates:

| candidate f3 | Spearman vs f2 |
|---|---|
| **parameter-change norm** (used here) | **+0.36** |
| KL to reference (the old f3) | +0.74 |

Edit cost never reads the data, so it is orthogonal by construction.

The MIA is computed as a **diagnostic only** — members are forget-class *train*
images, non-members are forget-class *test* images — and nothing reads it back
into an objective.

---

## The operator library

Eight gradient-free operators across two chromosome channels:

| channel | operators |
|---|---|
| `editor` | MASK, PRUNE, RANDOM_PRUNE |
| `smoother` | DAMP, NOISE, CLIP, QUANTIZE, RESET |

`REINIT` and `SIGN_FLIP` are **absent from the library**, not merely disabled by
config. They were the most destructive operators in the predecessor's
calibration — SIGN_FLIP took layer4 from 0.988 to 0.211 forget accuracy across
its ladder — and dominated the fronts that turned out to be full of wrecked
models. Excluding them at the library level means no config edit can bring them
back.

`PRUNE` and `RANDOM_PRUNE` pin themselves to data-free selection rules
(`magnitude`, `random`). They are the **controls**: if forget-informed selection
cannot beat them, the "forget-informed" part is doing nothing.

Intensity ladders are copied unchanged from the predecessor's calibrated table.

---

## The chromosome

```
x = (b, g, s, d_g, d_s)          each an integer vector of length L = 6
```

| gene | meaning | range |
|---|---|---|
| `b_i` | is layer group `i` active? | `{0,1}` |
| `g_i` | which editor operator | `0..2` |
| `s_i` | which smoother operator | `0..4` |
| `d_g,i` | editor intensity | `0..5` (0 = OFF) |
| `d_s,i` | smoother intensity | `0..5` |

Layer groups are `stem, layer1, layer2, layer3, layer4, fc`.

---

## Pipeline

```
1.  build the class split          D_f_train / D_r_train / D_f_test / D_r_test
2.  train_class_reference.py       W_ref on D_r_train only
3.  analyse_class_structure.py     D_f vs D_r activations, with a null control
4.  run_plan_a.py                  MicroGA / NSGA-II over the safe operators
5.  evaluate_class_front.py        full-fidelity re-measurement
6.  refine_candidate.py            optional, outside the search
```

### The reference checkpoint

`W_ref` is selected on **`D_r_test` accuracy**, ties broken by `D_r_test` loss,
and the file to use is `*_best_dr.pt`.

Selecting on full-test accuracy — as the predecessor did — is wrong here twice
over. It is diluted (1,000 of the 10,000 test images are frogs `W_ref` never
trained on), and it is backwards: the frog logit is never positively trained but
still fluctuates, so an epoch that places a few more frogs in class 6 scores
*higher*. That rewards the reference for recognising the thing it is supposed
never to have seen. `D_f_test` is logged every epoch as a diagnostic and is
deliberately excluded from selection.

Selecting after the fact from `{best, latest, final}` does not fix it either —
the best `D_r_test` epoch is usually none of those three — so `D_r_test` is
measured every epoch during training.

---

## What "forgotten" means here

**Not 0% accuracy on frogs.** A model at zero has learned to actively avoid the
answer, which is its own detectable signature. `W_ref` never saw a frog and
misclassifies frogs the way any naive model would. The target is therefore the
**gap to `W_ref`**, reported next to every number.

---

## Running it

```bash
python experiments/train_class_reference.py                    # ~5 h on a GTX 1650
python experiments/analyse_class_structure.py                  # ~2 min, no reference needed
python experiments/run_plan_a.py --config search/plan_a_frog_smoke.yaml
python experiments/run_plan_a.py --config search/plan_a_frog.yaml
python experiments/evaluate_class_front.py \
    --front results/search/plan_a_frog/pareto_front.csv
pytest tests/ -q
```

---

## What was carried over, and what was not

Copied from the predecessor after verification: the CIFAR ResNet-18, the loader
and transform handling, checkpoint save/load, the layer-group registry, the eight
safe gradient-free operators, the class-activation selector, NSGA-II and the
chromosome, and the config loader.

Deliberately **not** carried over: the instance-level SEC as the main evaluator,
random-instance splitting in any form, the `loss_kl` objective, full-test
checkpoint selection for the reference, the gradient-based operator families, and
the four-way objective-mode branching that made it possible to run a search whose
objectives did not mean what the report said they meant.
