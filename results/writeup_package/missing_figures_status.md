# Figure status -- what was generated, what was not, and why

Generated 2026-08-30T22:55:44+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

## Generated

All six requested figures were produced, into `results/writeup_package/figures/`.
**7 of 7 present** (verified on disk).

| figure | source | new computation? |
|---|---|---|
| `pure_vs_hybrid_acc_f_by_class.png` | `pure_medus_10_class_table.csv`, `hybrid_medus_10_class_table.csv` | none |
| `pure_vs_hybrid_acc_r_by_class.png` | the same two tables | none |
| `pure_vs_hybrid_composite_by_class.png` | the same two tables | none |
| `operator_frequency_selected_cstar.png` | `pure_medus_10_class_table.csv` | none |
| `benchmark_comparison.png` | `benchmark_comparison_table.csv` | composite derived as `ACC_r x (1 - ACC_f)` |
| `truck_failure_analysis.png` | `truck_prediction_distribution.csv` | **yes -- inference only, below** |

## The one figure that needed computation: `truck_failure_analysis.png`

It was **generated**, not skipped.

The tables record how much truck accuracy survives unlearning; they do not record where
the rest of the images went. Answering that needs a forward pass, so
`experiments/analyse_truck_predictions.py` classifies the 1,000 held-out truck images
with four models and records the full predicted-class distribution of each.

**No model was trained, searched or refined.** `W_0`, `W_ref` and the refined hybrid all
exist as checkpoints. The pure `C*` does not -- the search recorded genomes, not weights
-- so it was reconstructed by replaying its stored chromosome through the same
deterministic operators, which is exactly what `report_anchor_metrics.py` already does
in order to score it. That path verifies the rebuild against the recorded front row and
raises rather than proceeding if anything drifts.

**The reconstruction reproduced the published numbers exactly.** The rebuilt `C*`
classifies 42.10% of truck test images as truck, and the refined checkpoint 30.90% --
the `ACC_f` values already in `pure_medus_10_class_table.csv` and
`hybrid_medus_10_class_table.csv`. That agreement is the check that the figure describes
the published models and not some near neighbour of them.

### What it shows

| model | still truck | top non-truck destination |
|---|---:|---|
| `W_0` | 95.40% | automobile (2.80%) |
| `W_ref` | 0.00% | **automobile (68.40%)** |
| `C*` pure | 42.10% | automobile (16.70%) |
| `C*` hybrid | 30.90% | automobile (20.40%) |

A model that never saw a truck sends 68.4% of them to automobile. Pure MED-US sends only
16.7% there and leaves 42.1% still called truck; the refinement moves a further 11.2
points out, most of which arrive at automobile (20.4%). The failure is a **partial move
along the truck-automobile axis**, not a random scattering -- the reference's destination
is the same destination, reached less far.

This is direct evidence for the confusability reading of the truck result, and it did not
exist before. It belongs in the discussion chapter.

## Bird's `pareto_front_plot_data.csv` -- correction: it was never missing

An earlier note in this package reported bird as having a figure but no plot-data CSV.
**That was wrong, and the record is corrected here.** Bird's file was present and
correct all along; the class whose file does not exist under that name is **frog**.

Bird's CSV was regenerated anyway, by `plot_pareto_front_class.py` -- documented
read-only, no model loaded -- from `full_fidelity/front_full_fidelity.csv`. The result is
**byte-identical to the committed version**: `git diff` on that path comes back empty. So
the rebuild changed nothing, which is itself the proof that the original was fine.

Present at `results/search/plan_a_bird/pareto_front_plot_data.csv`: 10 rows, columns identical to
the other eight class-agnostic runs, front member #0 correctly carrying all three roles
(selected `C*`, best selectivity, strongest forgetting).

The committed bird figure was **not** touched. That script writes a PNG as well as the
CSV, so the PNG was directed to a scratch path outside the repository and discarded.
`results/search/plan_a_bird/pareto_front_plan_a_bird.png` is unchanged.

## Frog: a different filename, not a missing file

Frog has no `pareto_front_plot_data.csv` and does not need one. It predates the
class-agnostic plotter and carries the equivalent artefact as
`results/search/plan_a_frog/pareto_front_plot_table.csv` -- ten rows, one per front
member, with its own `role` column marking best selectivity and strongest forgetting.

It was written by `plot_pareto_front.py`, the frog-specific plotter that is deliberately
left frozen so the committed frog figure stays reproducible from the script that made
it. Regenerating frog's data under the newer name would produce a second file with
different columns describing the same front, so it was not done. **All ten classes have
their plotted values on disk.**

## Generated: `class_structure_analysis.png`, and it returned a null result

Built from `results/analysis/class_structure/summary.json` and
`channel_contrast_all_classes.csv` -- both committed. No new experiment; the only
computation is arithmetic over those files. It also writes
`class_structure_similarity.csv`, the 10x10 matrix behind panel C.

**Panel A -- structure exists, in every class.** Between 84.1% (horse) and 91.2% (ship)
of channels stand above the noise floor, against **0.55%** for the predecessor project's
instance-level forget set, where the null control gives 1.00% by construction. Three
orders of magnitude. This is the project's founding result and it holds for all ten.

**Panel B -- structure does NOT predict difficulty, and this is the important finding.**
Median SNR against pure `ACC_f` is a null scatter: **Pearson r = -0.04 over ten points**.
Truck is sixth of ten on median SNR and fifth on channels above the floor, yet its
`ACC_f` of 42.10 is more than double the next worst. Automobile has the **least**
structure of any class and forgets 3.3x better than truck.

This matters for the write-up. `limitations_future_work_notes.md` proposed regressing
per-class `ACC_f` on the activation-contrast statistic and called it "the strongest
contribution available". **That regression has now been run and it is null.** The
proposal should be corrected rather than left standing as future work, and the null
reported: a reader will otherwise assume the obvious explanation, which the data does
not support.

**Panel C -- the similarity matrix, which does support the truck reading.** Cosine
similarity between the per-class channel-contrast vectors recovers the semantic
grouping without being told it: vehicles with vehicles (airplane-ship 0.41,
truck-automobile 0.32, ship-truck 0.29), animals with animals (cat-dog 0.30,
deer-frog 0.30, bird-deer 0.29). That it reproduces a structure nobody encoded is the
evidence the measurement means something.

**Truck's nearest neighbour is automobile (0.32), and the relation is mutual** --
automobile's nearest is truck. That matches `truck_failure_analysis.png` exactly, where
a model that never saw a truck sends 68.4% of them to automobile. Two independent
measurements, one on activations of `W_0` and one on predictions of four unlearned
models, point at the same pair.

### What this figure must NOT be captioned as showing

That similarity predicts difficulty. **It does not**: Pearson r = -0.08 between each
class's maximum similarity to any other class and its `ACC_f`. Airplane is the
counterexample and it is decisive -- it has the highest similarity to another class of
any of the ten (0.41, with ship) and still reaches `ACC_f` 0.00.

The defensible claim is narrower and still worth making: truck's difficulty **coincides
with** its structure being most shared with a class that stays, and two independent
measurements agree on which class that is. Whether that sharing is the *cause* is not
established by these artefacts.

## Not generated, and why

**Seed-variance, ablation, baseline and runtime figures.** These cannot be built from
existing artefacts, because the underlying experiments have not been run. Building them
would require exactly the searches, ablations and baseline implementations that the
rules for this task excluded. They are listed in `limitations_future_work_notes.md` as
outstanding work, not as missing plots.

## Rules observed

No search was run. Nothing was trained. No refinement was re-run. No committed result
changed. The only computation anywhere in this task was the forward pass described
above, which the ask explicitly permitted for the truck figure. No `.pt` or `.zip` is
committed, and the reconstructed `C*` was held in memory and never written to disk.
