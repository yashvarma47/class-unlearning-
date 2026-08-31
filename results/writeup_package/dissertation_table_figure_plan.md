# Table and figure plan — chapter by chapter

Every asset below **already exists** in the repository. Nothing here requires a new
experiment. Paths are relative to the repository root.

Marking criteria referenced: **[P]** precise problem statement · **[C]** critical
synthesis · **[M]** rigorous methodology with alternatives · **[T]** technical
implementation · **[E]** systematic evaluation · **[R]** deep reflection on what worked
and failed · **[V]** clear visual communication · **[I]** independent decision-making.

---

## Chapter 1 — Introduction and Problem Statement

| asset | type | path |
|---|---|---|
| **F1.1** Instance-level vs class-level structure | figure — reuse **panel A only** of `class_structure_analysis.png`, or cite it forward | `results/writeup_package/figures/class_structure_analysis.png` |
| **T1.1** The predecessor's negative result | table — hand-built from `README.md` | 10,534 strategies · 3 operator families · 5 selectors · 4 objective formulations · best `S` = 1.158 · retraining `S` ≈ 932 |

> **Caption F1.1.** *Forget-specific structure exists at class level and not at instance
> level. Per-channel activation contrast against a null control built from two disjoint
> halves of `D_r`. An instance-level forget set puts 0.55% of channels above the noise
> floor, where chance alone gives 1.00%; every CIFAR-10 class puts 84.1–91.2% above it.
> This three-order-of-magnitude gap is the premise of the dissertation.*

> **Caption T1.1.** *The predecessor project's exhaustive instance-level search. No
> configuration of operator, selector, objective or algorithm achieved selectivity above
> 1.158, against approximately 932 for retraining from scratch — damage to the forget and
> retain sets was empirically indistinguishable.*

**Why these help.** [P] The problem statement stops being generic and becomes *this*
problem: a measured failure, a mechanism, and a prediction. [I] Showing the negative
result that motivated a change of direction is the clearest available evidence of
independent judgement. **Do not open with a method diagram** — open with the reason the
method exists.

**Optional, if you have time to draw it:** a one-page schematic of the unlearning
setting (`W_0`, `D_f`, `D_r`, `W_ref`, the edited model, and the gap-to-`W_ref` target).
No such figure exists; it would be hand-drawn. Worth it — it fixes notation visually and
carries §2.1.4's framing.

---

## Chapter 2 — Background and Foundations

| asset | type | path / source |
|---|---|---|
| **T2.1** Layer groups and parameter distribution | table — build from the context pack §4 | 6 groups; `layer4` = 8,393,728 params = 75.12% of 11,173,962 |
| **T2.2** Method-family taxonomy | table | `claudedocs/research_anchor_paper_20260827.md` comparison table |
| **F2.1** Pareto dominance schematic | figure — **would need drawing** | — |

> **Caption T2.1.** *Layer groups of the CIFAR ResNet-18, and the share of the 11,173,962
> parameters each holds. The concentration in `layer4` is the reason the edit-cost
> objective is normalised: an absolute parameter-change norm would price a large edit to
> `fc` as nearly free and a small edit to `layer4` as expensive, which is a statement
> about the architecture rather than about the strategy.*

**Why these help.** [T] T2.1 does double duty — it defines the chromosome length and
pre-justifies `f3` two chapters early. [C] T2.2 gives the reader the map before Chapter 3
starts criticising it.

**Honest note.** F2.1 does not exist and a generic Pareto-front schematic adds little
that a real front does not. Consider skipping it and forward-referencing a real
per-class front instead.

---

## Chapter 3 — Literature Review and Critical Analysis

| asset | type | path / source |
|---|---|---|
| **T3.1** Cross-method comparison | table | `claudedocs/research_anchor_paper_20260827.md` — dataset, architecture, forget setting, baselines, metrics, gradient-free?, deployable without retraining?, closeness to our setup |
| **T3.2** The anchor's Table 1 | table | `results/writeup_package/benchmark_comparison_table.csv`, **literature rows only** |
| **T3.3** Protocol fragmentation | table — build from T3.1 | Which classes each paper actually reports; which MIA definition each uses |

> **Caption T3.1.** *Seven candidate anchor studies compared along the axes that decide
> whether a comparison with this work is meaningful. Kodge et al. is the only study whose
> primary setting is identical to ours — CIFAR-10, ResNet-18, single-class forgetting,
> gradient-free, deployable without retraining.*

> **Caption T3.3.** *Protocol fragmentation across the field. No two studies report the
> same forget classes, and the membership-inference attack differs in feature, fitting
> set and scoring set between them. This is why the present work reimplements the anchor's
> measurement protocol rather than comparing headline numbers across papers.*

**Why these help.** [C] T3.3 is the table that converts a summary into a critique — it
makes a structural argument about the field rather than describing papers one at a time.
[I] It also pre-justifies the decision to adopt one anchor protocol, which is a
methodological choice you made and should own.

**Deliberately withheld until Chapter 5:** the row containing MED-US's own numbers. Do
not put your result in a literature-review table; T3.2 shows the field, T5.4 shows you
in it.

---

## Chapter 4 — MED-US

| asset | type | path / source |
|---|---|---|
| **F4.1** Chromosome and decoding schematic | figure — **would need drawing** | — |
| **T4.1** Chromosome gene table | table | context pack §4 / `src/medus_class/search/genome.py` |
| **T4.2** The operator library | table | `configs/operators/lookup.yaml` — 8 operators, 2 channels, ladders |
| **T4.3** Selection rules | table | `src/medus_class/operators/selection.py` — `class_contrast` / `magnitude` / `random` |
| **T4.4** NSGA-II configuration | table | population 10, generations 50, uniform crossover 0.9, random-reset mutation 0.10, binary tournament, elitist (μ+λ), seed 42, 510 evaluations |
| **T4.5** Objective definitions | table | `f1`, `f2`, `f3` with formula, direction, bound, what it measures |
| **T4.6** Refinement acceptance checks | table | `results/writeup_package/refinement_acceptance_table.md` — the six checks and shared hyperparameters |

> **Caption T4.2.** *The eight gradient-free operators, in two chromosome channels, with
> their calibrated intensity ladders. Only levels 1 and 2 were reachable in the
> experiments (`max_level = 2`). `REINIT` and `SIGN_FLIP` are absent from the library
> rather than disabled by configuration, so that no configuration change can reintroduce
> them; `PRUNE` and `RANDOM_PRUNE` pin themselves to data-free selection rules and serve
> as controls on the forget-informed selector.*

> **Caption T4.5.** *The three objectives. All are minimised, so domination is "no worse
> in every objective and strictly better in at least one". No privacy term appears:
> membership inference is computed as a diagnostic and as an evaluation metric, and is
> never read back into the search.*

> **Caption T4.6.** *The six checks every refinement had to pass before its checkpoint
> was retained, with the hyperparameters shared by all nine attempts. Check 6 — BatchNorm
> buffers unchanged — was added after an earlier attempt passed every weight-based guard
> while `D_r` batches silently re-estimated the running statistics and undid the operator
> edit.*

**Why these help.** [M] T4.2's exclusion column and T4.3's control column are the
clearest evidence of alternatives considered and rejected. [T] T4.1 and T4.4 make the
implementation checkable rather than asserted. [I] T4.6 documents a bug you found in your
own method and the guard you added — that is worth more than a clean table.

**F4.1 is the one figure genuinely worth drawing by hand.** A single diagram showing a
30-integer genome, the six layer groups it addresses, and one worked decoding — for
instance truck's `C*`, which decodes to `CLIP|MASK|QUANTIZE` — would carry §4.3 better
than any amount of prose. [V] Use truck's real chromosome from the context pack, not an
invented one.

---

## Chapter 5 — Experimental Study

The chapter with the most existing assets. Every figure below is already rendered.

### §5.2 — Reference and baseline models

| asset | path |
|---|---|
| **T5.1** Reference model validation | `results/writeup_package/reference_model_validation_table.md` |

> **Caption T5.1.** *The ten retain-only reference models. Each was trained from scratch
> on `D_r_train` only and never saw a single image of its forget class; `D_f_test`
> accuracy is 0.0000 for all ten, which is the correctness condition rather than a
> result. Checkpoint selection used `D_r_test` accuracy with `D_r_test` loss as
> tie-breaker; `D_f_test` was logged every epoch but never influenced selection.*

**Why.** [E] Establishes the gold standard before any result depends on it. The sha256
and source-bundle columns make the distributed training auditable. [I]

### §5.5 — Class structure

| asset | path |
|---|---|
| **F5.1** Class-structure analysis, three panels | `results/writeup_package/figures/class_structure_analysis.png` |
| **T5.2** Inter-class similarity matrix | `results/writeup_package/class_structure_similarity.csv` |

> **Caption F5.1.** *Class structure explains the regime, not the ranking. **(A)** Every
> CIFAR-10 class places 84.1–91.2% of channels above the noise floor, against 0.55% for
> an instance-level forget set where chance alone gives 1.00%. **(B)** That structure does
> not predict how hard a class is to forget: median SNR against pure `ACC_f` gives Pearson
> r = −0.04 over the ten classes. Truck is sixth of ten on structure and worst by far on
> forgetting; automobile has the least structure of any class and forgets 3.3× better.
> **(C)** Cosine similarity between per-class channel-contrast vectors recovers the
> semantic grouping without being told it — vehicles with vehicles, animals with animals —
> which is the evidence the measurement is meaningful. Truck's nearest neighbour is
> automobile, mutually.*

**Why.** [E] Answers RQ1 and reports the RQ3 null in one image. [R] Panel B is a negative
result presented as a finding, which is exactly the reflection the excellent band asks
for. [V] Three panels that make one argument.

### §5.6 — Pure MED-US

| asset | path |
|---|---|
| **T5.3** Pure MED-US, all ten classes | `results/writeup_package/pure_medus_10_class_table.md` |
| **F5.2–F5.4** Three Pareto fronts | `results/search/plan_a_{dog,truck,airplane}/pareto_front_plan_a_*.png` |

> **Caption T5.3.** *Pure MED-US across all ten CIFAR-10 classes. Every row is
> gradient-free weight surgery; no gradient step was applied to any of these models. `C*`
> was selected by one rule applied uniformly — the front member maximising the anchor
> composite `ACC_r × (1 − ACC_f)` — which is the anchor paper's own metric function rather
> than one devised here. Retention varies by 3.1 points across the ten classes; forgetting
> varies by 42.1.*

> **Caption F5.2–F5.4.** *Pareto fronts for the strongest, weakest and most complete
> classes: dog (`S` = 4427.91), truck (`S` = 93.99) and airplane (the only class reaching
> `ACC_f` 0.00). `C*` is starred on each. Three fronts side by side make the class-wise
> spread visible in a way the aggregate table cannot.*

**Why.** [E] T5.3 is the central table of the dissertation. [V] Three fronts rather than
ten — ten in sequence makes a reader skip the section; the remaining seven go to
Appendix E.

### §5.7 — Benchmark comparison

| asset | path |
|---|---|
| **T5.4** Benchmark comparison | `results/writeup_package/benchmark_comparison_table.md` |
| **F5.5** Benchmark comparison, four metrics | `results/writeup_package/figures/benchmark_comparison.png` |

> **Caption F5.5.** *Retention is competitive; forgetting is not. Each metric is a
> separate panel on its own scale — the panels are not comparable to one another. The
> Kodge et al. values are as reported in that paper's Table 1 and were not re-measured
> here; the two MED-US rows are measured in this harness. Dots rather than bars because
> three of the four panels are zoomed, and a bar drawn from a non-zero baseline misstates
> the ratio between its neighbours. The dashed line is retraining from scratch.*

**Why.** [E] Places the work in the field. [I] The reported-versus-measured distinction is
carried *in the figure itself*, not only in the text — that is the honest presentation of
an imperfect comparison, and markers notice it. [V] Small multiples rather than one
grouped chart avoids the dual-axis error.

### §5.8 — The hybrid variant

| asset | path |
|---|---|
| **T5.5** Hybrid, all ten classes | `results/writeup_package/hybrid_medus_10_class_table.md` |
| **T5.6** Pure vs hybrid summary + per-class deltas | `results/writeup_package/pure_vs_hybrid_summary_table.md` |
| **T5.7** Refinement acceptance record | `results/writeup_package/refinement_acceptance_table.md` |
| **F5.6** Per-class `ACC_f`, pure vs hybrid | `figures/pure_vs_hybrid_acc_f_by_class.png` |
| **F5.7** Per-class `ACC_r`, pure vs hybrid | `figures/pure_vs_hybrid_acc_r_by_class.png` |
| **F5.8** Per-class composite, pure vs hybrid | `figures/pure_vs_hybrid_composite_by_class.png` |

> **Caption F5.6.** *Forget-class accuracy falls for every refined class. Classes are
> ordered by pure `ACC_f`, worst first; the same order is used in all three pure-versus-
> hybrid figures so they can be read across. Airplane is a deliberate no-op — its pure
> `ACC_f` is already 0.00, so no refinement was attempted and its two markers coincide.
> **The hybrid is a different method: it applies two gradient steps outside the search and
> is not gradient-free.***

> **Caption F5.7.** *Retention is essentially unchanged by refinement. Note the axis: the
> entire ten-class range spans about three points and no single class moves by more than
> 0.25. Markers rather than bars are used precisely so that a non-zero axis is legitimate.*

> **Caption T5.7.** *Nine attempts, nine accepted, zero rejected. BatchNorm buffer
> movement is exactly 0.000000 on every accepted refinement, with zero counter changes,
> and parameter movement stays between 0.000303 and 0.000420 against a budget of 0.0400.*

**Why.** [E] Three figures in one order let the reader see the trade rather than read it.
[R] The buffer-movement column is the evidence that a subtle failure was caught, which is
reflection backed by a number. [I] Repeating the "not gradient-free" clause in the caption
means a reader who only skims figures still cannot mistake the hybrid for the main method.

### §5.9 — Operator analysis

| asset | path |
|---|---|
| **F5.9** Operator frequency across the ten `C*` | `figures/operator_frequency_selected_cstar.png` |

> **Caption F5.9.** *`MASK` appears in the selected candidate for all ten classes; no
> other operator appears in more than four. Three classes select `MASK` alone. This is a
> search outcome rather than a design choice — all eight operators were available at equal
> cost in all ten independent runs — but with no operator ablation it remains convergent
> evidence rather than a demonstrated property.*

**Why.** [E] A result that emerged rather than being designed in. [R] The caption states
its own limit, which is more persuasive than overclaiming.

### §5.10 — The truck case study

| asset | path |
|---|---|
| **F5.10** Truck failure analysis | `figures/truck_failure_analysis.png` |
| **T5.8** Truck prediction distribution | `results/writeup_package/truck_prediction_distribution.csv` |

> **Caption F5.10.** *Unlearning truck moves it toward automobile, and stalls. Predicted
> class of every held-out truck image under each of four models; inference only, no model
> was trained or re-refined. A model that never saw a truck sends 68.4% of them to
> automobile. Pure MED-US sends only 16.7% there and leaves 42.1% still classified truck;
> the refinement moves a further 11.2 points out, of which 20.4% reach automobile. The
> failure is a partial move along the truck–automobile axis, not a random scattering — the
> reference's destination is the same destination, reached less far.*

**Why.** [R] This is the figure that converts the project's weakest result into its most
mechanistically interesting one. [E] It is corroborated independently by panel C of F5.1,
and the rebuilt `C*` reproduces the published `ACC_f` exactly, which is stated in the
text as the check that the figure describes the published models. [I] Devoting a
subsection to your worst class is a deliberate choice; say so.

### §5.11 — Cost

| asset | source |
|---|---|
| **T5.9** Search cost per class | `results/search/plan_a_*/summary.json` — elapsed, evaluated, cache hits, front size |

> **Caption T5.9.** *Search cost per class. Ten runs, zero failures, fronts of ten members
> each. Cache hit rates of 37–58% reflect the latent genes in the encoding, which let
> distinct genomes decode to the same strategy.*

**Why.** [T] Runtime is a claimed advantage of a gradient-free method, so it is measured
rather than asserted — while being explicit that the implementation is unoptimised and a
wall-clock race against published figures would not be like-for-like.

---

## Chapter 6 — Conclusion

No new figures. **One optional summary table** consolidating RQ1–RQ4 against their
evidence, built from the key claims register in the context pack §7.

> **Caption T6.1.** *The four research questions, the evidence bearing on each, and the
> verdict — including the two questions answered in the negative.*

**Why.** [R] A conclusion that tabulates its own verdicts, including the nulls, reads as
confident rather than defensive.

---

## Assets deliberately not planned for

| asset | why not |
|---|---|
| Seed-variance plots | The experiment was not run (one seed only). |
| Ablation plots | No ablation was run — not random search at equal budget, not operator families, not the selector. |
| Baseline-comparison plots from this harness | No baseline was re-implemented. |
| Runtime comparison charts | Implementation is unoptimised; the comparison would mislead. |
| A pure-vs-hybrid *merged* table | Forbidden by the separation rules. |

Naming these explicitly in the text — rather than leaving their absence to be noticed —
is itself evidence of [R].

---

## Summary counts

| chapter | tables | figures | of which already rendered |
|---|---:|---:|---:|
| 1 | 1 | 1 | 1 (reuse) |
| 2 | 2 | 0–1 | 0 |
| 3 | 3 | 0 | 0 |
| 4 | 6 | 0–1 | 0 |
| 5 | 9 | 10 | **10** |
| 6 | 1 | 0 | 0 |
| **Total** | **22** | **11–13** | **11** |

Only **two** figures would need to be drawn by hand — the chromosome/decoding schematic
(F4.1, genuinely worth it) and optionally an unlearning-setting schematic for Chapter 1.
Everything else exists.
