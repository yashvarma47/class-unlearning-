# Dissertation writing context pack — MED-US

**Title:** Multi-Objective Evolutionary Design of Unlearning Strategies (MED-US)
**Setting:** CIFAR-10 · ResNet-18 (CIFAR variant) · single-class forgetting
**Written:** 2026-08-31. Every number below was re-read from the committed CSVs on that
date and matches. No experiment was run to produce this file.

---

## 1. Final project summary — 13 bullets

1. **The task is class-level unlearning, not instance-level.** `D_f` is one whole
   CIFAR-10 class (5,000 train / 1,000 test); `D_r` is the other nine (45,000 / 9,000).
   All ten classes were run in turn as ten independent experiments.

2. **The project exists because instance-level unlearning failed, and the failure was
   explained.** The predecessor searched 10,534 instance-level strategies across 3
   operator families, 5 selectors and 4 objective formulations, and the best selectivity
   `S` ever measured was **1.158** against ~932 for retraining — damage to `D_f` and
   `D_r` was empirically identical.

3. **A direct measurement said why.** Per-channel activation contrast against a null
   control put **0.55%** of channels above the noise floor for an instance-level `D_f`,
   where chance alone gives **1.00%**. Fewer stood out than chance produces: there was no
   forget-specific structure for any weight edit to remove.

4. **That explanation made a falsifiable prediction, and it held.** For class-level
   forget sets, **84.1%–91.2%** of channels stand above the noise floor. Three orders of
   magnitude. This is the founding result and it belongs early in the dissertation.

5. **Ten retain-only reference models `W_ref` exist and are validated.** One per class,
   trained from scratch on `D_r_train` only, 200 epochs, seed 42, selected on `D_r_test`
   accuracy. All ten reach `D_f_test` accuracy **0.0000**; `D_r_test` accuracy is
   **0.9506 ± 0.0057**. All ten verdicts `PASS`, all sha256-recorded.

6. **Pure MED-US is the main method and is entirely gradient-free.** NSGA-II over an
   integer-coded, layer-wise genome; a chromosome selects an operator and an intensity
   for each of six layer groups. No gradient is computed at any point — selection uses
   forward passes only.

7. **Pure MED-US result, ten-class mean:** `ACC_r` **93.84 ± 1.15**, `ACC_f`
   **12.55 ± 11.57**, composite **82.09 ± 11.00**, anchor MIA **92.54 ± 8.62**.

8. **Hybrid MED-US is a separate method**, not a variant to be merged in. It is the pure
   `C*` plus one clipped gradient-ascent step on `D_f` and one repair step on `D_r`,
   BatchNorm frozen, applied outside the search. Nine classes eligible, **nine attempted,
   nine accepted, zero rejected**. Airplane is a deliberate no-op.

9. **Hybrid result, ten-class mean:** `ACC_r` **93.72 ± 1.12**, `ACC_f` **7.55 ± 8.87**,
   composite **86.66 ± 8.52**, anchor MIA **95.05 ± 6.08**. Five points of `ACC_f` bought
   for 0.12 of retain accuracy.

10. **Against the anchor, retention is competitive and forgetting is not.** Kodge et al.
    report `ACC_r` 94.19 ± 0.50, `ACC_f` 0.03 ± 0.09, MIA 95.50 ± 14.23. Pure MED-US is
    0.35 behind on retention and **12.52 points worse on forgetting**. This is the honest
    headline and must be stated plainly, not buried.

11. **`MASK` appears in the selected `C*` of all ten classes** — the only operator that
    does. Frequency across the ten: MASK 10, CLIP 4, DAMP 3, PRUNE 2, RANDOM_PRUNE 2,
    RESET 2, QUANTIZE 1, NOISE 0. This is a search outcome, not a design choice.

12. **Per-class variation is large and stable.** Airplane reaches `ACC_f` **0.00**;
    truck sits at **42.10** pure and **30.90** hybrid. The same classes stay weak after
    refinement.

13. **Truck's failure has a mechanism, and it is not "less structure".** A model that
    never saw a truck sends **68.40%** of truck images to automobile; pure `C*` leaves
    **42.10%** still called truck and sends only 16.70% to automobile; hybrid leaves
    **30.90%** and sends 20.40%. The failure is a *partial move along the
    truck–automobile axis*, not a random scattering.

14. **The obvious explanation for per-class difficulty was tested and is null.**
    Correlation between structure magnitude and `ACC_f` across the ten classes is
    Pearson **−0.04**; between maximum inter-class similarity and `ACC_f`, **−0.08**.
    Truck is sixth of ten on median SNR; automobile has the *least* structure of any
    class and forgets 3.3× better. **Structure explains the regime, not the ranking.**

15. **Everything is one seed (42) and no baseline was re-implemented.** Every `±` is
    class-to-class spread, not run-to-run variance. These two facts must appear in the
    limitations and in the threats-to-validity section.

---

## 2. Updated dissertation outline

Chapter titles are fixed as required; subsections are updated to the final project.
Full version with per-section notes: `revised_dissertation_outline.md`.

```
1. Introduction and Problem Statement
   1.1 The right to erasure and the cost of exact retraining
   1.2 Approximate unlearning by post-hoc weight editing
   1.3 From an instance-level failure to a class-level hypothesis
   1.4 Problem statement: gradient-free single-class unlearning as multi-objective search
   1.5 Research questions (RQ1–RQ4)
   1.6 Contributions
   1.7 Structure of this dissertation

2. Background and Foundations (General SOTA)
   2.1 Machine unlearning: definitions and setting
   2.2 Deep networks and where class information lives
   2.3 Gradient-free model-editing primitives
   2.4 Multi-objective optimisation foundations
   2.5 Evaluating unlearning

3. Literature Review and Critical Analysis
   3.1 Families of unlearning methods, critically compared
   3.2 Cross-cutting critique I: protocol fragmentation
   3.3 Cross-cutting critique II: what a membership-inference number certifies
   3.4 Search-based and evolutionary model editing
   3.5 The anchor study, and what adopting it costs
   3.6 Gap analysis and positioning of MED-US

4. Multi-Objective Evolutionary Design of Unlearning Strategies (MED-US)
   4.1 Design rationale and the alternatives rejected
   4.2 The precondition: forget-specific structure
   4.3 The search space
   4.4 The operator library
   4.5 The three objectives
   4.6 NSGA-II for an integer-coded genome
   4.7 Two-tier fidelity and the selection of C*
   4.8 The hybrid variant — a separate method
   4.9 Implementation and reproducibility

5. Experimental Study
   5.1 Experimental setup
   5.2 Reference and baseline models
   5.3 Metrics as implemented
   5.4 Protocol and controls
   5.5 Result 1: class structure exists
   5.6 Result 2: pure MED-US across ten classes
   5.7 Result 3: comparison against the anchor benchmark
   5.8 Result 4: the hybrid variant
   5.9 Result 5: operator analysis
   5.10 Result 6: the truck case study
   5.11 Computational cost
   5.12 Threats to validity
   5.13 Critical discussion

6. Conclusion
   6.1 Answers to the research questions
   6.2 Contributions restated against the evidence
   6.3 Limitations
   6.4 Future work
   6.5 Closing reflection

7. Declaration on the Use of AI
8. References
```

---

## 3. Outdated subsections and their replacements

| # | Outdated content | Why it is wrong | Replace with |
|---|---|---|---|
| 1 | **"f1 maximise unlearning, f2 maximise utility, f3 minimise privacy leakage"** | Wrong on all three counts. There is no maximisation anywhere — **all three objectives are minimised** — and privacy leakage is not an objective at all. | `f1 = JS(P_ref(D_f) ‖ P_cand(D_f))`, minimised, bounded by ln 2 · `f2 = L_r`, retain cross-entropy, minimised · `f3 = ‖θ − θ₀‖₂ / ‖θ₀‖₂`, relative edit cost, minimised. Domination is "no worse in every objective, strictly better in at least one". |
| 2 | **MIA described as a search objective** | MIA is never read back into an objective. It is computed as a *diagnostic* during search and as an *anchor evaluation metric* afterwards. | Move MIA to §2.5 (background) and §5.3 (metrics as implemented). Add §3.3, a critique of what an MIA number certifies. |
| 3 | **Instance-level / random-subset forgetting as the experimental setting** | The final experiment is class-level only. The codebase supports no other split mode, deliberately. | Class-level: `D_f` = one class, `D_r` = nine. Keep the instance-level work as the *motivating failure* in §1.3, not as the setting. |
| 4 | **A single undifferentiated "MED-US" method** | There are two methods with different assumptions. | Pure MED-US (gradient-free, the main contribution) in §4.1–4.7; hybrid MED-US as a clearly separated §4.8, reported separately in §5.8. |
| 5 | **Any wording implying the hybrid is gradient-free** | It applies two gradient steps. | "Pure MED-US is gradient-free throughout. The hybrid applies two gradient steps outside the search and is therefore **not** gradient-free; it is not a like-for-like comparator for the anchor's Table 1." |
| 6 | **Composite defined as `ACC_r × (100 − ACC_f) × MIA`** | An early literature-review draft recorded this; it was read wrong. The anchor's own source computes `metric_function(x, y) = x × (1 − y)` on fractions. | `composite = ACC_r × (1 − ACC_f)`. **MIA is not a term in it.** The correction is recorded in `src/medus_class/evaluation/anchor.py`. |
| 7 | **Any claim of a second architecture or dataset** | Only ResNet-18 / CIFAR-10 exists. Docstrings mentioning VGG19 are inherited aspiration, and no such file exists. | State the single-setting scope explicitly in §5.1 and §6.3. |
| 8 | **`±` presented as an uncertainty estimate** | It is class-to-class spread over ten classes at one seed. | Say so at first use in §5.6 and again in §5.12. |

---

## 4. Exact final numbers and their provenance

Every value below was re-read on 2026-08-31 from the file named. **Do not retype these
from memory — copy them from the source file.**

### 4.1 Headline aggregates

| quantity | value | source file |
|---|---|---|
| Pure `ACC_r` | **93.84 ± 1.15** | `results/literature_alignment/ten_class_pure_mean_std.csv` |
| Pure `ACC_f` | **12.55 ± 11.57** | same |
| Pure composite | **82.09 ± 11.00** | same |
| Pure anchor MIA | **92.54 ± 8.62** | same |
| Pure selectivity `S` | 1053.88 ± 1413.47 | same |
| Hybrid `ACC_r` | **93.72 ± 1.12** | `results/literature_alignment/ten_class_hybrid_mean_std.csv` |
| Hybrid `ACC_f` | **7.55 ± 8.87** | same |
| Hybrid composite | **86.66 ± 8.52** | same |
| Hybrid anchor MIA | **95.05 ± 6.08** | same |
| Hybrid selectivity `S` | 1151.44 ± 1592.37 | same |
| Δ`ACC_r` pure→hybrid | −0.12 | `results/writeup_package/pure_vs_hybrid_summary_table.csv` |
| Δ`ACC_f` | −5.00 | same |
| Δcomposite | +4.58 | same |
| ΔMIA | +2.51 | same |

### 4.2 Benchmark row (reported by the anchor, not measured here)

| method | ACC_r | ACC_f | MIA | source |
|---|---|---|---|---|
| Kodge et al. 2024 | 94.19 ± 0.50 | 0.03 ± 0.09 | 95.50 ± 14.23 | `benchmark_comparison_table.csv`, row labelled `measured_in_this_harness = no` |
| Retraining (reported) | 94.81 ± 0.52 | 0.00 | 100.00 ± 0.00 | same |
| Original (reported) | 94.89 ± 0.31 | 94.89 ± 2.75 | 0.03 ± 0.03 | same |
| **`W_0` (measured here)** | 94.79 ± 0.29 | 94.79 ± 2.59 | 0.00 ± 0.00 | same, `= yes` |
| **`W_ref` (measured here)** | 95.06 ± 0.57 | 0.00 ± 0.00 | 100.00 ± 0.00 | same, `= yes` |

Harness-agreement evidence: our `W_0` and `W_ref` land within **0.10** and **0.25**
points of `ACC_r` of the paper's Original and Retraining rows.

### 4.3 Per-class extremes

| quantity | value | source |
|---|---|---|
| Best class: airplane, pure `ACC_f` | **0.00** | `pure_medus_10_class_table.csv` |
| airplane pure composite / MIA | 92.86 / 100.00 | same |
| Worst class: truck, pure `ACC_f` | **42.10** | same |
| truck pure composite / MIA / `S` | 53.80 / 69.60 / 93.99 | same |
| truck hybrid `ACC_f` | **30.90** | `hybrid_medus_10_class_table.csv` |
| Largest refinement gain | truck, Δ`ACC_f` −11.20 | `pure_vs_hybrid_summary_table.md` |
| Runner-up gain | ship, −9.30 (14.00 → 4.70) | same |
| Worst single-class retain cost | cat, −0.2444 | same |

### 4.4 Truck prediction distribution

Source: `results/writeup_package/truck_prediction_distribution.csv` (inference only).

| model | still truck | → automobile |
|---|---:|---:|
| `W_0` | 95.40% | 2.80% |
| `W_ref` | 0.00% | **68.40%** |
| `C*` pure | **42.10%** | 16.70% |
| `C*` hybrid | **30.90%** | 20.40% |

The rebuilt `C*` reproduces the published `ACC_f` exactly (42.10 / 30.90) — that
agreement is the check that the figure describes the published models.

### 4.5 Operator frequency

Source: recomputed from `pure_medus_10_class_table.csv` at build time.

`MASK 10 · CLIP 4 · DAMP 3 · PRUNE 2 · RANDOM_PRUNE 2 · RESET 2 · QUANTIZE 1 · NOISE 0`

Three classes select `MASK` alone (deer, dog, horse); five select it alone or with a
single partner.

### 4.6 Class structure

| quantity | value | source |
|---|---|---|
| Class-level channels above noise floor | 84.1% (horse) – 91.2% (ship) | `results/analysis/class_structure/summary.json` |
| Instance-level (predecessor) | 0.55% | `README.md` |
| Null control | 1.00% | `README.md` |
| r(median SNR, `ACC_f`) | −0.04 | computed in `experiments/build_class_structure_figure.py` |
| r(max similarity, `ACC_f`) | −0.08 | same |
| truck ↔ automobile similarity | 0.32, mutual | `class_structure_similarity.csv` |
| airplane ↔ ship similarity | 0.41, highest of any pair | same |

### 4.7 Refinement acceptance

| quantity | value | source |
|---|---|---|
| Attempts / accepted / rejected | 9 / 9 / 0 | `refinement_acceptance_table.csv` |
| BatchNorm buffer movement | **0.000000 on all nine**, 0 counter changes | same |
| Parameter movement range | 0.000303 (cat) – 0.000420 (frog) | same |
| Movement budget | 0.0400 | same |
| Acceptance checks | 6, all passed by all nine | `refinement_acceptance_table.md` |

### 4.8 Cost

| quantity | value | source |
|---|---|---|
| Search per class | 6.4 – 15.5 min (383.6 – 931.8 s) | `results/search/plan_a_*/summary.json` |
| Evaluations per class | 212 – 319 real, 191 – 298 cache hits | same |
| Evaluation budget | 510 per class (10 + 50 × 10) | config |
| Full-fidelity re-measurement | 9.3 – 10.9 min per class | run logs |
| End-to-end per class | ~30 – 35 min | sweep-driver log timestamps |
| Refinement | ~4 – 5 min per class | run logs |
| `W_ref` training | ~2.32 h per class, Tesla T4 | `class*_kaggle_manifest.json` |
| Search failures | **0** across all ten runs | `summary.json` |

---

## 5. Thesis-ready tables

All under `results/writeup_package/`. Each exists as both `.csv` (numeric, for import)
and `.md` (rendered, with the surrounding commentary already written).

| # | file | content | suggested placement |
|---|---|---|---|
| T1 | `reference_model_validation_table` | Ten `W_ref`: verdict, `D_f_test`, `D_r_test`, epoch, sha256, source bundle | §5.2 |
| T2 | `pure_medus_10_class_table` | Per-class `C*`: operators, ACC_r/ACC_f/composite/MIA, `S`, f1/f2/f3, search minutes | §5.6 — **the central table** |
| T3 | `hybrid_medus_10_class_table` | Nine refined rows + airplane no-op, with parameter and BN movement | §5.8 |
| T4 | `pure_vs_hybrid_summary_table` | Aggregate five-metric comparison; the `.md` also carries per-class deltas | §5.8 |
| T5 | `benchmark_comparison_table` | Anchor's eight rows + our four | §5.7 |
| T6 | `refinement_acceptance_table` | Six acceptance checks per class, movement vs budget, shared hyperparameters | §5.8 or Appendix |
| T7 | `truck_prediction_distribution.csv` | Full 10-class predicted distribution under four models | §5.10 |
| T8 | `class_structure_similarity.csv` | 10 × 10 cosine similarity matrix | §5.5 / §5.10 |

**Supporting prose already written** (mine these, do not rewrite):
`results_chapter_notes.md` (bullets for every results section, all numbers traced),
`key_numbers_summary.md` (single reference sheet),
`limitations_future_work_notes.md` (seven limitations with what each does not license),
`figure_inventory.md`, `missing_figures_status.md`.

---

## 6. Thesis-ready figures

All under `results/writeup_package/figures/`, 300 dpi PNG, light-surface, colour-blind
validated, every mark directly labelled.

| # | file | shows | placement |
|---|---|---|---|
| F1 | `class_structure_analysis.png` | 3 panels: structure per class · structure vs difficulty (null) · inter-class similarity matrix | §5.5 — **first results figure** |
| F2 | `pure_vs_hybrid_acc_f_by_class.png` | Per-class `ACC_f`, pure vs hybrid (dumbbell) | §5.8 |
| F3 | `pure_vs_hybrid_acc_r_by_class.png` | Per-class `ACC_r`, zoomed axis | §5.8 |
| F4 | `pure_vs_hybrid_composite_by_class.png` | Per-class composite | §5.8 |
| F5 | `operator_frequency_selected_cstar.png` | Operator frequency across the ten `C*` | §5.9 |
| F6 | `benchmark_comparison.png` | Anchor vs pure vs hybrid, four metrics, small multiples | §5.7 |
| F7 | `truck_failure_analysis.png` | Predicted class of 1,000 truck images under four models | §5.10 |
| F8–F17 | `results/search/plan_a_<class>/pareto_front_plan_a_<class>.png` | Ten per-class Pareto fronts | 2–3 in §5.6, remainder in Appendix |

Detailed captions and marking-criteria justification: `dissertation_table_figure_plan.md`.

**Figures that do not exist** — do not plan around them: seed-variance, ablation,
baseline-comparison and runtime plots. Each needs an experiment that was not run.

---

## 7. Key claims register

Every claim you intend to make, with the artefact that supports it. If a claim is not
in this table, it is not yet supported.

| # | Claim | Supporting file | Table / figure | Section |
|---|---|---|---|---|
| C1 | Instance-level forget sets carry no forget-specific structure (0.55% vs 1.00% chance) | `README.md`; predecessor project | F1 panel A | §1.3, §5.5 |
| C2 | Class-level forget sets do (84.1–91.2% of channels) | `results/analysis/class_structure/summary.json` | F1 panel A | §5.5 |
| C3 | All ten `W_ref` are valid: `D_f_test` = 0.0000, `D_r_test` = 0.9506 ± 0.0057 | `reference_model_validation_table.csv` | T1 | §5.2 |
| C4 | Pure MED-US: ACC_r 93.84 ± 1.15, ACC_f 12.55 ± 11.57, composite 82.09 ± 11.00, MIA 92.54 ± 8.62 | `ten_class_pure_mean_std.csv` | T2 | §5.6 |
| C5 | Retention is competitive with the anchor (0.35 behind) | `benchmark_comparison_table.csv` | T5, F6 | §5.7 |
| C6 | Forgetting is substantially worse than the anchor (12.52 points) | same | T5, F6 | §5.7 |
| C7 | The two harnesses agree on shared baselines (`W_0`, `W_ref` within 0.10 / 0.25) | same | T5 | §5.7 |
| C8 | Hybrid: 9 attempts, 9 accepted, 0 rejected | `refinement_acceptance_table.csv` | T6 | §5.8 |
| C9 | Hybrid buys −5.00 `ACC_f` for −0.12 `ACC_r` | `pure_vs_hybrid_summary_table.csv` | T4, F2, F3 | §5.8 |
| C10 | BatchNorm buffer movement exactly 0.000000 on all nine | `refinement_acceptance_table.csv` | T6 | §4.8, §5.8 |
| C11 | `MASK` appears in all ten selected `C*`; no other operator does | `pure_medus_10_class_table.csv` | F5 | §5.9 |
| C12 | Airplane reaches `ACC_f` 0.00, matching `W_ref` exactly | `pure_medus_10_class_table.csv` | T2 | §5.6 |
| C13 | Truck is the worst class on every headline metric, pure and hybrid | T2, T3 | F2, F4 | §5.10 |
| C14 | `W_ref` sends 68.40% of truck images to automobile; pure sends 16.70% and retains 42.10% | `truck_prediction_distribution.csv` | F7 | §5.10 |
| C15 | Truck's nearest neighbour in channel-contrast space is automobile (0.32), mutually | `class_structure_similarity.csv` | F1 panel C | §5.10 |
| C16 | Structure magnitude does **not** predict per-class difficulty (r = −0.04) | `build_class_structure_figure.py` output | F1 panel B | §5.5, §5.13 |
| C17 | Inter-class similarity does not predict difficulty either (r = −0.08); airplane is the counterexample | same | F1 panel C | §5.10, §5.13 |
| C18 | Zero search failures across ten runs, fronts of ten members each | `results/search/plan_a_*/summary.json` | — | §5.6, §5.11 |
| C19 | Cost: ~8 min search + ~10 min full fidelity per class, no retain-set training loop | `summary.json`, run logs | — | §5.11 |
| C20 | Search space is 3.87 × 10¹⁴ genomes at `max_level = 2` | `genome.py::search_space_size` | — | §4.3 |

---

## 8. Benchmark comparison summary

**What to say.** CIFAR-10 / ResNet-18, mean over all ten target classes — the same
aggregation the anchor uses, which is why all ten classes were swept rather than one.

| | pure MED-US | hybrid MED-US | Kodge et al. (reported) |
|---|---|---|---|
| `ACC_r` ↑ | 93.84 ± 1.15 | 93.72 ± 1.12 | 94.19 ± 0.50 |
| `ACC_f` ↓ | 12.55 ± 11.57 | 7.55 ± 8.87 | 0.03 ± 0.09 |
| MIA ↑ | 92.54 ± 8.62 | 95.05 ± 6.08 | 95.50 ± 14.23 |
| gradient-free | **yes** | **no** | yes |

Three sentences that carry the section:

- **Retention is competitive.** 0.35 points behind the anchor and inside the range of the
  published field; only NegGrad and SSD are materially worse on retention.
- **Forgetting is not.** 12.52 points worse than the anchor, and worse than UNSIR's
  10.89. UNSIR is the nearest published neighbour and, like this work, has a wide
  per-class spread rather than uniform behaviour.
- **Privacy is competitive on the anchor's own MIA** — but see the critique in §3.3
  before treating that as a win.

**Mandatory caveats.** The eight literature rows are *as reported* and were not
re-measured in this harness; no published baseline was re-run. The comparison inherits
every difference between two implementations. The supporting evidence is indirect: our
own `W_0` and `W_ref` reproduce the paper's Original and Retraining rows to within 0.10
and 0.25 of `ACC_r`.

**The MIA scepticism, worth its own paragraph.** In the same table SCRUB reaches `ACC_f`
0.00 with MIA 0.00, and Retraining is pinned at exactly 100.00. A metric where the gold
standard sits at the ceiling has limited discriminative power. This project's own MIA
AUC on the same models sits near 0.52–0.63 — far closer to chance than an anchor MIA of
92–100 implies. The two numbers cannot both describe the same thing.

---

## 9. Pure vs hybrid — separation rules

**Rule 1 — they are two methods, never one.** Pure MED-US applies no gradient at any
point. The hybrid applies one gradient-ascent step on `D_f` and one repair step on `D_r`
after the search has finished. Separate sections, separate tables, separate rows.

**Rule 2 — only pure is like-for-like with the anchor.** The anchor's method is
gradient-free. A hybrid row placed in its Table 1 would not be a fair comparison.

**Rule 3 — never quote 7.55 as "MED-US".** Quoting the hybrid's `ACC_f` as the project's
result would overstate what the gradient-free method achieves by 5.00 points of `ACC_f`
and 4.58 of composite.

**Rule 4 — name the temptation in the text.** The hybrid is better on every headline
metric. Say so, and say why the separation is kept anyway. An examiner who sees the
discipline being exercised deliberately will credit it; one who suspects it was
convenient will not.

**Rule 5 — the hybrid answers a narrower question.** Not "is MED-US good" but "what does
one constrained gradient step add to a gradient-free solution?" The answer is 5.00 points
of `ACC_f` for 0.12 of retain accuracy. That is a useful, separate result.

**Rule 6 — state the deployment cost.** The hybrid needs `D_f`, `D_r` and an optimiser
step at unlearning time, which removes part of the deployment argument for a
gradient-free method.

---

## 10. The truck failure — what to write

**The finding.** Truck is the worst class on every headline metric, before and after
refinement: pure `ACC_f` 42.10 against a ten-class mean of 12.55; composite 53.80 against
a next-worst of 78.10; MIA 69.60 against a next-worst of 91.10; selectivity 93.99 against
a next-worst of 154.57. Refinement helps truck **more than any other class** (−11.20
`ACC_f`, +10.24 composite, +9.50 MIA — the largest absolute gains in the sweep) and it is
*still* the worst class afterwards at 30.90.

**The mechanism, supported by two independent measurements.**

| evidence | what it shows |
|---|---|
| Prediction distribution (F7) | `W_ref` sends **68.40%** of truck images to automobile. Pure `C*` sends only 16.70% there and leaves 42.10% still called truck. Hybrid moves 11.2 points further out, of which most (20.40%) arrive at automobile. |
| Channel-contrast similarity (F1 panel C) | Truck's nearest neighbour is **automobile (0.32)**, and the relation is mutual — automobile's nearest is truck. |

Two measurements — one on `W_0` activations, one on the predictions of four different
models — agree on the same pair. **The failure is a partial move along the
truck–automobile axis, not a random scattering:** the reference's destination is the same
destination, reached less far.

**What must NOT be claimed.** That similarity *causes* the difficulty, or predicts it.
It does not: r = −0.08 across the ten classes, and **airplane is decisive against the
simple version** — it has the highest similarity to another class of any of the ten
(0.41, with ship) and still reaches `ACC_f` 0.00. Structure magnitude is ruled out too
(r = −0.04; truck is sixth of ten on median SNR, and automobile has the least structure
of any class while forgetting 3.3× better).

**The defensible claim, and it is still worth making.** The failure is stable and
reproducible, it is *not* explained by how much forget-specific structure the class has,
and it coincides with truck sharing more of its structure with a retained class than
with anything else. Why truck is hard remains open. Say that.

**Why this earns marks.** A method with a stable, identifiable, mechanistically
characterised failure mode is more useful than one that fails unpredictably. Treat truck
as a case study with its own subsection, not as an apology in a footnote.

---

## 11. Operator frequency — what to write

**The observation.** `MASK` appears in the selected `C*` of **all ten** classes. No other
operator appears in more than four. Three classes select `MASK` alone (deer, dog, horse);
five select it alone or with a single partner. `NOISE` appears in none.

**Why it is a finding and not a design choice.** All eight operators were available at
equal cost in all ten independent runs, with identical settings and no per-class tuning.
The search converged on `MASK` ten times out of ten.

**Why it is consistent with the project's mechanism.** `MASK` is the only editor operator
whose selector is the class-activation contrast — `|W| · (rms_f − rms_r)` — so it is the
one operator that can exploit the class-specific channels the structure analysis found.
`PRUNE` and `RANDOM_PRUNE` pin themselves to data-free selection (`magnitude`, `random`)
and are the deliberate controls: if a forget-informed operator could not beat them, the
"forget-informed" part would be doing nothing.

**The limit of the claim, which must be stated.** Ten runs at one seed is *convergent
evidence*, not proof. Without an operator ablation — running the search with `MASK`
removed, or with each family in isolation — this remains an observation about what the
search selected, not a demonstration that `MASK` is necessary. The ablation was not run.

---

## 12. Limitations and future work

Full version with what each limitation does and does not license:
`limitations_future_work_notes.md`. Summary:

| # | Limitation | Does not license | Cheapest fix |
|---|---|---|---|
| L1 | CIFAR-10 / ResNet-18 only, one `W_0` | Any claim about scale, other architectures, or class imbalance | CIFAR-100 superclasses on the same ResNet-18 |
| L2 | One seed (42) | Reading `±` as an uncertainty estimate | Extra *search* seeds on 2–3 classes — cheap, since `W_0` and `W_ref` are fixed and evaluation is deterministic |
| L3 | No baseline re-implemented | Claiming MED-US beats or loses to any specific method under controlled conditions | Finetune-on-`D_r`, NegGrad, random-relabel in this harness |
| L4 | `C*` selected on the test set it is reported on | Treating the headline numbers as unbiased | Halve the test sets into selection and reporting halves; re-run only the full-fidelity stage |
| L5 | The anchor MIA appears saturated | Treating MIA 95 as strong privacy evidence | A stronger attack (U-LiRA); already have our own AUC as contrast |
| L6 | No ablation of any kind | The claim that the evolutionary search is necessary; the `MASK` claim as a demonstrated property | Random search at equal evaluation budget — reuses the whole harness, only the sampler changes |
| L7 | Truck unexplained | Describing MED-US as reliable across classes | Overlap between edited channels and the channels a retained neighbour depends on — derivable from existing artefacts, untried |
| L8 | Runtime unoptimised (`num_workers: 0`, batch cap 3, serial population) | Wall-clock comparison against published runtimes | Not worth the effort; report the *shape* of the cost instead |
| L9 | Hybrid is gradient-based | Any deployment claim that assumes gradient-free | — (inherent to the method) |

---

## 13. What NOT to claim

1. **Do not claim MED-US beats the state of the art.** It does not. `ACC_f` 12.55 against
   the anchor's 0.03.
2. **Do not merge pure and hybrid**, or quote 7.55 as the project's `ACC_f`.
3. **Do not describe the hybrid as gradient-free.**
4. **Do not present `±` as run-to-run variance.** It is class-to-class spread at one seed.
5. **Do not claim a controlled comparison against NegGrad / SCRUB / SSD / Kodge.** No
   baseline was re-implemented.
6. **Do not claim structure predicts per-class difficulty.** Tested: r = −0.04. Null.
7. **Do not claim similarity predicts difficulty.** Tested: r = −0.08. Airplane refutes
   the simple version.
8. **Do not claim `MASK` is necessary or optimal.** No operator ablation was run.
9. **Do not claim a second architecture or dataset.** Only ResNet-18 / CIFAR-10 exists.
10. **Do not claim NSGA-II was compared against NSGA-III or random search.** Neither
    comparison was run. Design *reasoning* is fine if labelled as reasoning.
11. **Do not cite pymoo.** NSGA-II is hand-implemented; pymoo is not a dependency.
12. **Do not claim the MIA certifies privacy.** Report it, and critique it.
13. **Do not invent a citation.** Every reference must be traceable to
    `claudedocs/research_anchor_paper_20260827.md` or a paper you have actually read.

---

## 14. What still needs human checking

| # | Item | Why it needs you |
|---|---|---|
| H1 | **Word count and handbook compliance** | The outline assumes ~18,000 words. Confirm against your handbook and rescale the allocation. |
| H2 | **Citation style and completeness** | The anchor and the seven comparison papers are recorded with DOIs/links in `claudedocs/research_anchor_paper_20260827.md`. Every other citation must be one you have read. |
| H3 | **The `C*` selection-on-test issue** | Decide whether to fix it (≈2 h of full-fidelity re-measurement) or disclose it. Disclosing is acceptable; discovering it in a viva is not. |
| H4 | **Whether to attempt any ablation before submission** | Random search at equal budget is ~4 h and defends the whole method choice. Your call on time. |
| H5 | **Attribution for the distributed reference training** | Three people trained the ten references across seven Kaggle bundles. Confirm how you wish to credit Aditya and Pragati in the acknowledgements and in §5.2. |
| H6 | **The `S = nan` defect narrative** | Verify you are comfortable reporting it. It affected only the diagnostic `best_S` row for deer, dog and horse; no headline number changed. Reporting it is a strength. |
| H7 | **Supervisor's view on reporting a negative result as a contribution** | Panel B of F1 is a null. This pack treats it as a finding. Confirm your supervisor agrees with that framing. |
| H8 | **Figure sizing for print** | All figures are 300 dpi PNG sized for on-screen reading. Check they hold at your page width, particularly F1 (three panels) and F6/F7 (four panels each). |
| H9 | **Ethics form** | Public dataset, no human subjects — but the school's form still needs submitting. |
| H10 | **Every number, once more, at final draft** | Copy from the CSVs, not from this file or from memory. This pack was verified on 2026-08-31; re-verify at submission. |

---

*Sources for this pack, all committed: `results/writeup_package/*`,
`results/literature_alignment/*`, `results/analysis/class_structure/*`,
`results/search/plan_a_*/summary.json`, `README.md`, `docs/artifact_manifest.md`,
`claudedocs/research_anchor_paper_20260827.md`. No experiment was run.*
