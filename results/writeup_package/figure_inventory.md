# Figure inventory

Generated 2026-08-30T22:55:44+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Every figure listed as present was checked on disk while this file was written.

## 1. Per-class Pareto fronts -- ten figures, all present

| id | class | path | plot data CSV |
|---:|:---|:---|:---:|
| 0 | airplane | `results/search/plan_a_airplane/pareto_front_plan_a_airplane.png` | yes |
| 1 | automobile | `results/search/plan_a_automobile/pareto_front_plan_a_automobile.png` | yes |
| 2 | bird | `results/search/plan_a_bird/pareto_front_plan_a_bird.png` | yes |
| 3 | cat | `results/search/plan_a_cat/pareto_front_plan_a_cat.png` | yes |
| 4 | deer | `results/search/plan_a_deer/pareto_front_plan_a_deer.png` | yes |
| 5 | dog | `results/search/plan_a_dog/pareto_front_plan_a_dog.png` | yes |
| 6 | frog | `results/search/plan_a_frog/pareto_front_plan_a_frog.png` | yes (`_plot_table`) |
| 7 | horse | `results/search/plan_a_horse/pareto_front_plan_a_horse.png` | yes |
| 8 | ship | `results/search/plan_a_ship/pareto_front_plan_a_ship.png` | yes |
| 9 | truck | `results/search/plan_a_truck/pareto_front_plan_a_truck.png` | yes |

Each shows the ten-member non-dominated front for that class with `C*` marked.
`pareto_front_plot_data.csv` beside each figure holds the plotted values, so any figure
can be restyled for print without re-running anything.

**Dissertation use.** Do not print all ten in the results chapter. Put two or three in
the body -- **dog** (the strongest front, `S` = 4427.91), **truck** (the weakest,
`S` = 93.99), and optionally **airplane** (the only class reaching `ACC_f` 0.00) -- and
move the remaining seven to Appendix E. Three fronts side by side make the class-wise
spread visible in a way the table cannot; ten in sequence makes a reader skip the
section.

## 2. Write-up figures -- 7 of 7 present

All in `results/writeup_package/figures/`, 300 dpi PNG. The first six are built by
`experiments/build_writeup_figures.py` from the CSVs in this package;
`class_structure_analysis.png` is built by
`experiments/build_class_structure_figure.py` from
`results/analysis/class_structure/`.

| file | what it shows | size |
|:---|:---|---:|
| `pure_vs_hybrid_acc_f_by_class.png` | Pure vs hybrid ACC_f, all ten classes (dumbbell) | 204 KB |
| `pure_vs_hybrid_acc_r_by_class.png` | Pure vs hybrid ACC_r, all ten classes (dumbbell) | 219 KB |
| `pure_vs_hybrid_composite_by_class.png` | Pure vs hybrid composite, all ten classes (dumbbell) | 205 KB |
| `operator_frequency_selected_cstar.png` | Operator frequency across the ten selected C* (bar) | 136 KB |
| `benchmark_comparison.png` | Anchor vs pure vs hybrid on four metrics (dot, small multiples) | 240 KB |
| `truck_failure_analysis.png` | Predicted class of the 1,000 truck test images under four models (bar, small multiples) | 251 KB |
| `class_structure_analysis.png` | Structure per class, structure against difficulty, and the inter-class similarity matrix (bar + scatter + heatmap) | 561 KB |

### What each is for

**`pure_vs_hybrid_acc_f_by_class.png`** -- Results, the hybrid section. The clearest single picture of what the refinement buys, and of how far truck sits from everything else.

**`pure_vs_hybrid_acc_r_by_class.png`** -- Results, beside the ACC_f figure, to show the cost side of the same trade. Say in the caption that the axis is zoomed -- the whole ten-class range is three points.

**`pure_vs_hybrid_composite_by_class.png`** -- Results, as the summary of the previous two on the anchor's own metric. If only one of the three goes in the body, use this one and appendix the others.

**`operator_frequency_selected_cstar.png`** -- Results or Discussion, as the evidence for the MASK finding. Pairs with the caveat that ten runs at one seed is convergent evidence, not an ablation.

**`benchmark_comparison.png`** -- Results, the benchmark section. Carries the 'retention competitive, forgetting not' reading in one image.

**`truck_failure_analysis.png`** -- Discussion, the truck section. This is the figure that turns truck from a confession into a finding.

**`class_structure_analysis.png`** -- Results, as the FIRST figure of the chapter -- it is the measurement the whole project rests on. Panel B is also the honest answer to RQ3 and belongs in the discussion beside the truck section; do not caption it as though structure predicts difficulty, because it does not.

### Design notes worth carrying into the captions

Figures 1-3 share one class order -- by pure `ACC_f`, worst first -- so they read across
as a set. Figures 1-3 and 5 use dots rather than bars because those panels need a zoomed
axis, and a bar drawn from a non-zero baseline misstates the ratio between its
neighbours; a dot encodes position only, so the zoom is honest. Every mark is directly
labelled, so no reading depends on colour alone. The palette is the one already used by
the ten Pareto figures, so both sets read as one system in the same document; it is
validated for categorical use on a light surface, and light-surface only, which is what
a printed page is.

## 3. Class-structure figure -- built

`class_structure_analysis.png`, three panels, from
`results/analysis/class_structure/summary.json` and
`channel_contrast_all_classes.csv` by `experiments/build_class_structure_figure.py`.
The 10x10 matrix behind panel C is written out as `class_structure_similarity.csv`
beside this file.

It is the measurement the whole project rests on, so it belongs first in the results
chapter -- but read panel B before writing the caption. Structure explains the
**regime** (class-level forget sets have it, instance-level ones do not) and not the
**ranking** (it does not predict which class is hard). See
`missing_figures_status.md` for what the panels do and do not establish.

## 4. Figures that do not exist and would need new experiments

Named here so they are not assumed available: seed-variance plots, ablation plots
(random search at equal budget, operator families in isolation), baseline comparison
plots, and runtime charts. Each needs work that has not been run.

See `missing_figures_status.md` for what was attempted, what succeeded, and what did not.
