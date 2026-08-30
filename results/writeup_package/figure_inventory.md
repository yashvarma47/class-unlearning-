# Figure inventory

Generated 2026-08-30T20:45:08+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Every figure listed here already exists in the repository. No figure was regenerated
while building this package, and none is missing.

## 1. Per-class Pareto fronts -- ten figures, all present

| id | class | path | plot data CSV |
|---:|:---|:---|:---:|
| 0 | airplane | `results/search/plan_a_airplane/pareto_front_plan_a_airplane.png` | yes |
| 1 | automobile | `results/search/plan_a_automobile/pareto_front_plan_a_automobile.png` | yes |
| 2 | bird | `results/search/plan_a_bird/pareto_front_plan_a_bird.png` | yes |
| 3 | cat | `results/search/plan_a_cat/pareto_front_plan_a_cat.png` | yes |
| 4 | deer | `results/search/plan_a_deer/pareto_front_plan_a_deer.png` | yes |
| 5 | dog | `results/search/plan_a_dog/pareto_front_plan_a_dog.png` | yes |
| 6 | frog | `results/search/plan_a_frog/pareto_front_plan_a_frog.png` | no (figure only) |
| 7 | horse | `results/search/plan_a_horse/pareto_front_plan_a_horse.png` | yes |
| 8 | ship | `results/search/plan_a_ship/pareto_front_plan_a_ship.png` | yes |
| 9 | truck | `results/search/plan_a_truck/pareto_front_plan_a_truck.png` | yes |

Each shows the ten-member non-dominated front for that class with `C*` marked.
`pareto_front_plot_data.csv` beside each figure holds the plotted values, so any
figure can be restyled for print without re-running anything.

**Note:** `results/search/plan_a_bird/` has the figure but no
`pareto_front_plot_data.csv`. If a restyled bird figure is needed, the underlying
values are still available in that run's `full_fidelity/front_full_fidelity.csv`.

**Dissertation use.** Do not print all ten in the results chapter. Put two or three
in the body -- **dog** (the strongest front, `S` = 4427.91), **truck** (the weakest,
`S` = 93.99), and optionally **airplane** (the only class reaching `ACC_f` 0.00) --
and move the remaining seven to Appendix E. Three fronts side by side make the
class-wise spread visible in a way the table cannot; ten in sequence makes a reader
skip the section.

## 2. Pure vs hybrid plots -- NOT YET AVAILABLE

No pure-vs-hybrid figure exists in the repository. The numbers behind one do:

- `results/literature_alignment/pure_vs_hybrid_comparison.csv` -- per-class deltas
- `results/writeup_package/pure_vs_hybrid_summary_table.csv` -- the aggregate

The obvious figure is a paired slope chart or grouped bar of per-class `ACC_f`,
pure against hybrid, with truck's 42.10 -> 30.90 as the visible outlier. It would be
built from the CSVs above and requires no new experiment. It is not built here
because plotting was outside this task's scope.

## 3. Class-structure figures -- data present, figures not

`results/analysis/class_structure/` holds `channel_contrast_all_classes.csv`,
`per_class_groups.csv` and `summary.json`. These are the measurements that motivate
the whole project -- per-channel activation contrast against a null control -- and
they are the natural first figure of the results chapter. No plot of them exists yet.

## 4. Figures that do not exist and would need new work

Named here so they are not assumed available: seed-variance plots, ablation plots,
baseline comparison plots, and runtime charts. Each needs experiments that have not
been run.
