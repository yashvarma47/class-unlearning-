# Chapter 5 evidence map

Every numeric claim in `dissertation/chapters/05_experimental_study.tex`, traced to
the committed artefact it came from. Verified 2026-08-31 against the repository at
commit `3f7169d`.

> **Revision note, 2026-08-31.** Chapter 5 was revised for style and length after
> this map was first written. The revision changed wording only: no number was
> added, removed or altered, no claim was introduced, and no source file changed.
> Every entry below remains valid. Three rows were **added** to close pre-existing
> gaps (CUDA runtime version, and the 1.25 loss-ratio and 0.3 edit-cost acceptance
> thresholds); nothing was removed. One subsection heading changed, from "Where
> truck images actually go" to "The destination of forgotten truck images"; this
> map is organised by section number rather than by heading, so no cross-reference
> was affected.

**How to use this.** Before final submission, re-read each source file and confirm
each value. Copy numbers from the CSV, never from this map and never from memory.

**Path convention.** `WP/` = `results/writeup_package/` ·
`LA/` = `results/literature_alignment/` · `CS/` = `results/analysis/class_structure/` ·
`SR/` = `results/search/`.

---

## 5.2 Experimental Setup

| claim in text | value | source |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4.0 GB VRAM, CC 7.5 | `torch.cuda.get_device_properties(0)`, recorded in `docs/dissertation_context_dump.md` §9.1 |
| CPU | AMD Ryzen 5 3550H, 4 cores / 8 threads | `Win32_Processor` query, same source |
| RAM | 13.9 GB | `Win32_ComputerSystem`, same source |
| OS | Windows 11, build 10.0.26200 | `platform.platform()` |
| Python / PyTorch / torchvision | 3.11, 2.5.1+cu121, 0.20.1 | `requirements-torch.txt`; runtime check |
| CUDA runtime 12.1 | 12.1 | `torch.version.cuda`; `requirements-torch.txt` cu121 wheel index |
| fp32 440 samples/s, finite loss | 440 | `configs/base.yaml`, `device.amp` comment block |
| fp16 AMP 165 samples/s, NaN in `layer4` | 165 | same |
| Reference training on Tesla T4 | Tesla T4, 14.56 GB | `WP/../reference_training/class*_kaggle_manifest.json` → `environment.device.gpu_name` |
| Seed 42 for every search | 42 | `configs/search/plan_a_*.yaml`, `search.seed` |
| Deterministic algorithms enforced | `deterministic: true` | `configs/base.yaml` |
| Per-generation checkpointing incl. RNG | `checkpoint_every_generation: true` | `configs/search/plan_a_*.yaml`; `src/medus_class/search/nsga2.py` `run(resume_from=...)` docstring |

---

## 5.3 Dataset, Model, and Forgetting Protocol

| claim | value | source |
|---|---|---|
| `D_f` train | 5,000 | `results/splits/cifar10_class*.json` → `n_forget_train`; `WP/reference_model_validation_table.csv` col `D_f_train` |
| `D_r` train | 45,000 | same, `n_retain_train` / `D_r_train` |
| `D_f` test | 1,000 | same, `n_forget_test` / `D_f_test` |
| `D_r` test | 9,000 | same, `n_retain_test` / `D_r_test` |
| Counts identical for every class | — | all ten split JSONs carry the same four values |
| Split needs no seed, no stratification | — | `src/medus_class/data/class_split.py::build_class_split` (label determines partition) |
| Splits byte-compared before import | — | `kaggle/reference_training/validate_reference_zip.py`; `docs/artifact_manifest.md` §7 |
| Total trainable parameters | 11,173,962 | computed from `build_model` on `configs/model/resnet18.yaml`; recorded in `docs/dissertation_context_dump.md` §3.2 |
| `layer4` parameters | 8,393,728 | same, §3.4 |
| `layer4` share | 75.12% | same |
| Six layer groups | stem, layer1–layer4, fc | `configs/model/resnet18.yaml`, `model.layer_groups` |
| `W_0` training: 200 epochs, SGD 0.1, momentum 0.9, wd 5e-4, step ×0.1 every 40 | — | `configs/model/resnet18.yaml`, `training` block |
| `W_0` test accuracy 94.79% | 0.9479 | `results/checkpoints/cifar10_resnet18_seed42_best.json` → `metrics.test_acc`. **Note:** stated as 94.79% in text; the checkpoint sidecar records 0.9479 and the ten-class anchor mean is 94.79 ± 0.29 (`WP/benchmark_comparison_table.csv`). Both agree to 2 dp. |

---

## 5.4 Reference Model Validation

| claim | value | source |
|---|---|---|
| All ten `W_ref` validated, verdict PASS | 10 × `PASS` | `WP/reference_model_validation_table.csv`, col `verdict` |
| `D_f_test` accuracy 0.0000 for all ten | 0.0000 | same, col `D_f_test_acc` |
| `D_r_test` accuracy 0.9506 ± 0.0057 | mean 0.95061, sd 0.00568 | computed from same file, col `D_r_test_acc`, ddof=1; also stated in `WP/reference_model_validation_table.md` |
| Range 0.9423 (horse) to 0.9612 (cat) | — | same file |
| 200 epochs, seed 42 | 200, 42 | same, cols `log_epochs`, `seed` |
| ~2.32 h per model on T4 | 2.319 / 2.321 h | `results/reference_training/class9_truck_kaggle_manifest.json` → `trained_classes[].hours` |
| Selection on `D_r_test` acc, tie-break `D_r_test` loss | — | `results/reference_training/class*_training_summary.md`, "selection rule" row |
| `D_f_test` logged but excluded from selection | — | same |
| Three people, seven Kaggle bundles, one local | — | `WP/reference_model_validation_table.csv` col `source_zip`; frog = "(local training, never packaged)" |
| SHA-256 recorded per checkpoint | 12-char prefixes | `results/reference_training/all_reference_models_summary.md` |
| Full-test accuracy near 0.85 | 0.8481–0.8651 | `WP/reference_model_validation_table.csv`, col `full_test_acc` |

---

## 5.5 Pure MED-US Results

### Search configuration and behaviour

| claim | value | source |
|---|---|---|
| Population 10, 50 generations | 10, 50 | `configs/search/plan_a_*.yaml`; `SR/plan_a_*/summary.json` |
| Budget 510 evaluations | 10 + 50×10 | derived from the two above |
| Zero failed evaluations, all ten runs | `failures: 0` | `SR/plan_a_*/summary.json`, all ten |
| Front size 10 for every class | `front_size: 10` | same |
| Real evaluations 212 to 319 | min 212 (deer), max 319 (cat) | same, `evaluated` |
| Cache hits 191 to 298 | min 191 (cat), max 298 (deer) | same, `cache_hits` |
| Cache hit rate 37% to 58% | 191/510 = 37.5%, 298/510 = 58.4% | derived |
| `C*` rule = max composite `ACC_r × (1 − ACC_f)` | — | `experiments/run_class_sweep.py`, `composite()` and line 199 |

### Aggregate

| metric | value | source |
|---|---|---|
| `ACC_r` | **93.84 ± 1.15** | `LA/ten_class_pure_mean_std.csv`, row `anchor_ACC_r` (93.8433 / 1.1460) |
| `ACC_f` | **12.55 ± 11.57** | same, row `anchor_ACC_f` (12.5500 / 11.5658) |
| composite | **82.09 ± 11.00** | same, row `anchor_composite` (82.0868 / 11.0003) |
| MIA | **92.54 ± 8.62** | same, row `anchor_MIA` (92.5400 / 8.6171) |
| selectivity `S` | 1053.88 ± 1413.47 | same, row `selectivity_S` |
| std is sample std, ddof=1 | — | `experiments/build_writeup_package.py::agg`; cross-checked against the committed table |

### Spread claims

| claim | value | source |
|---|---|---|
| `ACC_r` range 92.52 (frog) to 95.62 (dog) = 3.10 | 3.10 | computed from `WP/pure_medus_10_class_table.csv` |
| `ACC_f` range 0.00 (airplane) to 42.10 (truck) = 42.10 | 42.10 | same |
| Airplane `ACC_f` 0.00, composite 92.86, MIA 100.00 | — | same, row `class_id = 0` |
| Truck `ACC_f` 42.10 | 42.10 | same, row `class_id = 9` |
| Next worst after truck is bird at 17.90 | 17.90 | same; sorted `ACC_f`: 0.00, 3.30, 7.80, 8.30, 9.60, 9.80, 12.70, 14.00, 17.90, 42.10 |
| Gap truck–bird (24.20) exceeds gap bird–dog (14.60) | derived | same. **Text wording:** "larger than the gap between bird and the best non-airplane class" (bird 17.90 − dog 3.30 = 14.60). Confirmed. |
| `S` range 93.99 (truck) to 4427.91 (dog) | — | same, col `selectivity_S` |
| Instance-level ceiling `S` = 1.158 over 10,534 strategies | 1.158 / 10,534 | `README.md`, "Why this project exists" table (predecessor project) |
| Every class-level `S` exceeds 1.158 by ≥ 1 order of magnitude | min 93.99 | derived: 93.99 / 1.158 = 81× |
| Strongest exceeds by > 3 orders | 4427.91 / 1.158 = 3824× | derived |

### Class structure (Figure 5.1)

| claim | value | source |
|---|---|---|
| 84.1% to 91.2% of channels above noise floor | min 84.15 (horse), max 91.22 (ship) | `CS/summary.json`, `ranked[].pct_beyond_noise` |
| Instance-level 0.55%, null control 1.00% | 0.55 / 1.00 | `README.md`, "Why this project exists" |
| Pearson r(median SNR, `ACC_f`) = −0.04 | −0.037 | computed in `experiments/build_class_structure_figure.py::pearson`; printed on build |
| Truck sixth of ten on median SNR | rank 6 (7.82) | `CS/summary.json`, `ranked` order |
| Truck fifth on channels above floor | rank 5 (88.72) | same, sorted by `pct_beyond_noise` |
| Automobile least structure (3.90 median SNR) | 3.90 | same |
| Automobile forgets 3.3× better than truck | 42.10 / 12.70 = 3.31 | derived from `WP/pure_medus_10_class_table.csv` |

---

## 5.6 Benchmark Comparison

| claim | value | source |
|---|---|---|
| Kodge `ACC_r` | **94.19 ± 0.50** | `WP/benchmark_comparison_table.csv`, row `Kodge et al. 2024 (anchor)`, `measured_in_this_harness = no` |
| Kodge `ACC_f` | **0.03 ± 0.09** | same |
| Kodge MIA | **95.50 ± 14.23** | same |
| Original (reported) 94.89 / 94.89 / 0.03 | — | same file, row `Original` |
| Retraining (reported) 94.81 / 0.00 / 100.00 | — | same file, row `Retraining (gold standard)` |
| UNSIR `ACC_f` 10.89 ± 8.79 | — | same file, row `UNSIR (Tarun et al. 2023)` |
| Retention gap 0.35 | 94.19 − 93.84 = 0.35 | derived |
| Forgetting gap 12.52 | 12.55 − 0.03 = 12.52 | derived |
| `W_0` measured here: 94.79 ± 0.29 / 94.79 ± 2.59 / 0.00 ± 0.00 | — | `WP/benchmark_comparison_table.csv`, row `Original W_0 (this work)` |
| `W_ref` measured here: 95.06 ± 0.57 / 0.00 ± 0.00 / 100.00 ± 0.00 | — | same, row `Retraining W_ref (this work)` |
| Agreement within 0.10 of `ACC_r` on Original | \|94.79 − 94.89\| = 0.10 | derived |
| Agreement within 0.25 of `ACC_r` on Retraining | \|95.06 − 94.81\| = 0.25 | derived |
| `ACC_f` and MIA agree to 0.10 and 0.03 | \|94.79 − 94.89\|, \|0.00 − 0.03\| | derived |
| No baseline re-implemented in this repo | — | `WP/dissertation_writing_context_pack.md` §7; no baseline module exists under `src/` or `experiments/` |
| Anchor protocol read from released source, not paper prose | — | `src/medus_class/evaluation/anchor.py` docstring |
| Composite = `ACC_r × (1 − ACC_f)`, MIA not a term | — | same docstring, quoting `utils.py::metric_function` |

**Citation placeholders used:** `[CITE: Kodge et al., year]`, `[CITE: Tarun et al., year]`,
`[CITE: Fan et al., year]`. Full bibliographic details for all three, plus four further
comparison studies, are in `claudedocs/research_anchor_paper_20260827.md`. **No citation
was invented; each placeholder names a study recorded in that file.**

---

## 5.7 Hybrid BN-Frozen Refinement Results

| claim | value | source |
|---|---|---|
| One ascent step on `D_f`, one repair step on `D_r` | — | `SR/plan_a_*_bn_frozen_refined/refinement.json`, `hyperparameters.forget_step` / `.retain_step` |
| BatchNorm frozen (eval mode both steps) | — | same, `hyperparameters.batchnorm` |
| Eight batches per step | 8 | same, `hyperparameters.batches_per_step` |
| lr 1e-4 both steps | 0.0001 | same, `forget_lr` / `retain_lr` |
| Six acceptance checks | 6 | same, `acceptance_checks` (6 keys) |
| Check 3 threshold: retain losses <= 1.25x | 1.25 | same, `hyperparameters.max_loss_ratio` |
| Check 4 threshold: edit cost <= 0.3 | 0.3 | same, `hyperparameters.max_edit_cost` |
| Check 2 threshold: `D_r_test` drop <= 0.010 | 0.01 | same, `hyperparameters.max_retain_test_drop` |
| Nine attempts, nine accepted, zero rejected | 9 / 9 / 0 | `WP/refinement_acceptance_table.csv`, col `accepted` |
| Airplane not attempted | `attempted = no` | same, row `class_id = 0` |
| Buffer movement exactly 0.000000, all nine | 0.000000 | same, col `buffer_movement` |
| BN counters changed = 0, all nine | 0 | same, col `batchnorm_counters_changed` |
| Parameter movement 0.000303 (cat) to 0.000420 (frog) | — | same, col `parameter_movement` |
| Budget 0.0400 | 0.040000 | same, col `movement_budget` |
| 0.8% to 1.1% of budget used | 0.000303/0.04 = 0.76%, 0.000420/0.04 = 1.05% | derived |
| Hybrid `ACC_r` | **93.72 ± 1.12** | `LA/ten_class_hybrid_mean_std.csv`, row `anchor_ACC_r` (93.7222 / 1.1201) |
| Hybrid `ACC_f` | **7.55 ± 8.87** | same, row `anchor_ACC_f` (7.5500 / 8.8680) |
| Hybrid composite | **86.66 ± 8.52** | same, row `anchor_composite` (86.6649 / 8.5219) |
| Hybrid MIA | **95.05 ± 6.08** | same, row `anchor_MIA` (95.0500 / 6.0813) |
| The unfrozen-BN silent failure | — | `docs/artifact_manifest.md` §9; commit `49a962b` message; `WP/refinement_acceptance_table.md` "Why check 6 exists" |

---

## 5.8 Pure vs Hybrid Trade-off

| claim | value | source |
|---|---|---|
| Δ`ACC_f` −5.00 | −5.0000 | `WP/pure_vs_hybrid_summary_table.csv`, row `ACC_f (%)`, col `delta_mean` |
| Δ`ACC_r` −0.12 | −0.1211 | same, row `ACC_r (%)` |
| Δcomposite +4.58 | +4.5781 | same, row `composite (%)` |
| ΔMIA +2.51 | +2.5100 | same, row `anchor MIA (%)` |
| Nine of ten improved on composite | 9 | `LA/pure_vs_hybrid_comparison.csv`, count of `delta_composite > 0` |
| Tenth is airplane no-op | `refinement_status = no-op` | same, row `class_id = 0` |
| No class regressed on any headline metric | — | same file: `delta_ACC_f ≤ 0`, `delta_composite ≥ 0`, `delta_MIA ≥ 0` for all ten |
| Truck Δ`ACC_f` −11.20, Δcomposite +10.24, ΔMIA +9.50 | −11.2000 / +10.2448 / +9.5000 | same, row `class_id = 9` |
| Ship Δ`ACC_f` −9.30, 14.00 → 4.70, hybrid composite 90.05 | — | same, row `class_id = 8`; `WP/hybrid_medus_10_class_table.csv` |
| Dog Δ`ACC_f` −1.40 from 3.30 | −1.4000 | same, row `class_id = 5` |
| Largest retain cost: cat, −0.2444 | −0.2444 | same, col `delta_ACC_r`, min |
| Frog is the one class whose `ACC_r` improves, +0.0333 | +0.0333 | same, row `class_id = 6` |
| No class exceeded the 0.010 retain-drop check | max `retain_test_drop` = 0.002444 (cat) | `WP/refinement_acceptance_table.csv` |
| Overstatement if merged: 5.00 `ACC_f` and 4.58 composite | derived | from the delta row above |

---

## 5.9 Operator Frequency Analysis

| claim | value | source |
|---|---|---|
| MASK 10, CLIP 4, DAMP 3, PRUNE 2, RANDOM_PRUNE 2, RESET 2, QUANTIZE 1 | — | recomputed from `WP/pure_medus_10_class_table.csv`, col `operators`, split on `\|` |
| NOISE appears in none | 0 | same |
| MASK in all ten; only operator with that property | 10 | same |
| Three classes select MASK alone: deer, dog, horse | 3 | same, rows where `operators == "MASK"` |
| Five select MASK alone or with one partner | 5 | same, rows where `len(operators.split("\|")) ≤ 2` |
| Eight operators available at equal cost | 8 | `configs/operators/lookup.yaml`: 3 editor + 5 smoother |
| MASK selector is class-activation contrast | `\|W\| · (rms_f − rms_r)` | `configs/operators/lookup.yaml`, id 0 `atomic_action`; `src/medus_class/operators/selection.py` |
| PRUNE pins to magnitude, RANDOM_PRUNE to random | — | `configs/operators/lookup.yaml`, ids 1 and 2, `requires: []` and comments |
| No operator ablation was run | — | `WP/limitations_future_work_notes.md` item 6; `WP/missing_figures_status.md` |

---

## 5.10 Truck Failure Analysis

| claim | value | source |
|---|---|---|
| Truck pure `ACC_f` 42.10 | 42.10 | `WP/pure_medus_10_class_table.csv`, row `class_id = 9` |
| Ten-class mean `ACC_f` 12.55 | 12.55 | `LA/ten_class_pure_mean_std.csv` |
| Truck composite 53.80; next worst bird 78.10 | — | `WP/pure_medus_10_class_table.csv`, sorted composite |
| Truck MIA 69.60; next worst horse 91.10 | — | same, sorted MIA: 69.6, 91.1, 91.7, 91.7, 94.2, 95.1, 96.0, 97.0, 99.0, 100.0 |
| Truck `S` 93.99, lowest of ten, only value below 100 | — | same, sorted `S`: 93.99, 154.57, 168.69, … |
| Truck hybrid `ACC_f` 30.90 | 30.90 | `WP/hybrid_medus_10_class_table.csv`, row `class_id = 9` |
| Hybrid ten-class mean 7.55 | 7.55 | `LA/ten_class_hybrid_mean_std.csv` |
| Truck sixth of ten on median SNR | 7.82, rank 6 | `CS/summary.json` |
| Truck fifth on channels above floor | 88.72, rank 5 | same |
| r(structure, `ACC_f`) = −0.04 | −0.037 | `experiments/build_class_structure_figure.py` |
| Automobile forgets 3.3× better | 42.10 / 12.70 = 3.31 | derived |
| `W_0` → 95.40% truck | 95.40 | `WP/truck_prediction_distribution.csv`, model `W_0`, class `truck` |
| `W_ref` → 68.40% automobile | 68.40 | same, model `W_ref`, class `automobile` |
| `W_ref` → 0.00% truck | 0.00 | same |
| Pure `C*` → 42.10% truck, 16.70% automobile | 42.10 / 16.70 | same, model `C_star_pure` |
| Hybrid → 30.90% truck, 20.40% automobile | 30.90 / 20.40 | same, model `C_star_hybrid` |
| Rebuilt `C*` reproduces published `ACC_f` exactly | 42.10 and 30.90 | `experiments/analyse_truck_predictions.py` console output; matches the two tables above |
| Inference only, nothing trained | — | `experiments/analyse_truck_predictions.py` docstring; `@torch.no_grad()` |
| `C*` has no stored checkpoint, rebuilt from chromosome | — | same; `report_anchor_metrics.py::rebuild_candidate` verifies against the recorded front row |
| Truck ↔ automobile similarity 0.32, mutual | 0.322779 | `WP/class_structure_similarity.csv` |
| r(max similarity, `ACC_f`) = −0.08 | −0.084 | `experiments/build_class_structure_figure.py` |
| Airplane ↔ ship 0.41, highest of any pair | 0.413182 | `WP/class_structure_similarity.csv` |
| Airplane `ACC_f` 0.00 | 0.00 | `WP/pure_medus_10_class_table.csv` |
| Matrix recovers semantic grouping | vehicles / animals cluster | same matrix; cat–dog 0.30, deer–frog 0.30, bird–deer 0.29 |

---

## 5.11 Summary of Experimental Findings

All figures in this section are restatements of values traced above. Additional
claims:

| claim | value | source |
|---|---|---|
| Total search time ≈ 1.5 h across ten classes | 5,441 s = 1.51 h | sum of `elapsed_seconds` over `SR/plan_a_*/summary.json` |
| Search 6.4 to 15.5 min per class | 383.6 s (deer) to 931.8 s (airplane) | same |
| Full-fidelity re-measurement 9.3 to 10.9 min per class | — | `results/plan_a_*sweep.out`, `plan_a_*fullfidelity.out`, "timing mean … total" lines |
| End to end 30 to 35 min per class | 30, 35, 31, 32 min | consecutive sweep-driver log mtimes (`results/plan_a_*_sweep.out`) |
| Reference training 2.32 h per model | 2.319 / 2.321 | `class9_truck_kaggle_manifest.json` |
| `num_workers: 0`, `batch_cap: 3`, serial population | — | `configs/search/plan_a_*.yaml`, `evaluation` block |
| Anchor MIA saturation: retraining 100.00, SCRUB 0.00 with `ACC_f` 0.00 | — | `WP/benchmark_comparison_table.csv`, rows `Retraining (gold standard)` and `SCRUB (Kurmanji et al. 2023)` |
| Own MIA AUC far closer to chance | 0.5174 to 0.6294 | `WP/pure_medus_10_class_table.csv` has no AUC column; values are in `SR/plan_a_*/<class>_anchor_metrics.json` → `rows.C_star.mia_auc`, and in `LA/ten_class_pure_summary.csv` col `our_mia_auc` |
| `C*` selected on the same test sets it is reported on | — | `experiments/run_class_sweep.py`, `composite()` reads `retain_test_acc` / `forget_test_acc`; selection at line 199 |

---

## Tables referenced (6)

| label in chapter | placeholder name | source CSV | rendered MD |
|---|---|---|---|
| `tab:reference-validation` | reference model validation table | `WP/reference_model_validation_table.csv` | `.md` |
| `tab:pure-ten-class` | pure MED-US 10-class table | `WP/pure_medus_10_class_table.csv` | `.md` |
| `tab:benchmark` | benchmark comparison table | `WP/benchmark_comparison_table.csv` | `.md` |
| `tab:refinement-acceptance` | refinement acceptance table | `WP/refinement_acceptance_table.csv` | `.md` |
| `tab:hybrid-ten-class` | hybrid MED-US 10-class table | `WP/hybrid_medus_10_class_table.csv` | `.md` |
| `tab:pure-vs-hybrid` | pure vs hybrid summary table | `WP/pure_vs_hybrid_summary_table.csv` | `.md` |

## Figures referenced (6)

| label in chapter | file | exists |
|---|---|---|
| `fig:class-structure` | `WP/figures/class_structure_analysis.png` | yes |
| `fig:benchmark` | `WP/figures/benchmark_comparison.png` | yes |
| `fig:pvh-accf` | `WP/figures/pure_vs_hybrid_acc_f_by_class.png` | yes |
| `fig:pvh-accr` | `WP/figures/pure_vs_hybrid_acc_r_by_class.png` | yes |
| `fig:pvh-composite` | `WP/figures/pure_vs_hybrid_composite_by_class.png` | yes |
| `fig:operator-frequency` | `WP/figures/operator_frequency_selected_cstar.png` | yes |
| `fig:truck-failure` | `WP/figures/truck_failure_analysis.png` | yes |

All seven exist and are committed. The ten per-class Pareto-front figures under
`SR/plan_a_*/` are **not** referenced in this chapter draft; the table/figure plan
recommends placing two or three of them in section 5.5 and the rest in an appendix.
Consider adding them.

---

## `[CHECK]` items in the chapter

**None.** Every numeric claim in the draft resolved to a committed artefact, so no
`[CHECK: evidence needed]` marker was required.

---

## Evidence that was requested but does not exist

| requested source | status |
|---|---|
| `WP/dissertation_word_page_plan.md` | **does not exist** |
| `WP/revised_18k_dissertation_outline.md` | **does not exist** |

Both were named conditionally in the task. The chapter was written from the five
planning documents that do exist: `dissertation_writing_context_pack.md`,
`dissertation_table_figure_plan.md`, `revised_dissertation_outline.md`,
`key_numbers_summary.md` and `results_chapter_notes.md`, plus
`limitations_future_work_notes.md`, `figure_inventory.md` and
`missing_figures_status.md`.

## Cross-references left open

The chapter refers to `Chapter~\ref{ch:introduction}` and `Chapter~\ref{ch:medus}`.
Those labels must match the `\label{}` commands used when Chapters 1 and 4 are
written, or the references will not resolve.

## Citation placeholders

| placeholder | study | bibliographic source |
|---|---|---|
| `[CITE: Kodge et al., year]` | Kodge, Saha & Roy, *Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting*, TMLR 07/2024 | `claudedocs/research_anchor_paper_20260827.md`; openreview.net/forum?id=BmI5p6wBi0; arXiv:2312.00761 |
| `[CITE: Tarun et al., year]` | Tarun, Chundawat, Mandal & Kankanhalli, *Fast Yet Effective Machine Unlearning* (UNSIR) | same file; arXiv:2111.08947 |
| `[CITE: Fan et al., year]` | Fan, Liu, Zhang, Wong, Wei & Liu, *SalUn*, ICLR 2024 | same file; arXiv:2310.12508 |

Replace the year placeholders with the correct years and confirm each reference
against the paper itself before submission.
