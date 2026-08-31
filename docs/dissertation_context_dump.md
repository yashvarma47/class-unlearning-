# MED-US — Dissertation Context Dump

**Repository:** `MEDUS_Class_Unlearning`
**Dissertation title:** *Multi-Objective Evolutionary Design of Unlearning Strategies (MED-US)*
**Degree:** MSc Computer Science with Artificial Intelligence, University of Nottingham
**Extracted:** 2026-08-31, at commit `207a874`, working tree clean.

> **Read this box before writing anything.** Four things a reader would reasonably
> assume about this project are **false**, and the dump states them plainly where
> they arise:
>
> 1. **pymoo is not used.** It is not installed, not in `requirements.txt`, and not
>    imported anywhere. NSGA-II is implemented from scratch in
>    `src/medus_class/search/nsga2.py`. See §5.
> 2. **No baseline unlearning method is implemented in this repository.** The
>    comparison against NegGrad / SCRUB / SSD / Kodge is against those papers'
>    *published* numbers, transcribed by hand. See §7.
> 3. **Only one architecture and one dataset exist.** ResNet-18 / CIFAR-10. There is
>    no VGG-19, no CIFAR-100. Some docstrings mention VGG19 as a future intent; no
>    such file exists. See §3.
> 4. **Everything is one seed (42).** The `±` in every table is class-to-class
>    spread over ten classes, **not** run-to-run variance. See §8.

---

## 1. PROJECT STRUCTURE

Version-controlled files only. Large binaries (`*.pt`, `*.zip`, the CIFAR-10
download) are git-ignored — see the note after the tree.

```
MEDUS_Class_Unlearning/
├── .gitattributes                      # Git LFS rules
├── .gitignore
├── README.md                           # 216 lines; full text quoted in §11
├── requirements.txt
├── requirements-torch.txt
│
├── claudedocs/
│   └── research_anchor_paper_20260827.md   # anchor-paper selection + Table 1
│
├── configs/
│   ├── base.yaml                       # seed, device, paths, logging
│   ├── data/cifar10_class.yaml         # CIFAR-10 class-unlearning data config
│   ├── model/resnet18.yaml             # architecture + layer groups + training
│   ├── operators/lookup.yaml           # THE operator table (only place ratios live)
│   └── search/
│       ├── base? (none — search configs are self-contained)
│       ├── kaggle_class_frog_main.yaml
│       ├── kaggle_class_frog_smoke.yaml
│       ├── plan_a_airplane.yaml        ├── plan_a_airplane_smoke.yaml
│       ├── plan_a_automobile.yaml      ├── plan_a_bird.yaml
│       ├── plan_a_cat.yaml             ├── plan_a_deer.yaml
│       ├── plan_a_dog.yaml             ├── plan_a_frog.yaml
│       ├── plan_a_frog_smoke.yaml      ├── plan_a_horse.yaml
│       ├── plan_a_ship.yaml            ├── plan_a_ship_smoke.yaml
│       └── plan_a_truck.yaml
│
├── docs/
│   ├── artifact_manifest.md            # provenance of every artefact, 22.9 KB
│   ├── dissertation_outline.html       # chapter plan
│   ├── old_repo_class_changes.patch
│   └── ship_plan_a_smoke_report.md
│
├── experiments/                        # 20 scripts, all CLI entry points
│   ├── analyse_class_structure.py      # D_f vs D_r activation contrast + null control
│   ├── analyse_truck_predictions.py    # inference-only prediction distribution
│   ├── build_class_structure_figure.py
│   ├── build_hybrid_summary.py
│   ├── build_ten_class_summary.py
│   ├── build_writeup_figures.py
│   ├── build_writeup_package.py
│   ├── check_class_baselines.py        # sanity check of W_0 / W_ref wiring
│   ├── evaluate_class_front.py         # full-fidelity re-measurement of a front
│   ├── measure_refined_anchor.py
│   ├── plot_pareto_front.py            # frog-specific, deliberately frozen
│   ├── plot_pareto_front_class.py      # class-agnostic plotter
│   ├── refine_candidate.py             # the BN-frozen hybrid step
│   ├── report_anchor_metrics.py        # ACC_r / ACC_f / composite / MIA
│   ├── report_final_objectives.py
│   ├── run_class_sweep.py              # drives the ten-class sweep
│   ├── run_plan_a.py                   # the search entry point
│   ├── save_cstar_checkpoint.py
│   ├── smoke_objectives.py
│   ├── summarise_references.py
│   ├── train_class_reference.py        # trains W_ref on D_r only
│   └── write_anchor_markdown.py
│
├── kaggle/                             # distributed reference training
│   ├── kaggle_run_plan_a.ipynb
│   ├── README_KAGGLE.md
│   └── reference_training/
│       ├── classes_aditya.yaml  classes_pragati.yaml  classes_yash.yaml
│       ├── trial_ship.yaml
│       ├── FRIEND_INSTRUCTIONS.md   README.md
│       ├── import_reference_zip.py  package_outputs.py
│       ├── train_references.py      train_references_kaggle.ipynb
│       └── validate_reference_zip.py
│
├── scripts/
│   └── package_for_kaggle.py
│
├── src/medus_class/
│   ├── __init__.py
│   ├── data/
│   │   ├── cifar10.py                  # loaders, transforms, normalisation
│   │   └── class_split.py              # D_f / D_r partition, save/load/validate
│   ├── evaluation/
│   │   ├── anchor.py                   # Kodge et al. ACC_r/ACC_f/composite/MIA
│   │   ├── evaluator.py                # ClassEvaluator — chromosome -> objectives
│   │   ├── metrics.py                  # normalise_objectives, accuracy/loss
│   │   ├── objectives.py               # f1, f2, f3, selectivity S
│   │   └── privacy.py                  # the project's own AUC-based MIA
│   ├── models/
│   │   ├── checkpoint.py               # save/load + CheckpointMetadata sidecars
│   │   ├── layer_groups.py             # LayerGroupRegistry, disjointness/coverage
│   │   └── resnet18.py                 # ResNetCIFAR (3x3 stem, no max-pool)
│   ├── operators/
│   │   ├── base.py                     # OperatorContext, group snapshots
│   │   ├── gradient_free.py            # the 8 operator implementations
│   │   ├── registry.py                 # lookup.yaml -> specs, MAX_LEVEL
│   │   └── selection.py                # class_contrast / magnitude / random
│   ├── search/
│   │   ├── checkpoint.py               # resumable search state + RNG
│   │   ├── decoder.py                  # chromosome -> DecodedStrategy
│   │   ├── genome.py                   # Chromosome, ChromosomeBounds
│   │   ├── nsga2.py                    # NSGA-II from scratch
│   │   └── population.py
│   └── utils/
│       ├── config.py                   # _base_ composition, deep_merge, paths
│       ├── device.py
│       └── seeding.py
│
├── tests/                              # 65 tests, all passing
│   ├── test_anchor_metrics.py
│   ├── test_class_split.py
│   ├── test_objectives.py
│   └── test_operators.py
│
└── results/
    ├── analysis/
    │   ├── class_structure/
    │   │   ├── channel_contrast_all_classes.csv   # 43,550 rows (10 classes x 4,355 ch)
    │   │   ├── per_class_groups.csv               # 60 rows (10 classes x 6 groups)
    │   │   └── summary.json
    │   ├── class_structure_console.out
    │   └── objective_smoke.json
    ├── checkpoints/                    # only 3 tracked (Git LFS); rest ignored
    │   ├── cifar10_resnet18_seed42_best.pt        + .json     (W_0)
    │   ├── class6_frog_reference_best_dr.pt       + .json     (frog W_ref)
    │   └── class6_frog_C_star_pure_gradient_free.pt + .json   (frog C*)
    ├── splits/                         # cifar10_class{0..9}_{name}.json — all 10
    ├── literature_alignment/
    │   ├── frog_anchor_metrics.{csv,json,md}
    │   ├── protocol_validation_report.md
    │   ├── pure_vs_hybrid_comparison.{csv,md}
    │   ├── sweep_progress.json
    │   ├── ten_class_hybrid_summary.{csv,md}  ten_class_hybrid_mean_std.csv
    │   └── ten_class_pure_summary.{csv,md}    ten_class_pure_mean_std.csv
    ├── reference_training/
    │   ├── all_reference_models_summary.md
    │   ├── reference_validation_summary.csv
    │   ├── class{0,1,2,3,4,5,7,8,9}_*_training_summary.md
    │   ├── class{0,1,2,3,4,5,7,8,9}_*_kaggle_manifest.json
    │   ├── {airplane,automobile,bird,cat,deer,dog,horse}_baseline_check.md
    │   └── ship_reference_validation.md
    ├── search/
    │   ├── plan_a_{airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck}/
    │   │   ├── pareto_front.csv
    │   │   ├── pareto_front_plot_data.csv     (frog: pareto_front_plot_table.csv)
    │   │   ├── pareto_front_plan_a_<class>.png
    │   │   ├── summary.json
    │   │   ├── <class>_anchor_metrics.{csv,json}
    │   │   └── full_fidelity/{front_full_fidelity.csv, baselines.json}
    │   └── plan_a_{...}_bn_frozen_refined/     (9 dirs: all but airplane)
    │       ├── refinement.json
    │       └── refined_best.json               (+ refined_best.pt, ignored except frog)
    └── writeup_package/                # the dissertation-ready package
        ├── benchmark_comparison_table.{csv,md}
        ├── class_structure_similarity.csv
        ├── figure_inventory.md
        ├── hybrid_medus_10_class_table.{csv,md}
        ├── key_numbers_summary.md
        ├── limitations_future_work_notes.md
        ├── missing_figures_status.md
        ├── pure_medus_10_class_table.{csv,md}
        ├── pure_vs_hybrid_summary_table.{csv,md}
        ├── reference_model_validation_table.{csv,md}
        ├── refinement_acceptance_table.{csv,md}
        ├── results_chapter_notes.md
        ├── truck_prediction_distribution.csv
        └── figures/
            ├── benchmark_comparison.png
            ├── class_structure_analysis.png
            ├── operator_frequency_selected_cstar.png
            ├── pure_vs_hybrid_acc_f_by_class.png
            ├── pure_vs_hybrid_acc_r_by_class.png
            ├── pure_vs_hybrid_composite_by_class.png
            └── truck_failure_analysis.png
```

**Present on disk but git-ignored** (state this in the reproducibility appendix):
`data/cifar10/` (the dataset), `results/checkpoints/class{0..9}_*_reference_best_dr.pt`
(ten × ~85 MB), the nine `refined_best.pt` (~43 MB each), `reference_outputs_*.zip`
(seven × ~83 MB), `results/*.out` run logs, and `evaluation_history.csv` in each run
directory. Only four checkpoints are in Git LFS: `W_0`, frog `W_ref`, frog `C*`, frog
refined. The storage decision for the remaining references is recorded as **open** in
`docs/artifact_manifest.md` §6.

---

## 2. DATASETS

### 2.1 Dataset

| item | value |
|---|---|
| Name | CIFAR-10 |
| Classes | 10 |
| Image size | 32 × 32 × 3 |
| Total train | 50,000 |
| Total test | 10,000 |
| Validation split | **none** — there is no separate validation set |
| Root | `data/cifar10` (torchvision download) |
| Config | `configs/data/cifar10_class.yaml` |

Normalisation: mean `[0.4914, 0.4822, 0.4465]`, std `[0.2470, 0.2435, 0.2616]`.
Train augmentation: random crop 32 with padding 4, random horizontal flip.
Batch sizes: `train: 64`, `eval: 128`, `debug: 32` (tuned for a 4 GB GTX 1650).

### 2.2 Forget / retain split — exact counts

The split is **class-level only**. `D_f` is one whole class; `D_r` is the other nine.
Identical counts for every class because CIFAR-10 is balanced:

| set | size | role |
|---|---:|---|
| `D_f_train` | 5,000 | the forget set (excluded from `W_ref` training) |
| `D_r_train` | 45,000 | the retain set (`W_ref` trains on exactly this) |
| `D_f_test` | 1,000 | held-out forget-class images — `ACC_f` is measured here |
| `D_r_test` | 9,000 | held-out retain images — `ACC_r` is measured here |

### 2.3 How splits were created

**There is no random seed for the split and no stratification, because there is no
sampling.** The label determines the partition entirely:
`forget_test = sort(flatnonzero(test_labels == forget_class))`,
`retain_test = setdiff1d(arange(n), forget_test)`. The split is therefore
deterministic and exactly reproducible from the class index alone.
Implementation: `src/medus_class/data/class_split.py::build_class_split`.

Splits are serialised as explicit index lists to
`results/splits/cifar10_class{ID}_{name}.json` — all ten are version-controlled.
Every split inside every imported Kaggle bundle was **byte-compared** against the
local version-controlled split before the checkpoint was accepted.

### 2.4 Classes unlearned

**All ten**, each in turn, as ten independent experiments. This was required by the
anchor paper, whose Table 1 reports a ten-class mean.

| index | label | index | label |
|---:|---|---:|---|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

Frog (6) is the config default and the first class run, chosen by measurement rather
than convention — see §10.3.

### 2.5 Verified fingerprints

sha256 (first 12 hex characters) of each retain-only reference checkpoint, with the
epoch selected and the validation result. All ten verdicts are `PASS`.

| id | class | `D_f_test` acc | `D_r_test` acc | epoch | sha256 (12) | source bundle |
|---:|---|---:|---:|---:|---|---|
| 0 | airplane | 0.0000 | 0.9516 | 135 | `283da38e314e` | `reference_outputs_yash_airplane.zip` |
| 1 | automobile | 0.0000 | 0.9476 | 191 | `5d07d03e6655` | `reference_outputs_yash_automobile.zip` |
| 2 | bird | 0.0000 | 0.9526 | 159 | `3750f7f565c1` | `reference_outputs_yash_bird.zip` |
| 3 | cat | 0.0000 | 0.9612 | 195 | `60ba8b8bf71a` | `reference_outputs_pragati_cat.zip` |
| 4 | deer | 0.0000 | 0.9457 | 189 | `cca3b78022c8` | `reference_outputs_pragati_deer.zip` |
| 5 | dog | 0.0000 | 0.9576 | 151 | `f2bb585dbc49` | `reference_outputs_pragati_dog.zip` |
| 6 | frog | 0.0000 | 0.9459 | 163 | `c44f3f99e8a3` | (local training, never packaged) |
| 7 | horse | 0.0000 | 0.9423 | 162 | `478d67d102fe` | `reference_outputs_aditya.zip` |
| 8 | ship | 0.0000 | 0.9502 | 181 | `852fc08e8cb0` | `reference_outputs_trial_ship.zip` |
| 9 | truck | 0.0000 | 0.9514 | 174 | `52c6e8d7c132` | `reference_outputs_aditya.zip` |

`D_r_test` accuracy across the ten references: **0.9506 ± 0.0057**
(min 0.9423 horse, max 0.9612 cat).

Full table with losses and split sizes: `results/writeup_package/reference_model_validation_table.csv`.

---

## 3. MODEL ARCHITECTURES

### 3.1 What exists

**One architecture only: ResNet-18, CIFAR variant.** Defined in
`src/medus_class/models/resnet18.py` (class `ResNetCIFAR`, block `BasicBlock`),
configured by `configs/model/resnet18.yaml`.

There is **no VGG-19 and no second architecture in the repository.** The docstrings in
`genome.py` and `layer_groups.py` say "currently 6 [layer groups] for both ResNet-18
and VGG19" — that is aspirational text carried over from the predecessor project. No
VGG file exists. Do not claim a second architecture.

### 3.2 Specification

| item | value |
|---|---|
| Input | 32 × 32 × 3 |
| Stem | single 3×3 conv, stride 1, **no max-pool** (the CIFAR variant) |
| Stages | `layer1`–`layer4`, two `BasicBlock`s each, widths 64/128/256/512 |
| Head | `fc`, 512 → 10 |
| Depth | 18 weighted layers |
| **Total parameters** | **11,173,962** (all trainable) |
| Buffer elements (BN running stats) | 9,620 |
| Checkpoint size on disk | ~89.5 MB (weights + optimiser + scheduler state) |

### 3.3 Modification from the standard architecture

torchvision's ResNet-18 uses a 7×7 stride-2 stem plus a 3×3 stride-2 max-pool,
downsampling 4× before the first residual stage — which discards almost all spatial
information on a 32×32 image. The CIFAR variant replaces the stem with a single 3×3
stride-1 convolution and removes the max-pool. This is the setup used by AMUN and by
the SalUn / Unlearn-Sparse line the operators come from. **Module names are kept
identical to torchvision's** (`conv1`, `bn1`, `layer1`…`layer4`, `fc`) because the
layer-group registry — and therefore the chromosome — addresses parameters by name.

### 3.4 Layer groups (this defines the chromosome length L = 6)

| i | group | modules | parameters | share of model | tensors |
|---:|---|---|---:|---:|---:|
| 0 | `stem` | `conv1`, `bn1` | 1,856 | 0.02% | 3 |
| 1 | `layer1` | `layer1` | 147,968 | 1.32% | 12 |
| 2 | `layer2` | `layer2` | 525,568 | 4.70% | 15 |
| 3 | `layer3` | `layer3` | 2,099,712 | 18.79% | 15 |
| 4 | `layer4` | `layer4` | 8,393,728 | **75.12%** | 15 |
| 5 | `fc` | `fc` | 5,130 | 0.05% | 2 |

The registry (`layer_groups.py`) enforces **disjointness** (no parameter in two
groups) and **coverage** (every trainable parameter belongs to some group). The 75/25
split between `layer4` and everything else is the direct reason `f3` is a *relative*
norm — see §6.3.

### 3.5 The original model `W_0`

One original model, shared by all ten class experiments.
`results/checkpoints/cifar10_resnet18_seed42_best.pt`:

| item | value |
|---|---|
| Trained on | the **full** CIFAR-10 training set (includes every `D_f`) |
| Epochs | 200 (best epoch 199) |
| Optimizer | SGD, lr 0.1, momentum 0.9, weight decay 5e-4 |
| Scheduler | step, ×0.1 every 40 epochs |
| Seed | 42 |
| Precision | fp32, AMP disabled |
| Train acc / loss | 0.99982 / 0.003027 |
| Test acc / loss | **0.9479** / 0.198381 |

### 3.6 The retain-only references `W_ref`

Ten models, one per class, each trained from scratch on `D_r_train` only (45,000
images) under the identical protocol: 200 epochs, seed 42, SGD as above. **Checkpoint
selection is on `D_r_test` accuracy with `D_r_test` loss as tie-breaker;** `D_f_test`
is logged every epoch but never influences selection. Wall clock ≈ 2.32 h per
reference on a Kaggle Tesla T4.

---

## 4. CHROMOSOME / ENCODING

### 4.1 Structure

A candidate unlearning strategy is

```
x = (b, g, s, d_g, d_s)
```

five **integer** vectors, each of length `L = 6` (the number of layer groups).
Flat genome length = **5 × 6 = 30 genes**. Defined in
`src/medus_class/search/genome.py` (`Chromosome`, `ChromosomeBounds`).

| gene | type | length | range | meaning |
|---|---|---:|---|---|
| `b` | int | 6 | `{0, 1}` | is layer group *i* active? |
| `g` | int | 6 | `0..2` | which **editor** operator to apply to group *i* |
| `s` | int | 6 | `0..4` | which **smoother** operator to apply to group *i* |
| `d_g` | int | 6 | `0..max_level` | ordinal intensity of the editor operator |
| `d_s` | int | 6 | `0..max_level` | ordinal intensity of the smoother operator |

`max_level` is **2 in every experiment actually run** (set in each search config),
though the library defines levels up to 5.

Intensity is an **ordinal level, not a continuous value**:
`0 = OFF, 1 = VERY_LOW, 2 = LOW, 3 = MEDIUM, 4 = HIGH, 5 = VERY_HIGH`.
Level 0 means "skip this operator" and never indexes a ladder; levels 1..5 map onto
ladder entries `levels[0..4]` in `configs/operators/lookup.yaml`.

Flat layout is **gene-major**: `[b | g | s | d_g | d_s]`. This keeps the `b` mask
contiguous and is the standard choice for integer-coded NSGA-II.

### 4.2 Search-space size

Per group: `2 × n_editor × (max_level+1) × n_smoother × (max_level+1)`.
With 3 editors, 5 smoothers, L = 6:

| `max_level` | genomes |
|---:|---:|
| **2** (as run) | **387,420,489,000,000** ≈ 3.87 × 10¹⁴ |
| 5 (library maximum) | 1,586,874,322,944,000,000 ≈ 1.59 × 10¹⁸ |

This counts *genomes*, not distinct decoded strategies — latent genes (below) make
the number of distinct strategies smaller. Set against ~210–320 real evaluations per
class, this is the justification for a search rather than enumeration.

### 4.3 Decoding rule, in plain English

For each layer group *i*, in **forward order** (`stem → layer1 → … → fc`):

1. If `b_i = 0`, the group is **frozen** and nothing happens to it. Its other four
   genes are ignored.
2. If `b_i = 1`:
   a. If `d_g,i > 0`, apply editor operator `g_i` to group *i* at intensity level
      `d_g,i`, using the hyperparameters from that operator's ladder.
   b. If `d_s,i > 0`, apply smoother operator `s_i` to group *i* at intensity level
      `d_s,i`.
3. A group with `b_i = 1` but both intensities at 0 contributes nothing.
4. A chromosome where no group contributes anything is a **no-op** — legal, and it
   means "leave this model alone". The identity chromosome is forced in as
   individual 0 of the initial population as a plumbing check: its objectives are
   known in advance, so if the search reports anything else the harness is broken.

**Latent genes.** When `b_i = 0` the other four genes are *kept* in the genome as
latent genetic material — a later mutation flipping `b_i` on recovers a strategy the
population already found. The cost is that distinct genomes can decode to the same
strategy, which is why the decoder also emits a canonical form for the objective
cache, and why identical individuals share a Pareto front.

**Execution order** within an active group is editor then smoother; across groups it
is forward order. `RESET` depends on the snapshot taken before a group is modified,
so it is only meaningful after another operator has run on the same group.

### 4.4 Operator library — all eight

Defined in `configs/operators/lookup.yaml`, implemented in
`src/medus_class/operators/gradient_free.py`. All are **gradient-free**: no loss is
formed, no `backward()` is called, and `torch.no_grad()` wraps the measurement.

Ratios below are the ladder for levels 1–5. **Only levels 1 and 2 were reachable in
the experiments** (`max_level: 2`), so the two gentlest rungs of each ladder are the
ones that actually ran.

#### Channel A — `editor` operators (choose *which* connections, then switch them off)

| id | name | what it does | selection rule | level 1 | 2 | 3 | 4 | 5 |
|---:|---|---|---|---|---|---|---|---|
| 0 | **MASK** | zero the top-ratio connections by `\|W\| · (rms_f − rms_r)` | `class_contrast` (forget-informed) | ratio 0.01 | 0.05 | 0.10 | 0.20 | 0.40 |
| 1 | **PRUNE** | zero the bottom-ratio connections by `\|W\|` | `magnitude` — **data-free control** | ratio 0.50 | 0.70 | 0.85 | 0.95 | 0.99 |
| 2 | **RANDOM_PRUNE** | zero a uniformly random ratio of connections | `random` — **null model for selection** | ratio 0.05 | 0.15 | 0.30 | 0.50 | 0.70 |

#### Channel B — `smoother` operators (keep the connections, change what they hold)

| id | name | what it does | level 1 | 2 | 3 | 4 | 5 |
|---:|---|---|---|---|---|---|---|
| 0 | **DAMP** | multiply selected connections by `(1 − strength)` | r 0.05, s 0.10 | 0.10 / 0.25 | 0.20 / 0.50 | 0.30 / 0.75 | 0.40 / 0.90 |
| 1 | **NOISE** | add `N(0, σ·std(W))` to selected connections | r 0.10, σ 0.5 | 0.20 / 1.0 | 0.30 / 1.5 | 0.40 / 2.0 | 0.50 / 3.0 |
| 2 | **CLIP** | clamp selected connections to `± limit · std(W)` | r 0.10, lim 3.0 | 0.20 / 2.0 | 0.30 / 1.0 | 0.40 / 0.5 | 0.50 / 0.2 |
| 3 | **QUANTIZE** | round selected connections onto a `2^bits` uniform grid | r 0.30, 5 bits | 0.40 / 4 | 0.50 / 4 | 0.60 / 3 | 0.70 / 3 |
| 4 | **RESET** | copy selected connections back from the group snapshot | ratio 0.05 | 0.10 | 0.20 | 0.35 | 0.50 |

**NOISE is the only operator that never appears in any selected `C*`.**

#### Two operators deliberately absent from the library

`REINIT` (re-draw weights from the init distribution) and `SIGN_FLIP` (negate
weights) are **excluded at library level, not merely disabled by config**, so no
config edit can bring them back. They were the most destructive operators in the
predecessor's calibration — SIGN_FLIP took `layer4` from 0.988 to 0.211 forget
accuracy across its ladder — and dominated fronts that turned out to be full of
wrecked models.

### 4.5 Selection rules (which connections an operator touches)

`src/medus_class/operators/selection.py`. Three rules; **`class_contrast` is the
production rule** and is set in every search config.

| rule | criterion | role |
|---|---|---|
| `class_contrast` | highest `\|W_ij\| · (rms_f_j − rms_r_j)` | forget-informed; **the production rule** |
| `magnitude` | largest `\|W_ij\|` | data-free ablation; `PRUNE` pins itself here |
| `random` | uniform | the null model; `RANDOM_PRUNE` pins itself here |

The importance criterion is the activation-aware one used by Wanda,
`importance(W_ij) = |W_ij| · ||a_j||₂`, computed **from forward passes only**,
separately over `D_f` and `D_r`, and then subtracted. Activation norms are
root-mean-square, not raw sums, so `D_f` and `D_r` contribute on the same scale
despite holding very different numbers of samples.

Rationale, quoted from the module docstring: the predecessor's 50×100 search measured
selectivity at "1.002 median, 1.047 best" — damage was exactly indiscriminate. An
operator choosing targets from the weights alone has no mechanism to distinguish
`D_f` from `D_r`, so `S ≈ 1` is structural rather than observed. The supervisor
confirmed the requirement directly: *"the connections to change must be chosen using
the forget set `D_f`; otherwise, the operators will damage the forgotten and retained
data equally."*

### 4.6 Taxonomy

Two channels: **editor** (connection-selection edits — choose which connections to
touch and switch them off or overwrite them) and **smoother** (value-perturbation
edits — keep the connections and change what they hold). Every operator is marked
`atomic: true` and `role: operator` in the lookup table; composite entries such as
`RL` and `L1_SPARSE` are *methods*, not atomic operators, and are excluded from the
chromosome by the `selectable` flag.

---

## 5. NSGA-II CONFIGURATION

### 5.1 pymoo is NOT used

**There is no pymoo dependency.** It is not in `requirements.txt`, it is not
installed in the environment (`ModuleNotFoundError: No module named 'pymoo'`), and no
module imports it. NSGA-II is **implemented from scratch** in
`src/medus_class/search/nsga2.py`, following Deb et al. (2002).

The docstring states the reason: the genome is not a real vector — it is five integer
gene blocks with different bounds, latent genes behind a `b` mask, and an expensive
cacheable objective function. The standard SBX / polynomial-mutation operators assume
continuous variables and do not apply.

**Class and function names in this repository (use these in the dissertation):**

| symbol | file | role |
|---|---|---|
| `NSGA2` | `search/nsga2.py` | the algorithm; takes an injected `evaluate` callback |
| `NSGA2Config` | `search/nsga2.py` | the hyperparameter dataclass |
| `NSGA2Result` | `search/nsga2.py` | final population, objectives, rank, distance, history |
| `fast_non_dominated_sort` | `search/nsga2.py` | Deb's O(MN²) front-peeling |
| `crowding_distance` | `search/nsga2.py` | density estimate, boundaries `inf` |
| `crowded_comparison` | `search/nsga2.py` | rank first, then larger crowding distance |
| `binary_tournament` | `search/nsga2.py` | parent selection |
| `uniform_crossover` | `search/nsga2.py` | variation |
| `random_reset_mutation` | `search/nsga2.py` | variation |
| `dominates` | `search/nsga2.py` | minimisation-sense Pareto domination |
| `Chromosome`, `ChromosomeBounds` | `search/genome.py` | the genome |
| `GenerationRecord` | `search/nsga2.py` | per-generation history row |

### 5.2 Hyperparameters — identical in all ten class configs

| setting | value | source |
|---|---|---|
| Algorithm | `nsga2` | `search.algorithm` |
| **Population size** | **10** | `search.population_size` |
| **Generations** | **50** | `search.generations` |
| **Crossover** | **uniform**, per-gene swap probability 0.5 | `uniform_crossover` |
| **Crossover probability** | **0.9** | `search.crossover_probability` |
| **Mutation** | **random reset**, per-gene, drawn uniformly from that gene's own bounds | `random_reset_mutation` |
| **Mutation probability** | **0.10** per gene | `search.mutation_probability` |
| Default if unset | `1 / n_genes` = 1/30 ≈ 0.0333 | `NSGA2.__init__` |
| Selection | binary tournament on the crowded comparison | `binary_tournament` |
| Survival | elitist (μ + λ) truncation, fronts admitted whole, first overflowing front sorted by crowding distance | `select_survivors` |
| `p_active` | 0.5 (probability a group starts active) | `search.p_active` |
| `normalise_objectives` | `true` — per-generation min-max, **for selection only** | `search.normalise_objectives` |
| `cache_objectives` | `true` | `search.cache_objectives` |
| `checkpoint_every_generation` | `true` | resumable search |
| **Seed** | **42** | `search.seed` |
| Total evaluation budget | 10 + 50 × 10 = **510 chromosome evaluations** per class | |

### 5.3 Constraints and repair

There is **no constraint-handling mechanism and no repair operator** in the NSGA-II
sense. Validity is enforced structurally instead:

- `random_reset_mutation` draws from **each gene's own bounds**, so it cannot produce
  an invalid genome. The docstring notes that a single shared range plus clipping
  afterwards would bias the distribution towards the bounds.
- `uniform_crossover` swaps values between two already-valid parents, so children are
  valid by construction.
- `Chromosome.validate()` raises on any out-of-range gene; `Chromosome.clip()` exists
  as a safety net.
- Failed evaluations are recorded as `failures` in the generation record. **Across all
  ten runs, `failures = 0`.**

### 5.4 Two implementation details worth citing

**Crowding distance guards a zero-range objective.** An objective with zero range
across a front contributes nothing rather than dividing by zero. The docstring names
the exact case: `obj2` is 0 for every chromosome that has not yet damaged utility, so
without the guard a whole early generation would come out `nan` and selection would
collapse.

**Normalisation cannot change the fronts.** Min-max is a strictly increasing
per-objective transform and Pareto dominance is invariant under such transforms. Its
purpose is to put objectives of very different scale on a common footing for crowding
distance (`f2` is an unbounded loss, `f1` is capped at ln 2). **Raw values are what
reach the result files**; normalised values would not be comparable across
generations because the range is recomputed each time.

### 5.5 Two-tier fidelity (important for the methodology chapter)

During the search, objectives are computed on **subsets**: `forget_subset_size: 64`,
`retain_subset_size: 256`, `batch_cap: 3`. After the search, the ten members of the
final front are **re-measured at full fidelity** on the complete sets by
`evaluate_class_front.py`. This is why a class costs ~8 minutes of search plus ~10
minutes of re-measurement, rather than hours. Every headline number comes from the
full-fidelity pass, never from the search-time subsets.

---

## 6. OBJECTIVE FUNCTIONS

All three are **minimised**. Domination is therefore "no worse in every objective and
strictly better in at least one" — implemented directly in `dominates()`:

```python
no_worse = all(x <= y for x, y in zip(a, b))
strictly_better = any(x < y for x, y in zip(a, b))
return no_worse and strictly_better
```

There is **no sign-flipping or maximisation wrapper**, because there is no pymoo. The
objective tuple order is fixed by `OBJECTIVE_NAMES = ("obj1_js", "obj2_retain_loss",
"obj3_edit_cost")`. All three are defined in
`src/medus_class/evaluation/objectives.py`.

### 6.1 `f1` — Jensen–Shannon divergence to the reference on `D_f`

```
f1 = JS( P_ref(D_f) || P_cand(D_f) )
   = ½·KL(P ‖ M) + ½·KL(Q ‖ M),   where M = ½(P + Q)
```

- Function: `js_to_reference(model, loader, cached_reference_logits, device)`
- Column name: `obj1_js`
- Variables: `P` = softmax of the cached `W_ref` logits, `Q` = softmax of the
  candidate's logits, both over `D_f` (the `forget_eval` loader).
- Units: nats. **Bounded above by `ln 2` ≈ 0.6931** (`JS_MAX_NATS`), attained only for
  disjoint support.
- Measures: how far the candidate's *predictive distribution* on the forget class is
  from that of a model which never saw the class.
- Implementation note worth citing: computed from log-softmax with `logaddexp` rather
  than by adding probabilities and taking a log, which loses precision in the tail
  where the objective is most sensitive. The reference logits are cached once at
  evaluator construction, which requires the `forget_eval` loader to be deterministic
  and unshuffled — it is — or the cached rows would not line up.

Three stated reasons for JS over the KL the predecessor used: **symmetric** (we want
agreement, not one-sided coverage), **bounded** (one destroyed candidate cannot
flatten every other individual's `f1` under min-max normalisation), **finite
everywhere** (forward KL diverges as the candidate's probability on a
reference-supported class approaches zero; the mixture `M` always covers both).

### 6.2 `f2` — retain loss

```
f2 = L_r
```

- Column name: `obj2_retain_loss`
- Cross-entropy of the candidate over `D_r_train` (subset-sampled during search, full
  set at full fidelity).
- Measures: utility damage to the nine retained classes.
- Unbounded above.

### 6.3 `f3` — relative edit cost

```
f3 = ‖θ − θ₀‖₂ / ‖θ₀‖₂
```

- Function: `relative_parameter_delta(model, original_state, weights_only=True)`
- Column name: `obj3_edit_cost`
- Variables: `θ` = candidate parameters, `θ₀` = `W_0` parameters. Accumulated in
  **float64**. Exactly 0 for the identity edit.
- `weights_only=True` restricts the sum to `*.weight` tensors: **BatchNorm running
  statistics are buffers, not edits, and counting them would charge a strategy for the
  model merely having seen data.**
- Measures: how much of the network the edit moved, relative to the network's own
  scale.

Why *relative*: absolute movement is dominated by whichever group holds the most
parameters — `layer4` carries 75.12% of ResNet-18 — so an absolute cost would price a
large edit to `fc` as nearly free and a small edit to `layer4` as expensive, which is
a statement about the architecture rather than about the strategy.

### 6.4 Why `f3` is an edit cost and not a second reference term

The predecessor used `f2 = L_r` alongside `f3 = KL(P_ref(D_f) ‖ P(D_f))`, and the two
behaved as near-duplicates — both punish a damaged model — so a nominally
three-objective search was really a two-objective one wearing three labels. Measured
Spearman rank correlation against `f2` on random candidates:

| candidate `f3` | Spearman vs `f2` |
|---|---:|
| **parameter-change norm** (used here) | **+0.36** |
| KL to reference (the old `f3`) | +0.74 |

Edit cost never reads the data, so it is orthogonal by construction.

### 6.5 Diagnostics recorded beside the objectives (never optimised)

| quantity | definition | file |
|---|---|---|
| `selectivity S` | `(forget loss gained) / (retain loss paid)`, both relative to `W_0`. `inf` if the candidate raises forget loss while paying nothing measurable; `nan` for an identity edit (0/0). | `objectives.py::selectivity` |
| `kl_to_reference` | forward `KL(P_ref ‖ P)` on `D_f` — retained only so the duplication claim above can be re-measured from any run's own output | `objectives.py` |
| `mia_auc` | the project's own MIA: members are forget-class **train** images, non-members forget-class **test** images; AUC of a max-softmax attack | `evaluation/privacy.py` |
| anchor `ACC_r`, `ACC_f`, `composite`, `MIA` | the Kodge et al. protocol, §7.2 | `evaluation/anchor.py` |

**`S = nan` for an identity edit caused a real defect.** `--best-s` selected the
highest-`S` front member with a plain `max()`; every comparison against `nan` is
`False`, so `max()` returned the first `nan` row and reported the *unedited* model as
most selective. Three classes (deer, dog, horse) were written that way before a
finite-value guard was added. **No headline number was ever affected** — `C*` is
selected on the composite, which is never `nan` — only the diagnostic `best_S` row.
Worth a paragraph in the methodology chapter.

---

## 7. BASELINES / PEER ALGORITHMS

### 7.1 Implemented in this repository

**None.** No published unlearning baseline (NegGrad, NegGrad+, UNSIR, SCRUB, SSD,
Kodge, finetune-on-`D_r`, random-relabel, Fisher, …) is implemented or re-run here.
State this explicitly in the dissertation; it is the single largest methodological
limitation.

What *is* measured in this harness, and can legitimately be called a baseline:

| baseline | what it is | where |
|---|---|---|
| `W_0` — Original | the un-edited model | `results/checkpoints/cifar10_resnet18_seed42_best.pt`, measured by `report_anchor_metrics.py` |
| `W_ref` — Retraining (gold standard) | trained from scratch on `D_r_train` only, one per class | `experiments/train_class_reference.py`, `kaggle/reference_training/train_references.py` |
| `PRUNE` (magnitude selection) | data-free operator control **inside** the search | `operators/gradient_free.py`, `lookup.yaml` id 1 |
| `RANDOM_PRUNE` (random selection) | the null model for selection, **inside** the search | `lookup.yaml` id 2 |
| `magnitude` / `random` selection rules | data-free ablation of the selector | `operators/selection.py` |

Note the distinction: `PRUNE` and `RANDOM_PRUNE` are *operator-level* controls
competing inside the same search, not independent unlearning *methods*.

### 7.2 The comparison that does exist — against published numbers

`src/medus_class/evaluation/anchor.py` reimplements the **measurement protocol** of

> Sangamesh Kodge, Gobinda Saha, Kaushik Roy. *"Deep Unlearning: Fast and Efficient
> Gradient-free Class Forgetting."* Transactions on Machine Learning Research, 07/2024.
> <https://openreview.net/forum?id=BmI5p6wBi0> · arXiv:2312.00761 ·
> code: <https://github.com/sangamesh-kodge/class_forgetting>

Every formula was read from their released source, not inferred from the prose:

- **`ACC_r` / `ACC_f`** — plain micro-accuracy over each group of samples, from
  `utils.py::test`. Reported in percent.
- **composite** — `utils.py::metric_function(x, y) = x * (1 - y)` with `x = retain_acc`
  and `y = forget_acc` as fractions, logged as `100 × metric`.
  **The MIA is *not* a term in it.** An earlier draft of the literature review
  recorded it as `ACC_r × (100 − ACC_f) × MIA`; that was wrong and is corrected in the
  module docstring.
- **`MIA`** — `utils.py::SVC_MIA`. Three details a paraphrase loses:
  (i) the feature is the probability assigned to the sample's **ground-truth label**
  (`torch.gather(prob, 1, target)`), not the max softmax the project's own MIA uses;
  (ii) the attacker is **fit** on retain-train (members) against forget-train
  (non-members) and then **scored** on a third, disjoint set, `D_f_test`;
  (iii) the reported number is `1 − mean(predict(D_f_test))`, the fraction of
  forget-class *test* images the attacker calls non-members. Higher is better.

The eight literature rows in the benchmark table are **transcribed by hand** from that
paper's Table 1 and are labelled "reported" in every rendering. See §8.4.

---

## 8. EXPERIMENTAL RESULTS

### 8.0 What was run

| | |
|---|---|
| Dataset / model | CIFAR-10 / ResNet-18 (CIFAR variant), for every experiment |
| Forget set | one whole class, ten separate experiments (classes 0–9) |
| Seeds | **one — 42** for `W_0`, all ten `W_ref`, and every NSGA-II run |
| Searches | 10 (one per class), 50 generations × population 10 |
| Failures | 0 across all ten runs |
| Ablations | **none run** — no random-search comparison, no operator ablation, no budget sweep, no selector ablation |

**`C*` selection rule** — identical for all ten classes:
`C* = the front member maximising the anchor composite, ACC_r × (1 − ACC_f)`.
This is the anchor paper's own metric function, not one invented here. It reproduces
the three classes selected by hand before the sweep existed (frog #8, ship #6,
airplane #0), so no previously reported number changed.

> **Methodological caveat to state in the dissertation.** `C*` is selected on
> `retain_test_acc` and `forget_test_acc` — the same test-set quantities that are then
> reported (`run_class_sweep.py:199`, `composite()` at line 123). It is a best-of-ten
> on the reported metric. No held-out selection split was used. The test sets are
> large enough to be halved (1,000 / 9,000), so the fix is cheap, but it was not done.

### 8.1 Pure MED-US — per class

Every row is pure gradient-free weight surgery; no gradient step was applied.
Source: `results/writeup_package/pure_medus_10_class_table.csv`.

| id | class | `C*` | operators | ACC_r ↑ | ACC_f ↓ | composite ↑ | MIA ↑ | S | `f1` | `f2` | `f3` | search min |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | #0 | `CLIP\|DAMP\|MASK` | 92.86 | **0.00** | 92.86 | 100.00 | 221.25 | 0.3892 | 0.0392 | 0.1795 | 15.5 |
| 1 | automobile | #5 | `CLIP\|MASK\|PRUNE` | 92.68 | 12.70 | 80.91 | 91.70 | 154.57 | 0.3647 | 0.0271 | 0.1210 | 11.9 |
| 2 | bird | #0 | `CLIP\|MASK\|RANDOM_PRUNE` | 95.12 | 17.90 | 78.10 | 91.70 | 1320.40 | 0.2894 | 0.0042 | 0.1819 | 8.3 |
| 3 | cat | #0 | `DAMP\|MASK\|PRUNE\|RANDOM_PRUNE\|RESET` | 94.80 | 9.60 | 85.70 | 95.10 | 168.69 | 0.4477 | 0.0207 | 0.1350 | 10.0 |
| 4 | deer | #2 | `MASK` | 94.20 | 7.80 | 86.85 | 96.00 | 1257.14 | 0.3623 | 0.0047 | 0.1176 | 6.4 |
| 5 | dog | #3 | `MASK` | **95.62** | 3.30 | 92.47 | 99.00 | **4427.91** | 0.1403 | 0.0023 | 0.1126 | 7.0 |
| 6 | frog | #8 | `DAMP\|MASK` | 92.52 | 8.30 | 84.84 | 94.20 | 281.76 | 0.3637 | 0.0189 | 0.1451 | 8.2 |
| 7 | horse | #7 | `MASK` | 93.14 | 9.80 | 84.02 | 91.10 | 176.56 | 0.4889 | 0.0302 | 0.1220 | 7.3 |
| 8 | ship | #6 | `MASK\|RESET` | 94.58 | 14.00 | 81.34 | 97.00 | 2436.50 | 0.2090 | 0.0027 | 0.1128 | 7.8 |
| 9 | truck | #0 | `CLIP\|MASK\|QUANTIZE` | 92.91 | **42.10** | **53.80** | **69.60** | **93.99** | 0.4205 | 0.0257 | 0.1166 | 8.2 |

**Aggregate (mean ± sample std, ddof = 1, over the ten classes — *not* run-to-run variance):**

| metric | mean ± std | min | max |
|---|---|---|---|
| `ACC_r` (%) | **93.84 ± 1.15** | 92.52 (frog) | 95.62 (dog) |
| `ACC_f` (%) | **12.55 ± 11.57** | 0.00 (airplane) | 42.10 (truck) |
| composite (%) | **82.09 ± 11.00** | 53.80 (truck) | 92.86 (airplane) |
| anchor MIA (%) | **92.54 ± 8.62** | 69.60 (truck) | 100.00 (airplane) |
| selectivity `S` | **1053.88 ± 1413.47** | 93.99 (truck) | 4427.91 (dog) |

### 8.2 The Pareto-optimal points selected — exact chromosomes

Flat genome `[b(6) | g(6) | s(6) | d_g(6) | d_s(6)]`, 30 integers, gene-major.

| class | `C*` | chromosome |
|---|---:|---|
| airplane | #0 | `1 0 0 0 1 1  1 0 1 0 0 0  0 2 1 0 0 2  0 0 0 2 2 0  2 1 2 0 1 2` |
| automobile | #5 | `1 0 0 0 1 0  1 1 0 0 0 0  2 3 3 2 1 1  1 0 0 1 1 2  1 0 0 2 0 1` |
| bird | #0 | `1 0 1 1 1 0  1 0 1 2 0 2  2 3 3 2 1 3  0 1 0 1 1 0  1 1 0 0 0 1` |
| cat | #0 | `1 0 0 0 1 1  1 2 0 2 0 2  4 3 0 3 0 4  2 0 2 2 1 1  1 0 0 0 2 1` |
| deer | #2 | `0 0 1 0 1 0  0 2 0 0 0 0  1 2 4 0 0 0  0 0 0 2 1 2  0 2 0 0 0 0` |
| dog | #3 | `0 0 0 0 1 0  0 0 0 0 0 1  2 3 0 1 1 3  1 1 0 0 1 0  1 0 1 0 0 1` |
| frog | #8 | `0 0 0 1 1 0  0 2 0 0 0 1  2 0 2 0 0 3  0 1 2 0 1 0  0 1 0 2 0 0` |
| horse | #7 | `0 0 0 0 1 1  1 2 2 0 0 0  2 2 1 2 3 4  2 1 0 0 1 0  1 2 0 0 0 0` |
| ship | #6 | `1 0 0 1 1 0  1 0 1 2 0 2  4 3 3 0 0 3  0 2 0 0 1 0  2 2 2 0 0 2` |
| truck | #0 | `1 0 1 1 1 0  1 0 1 2 0 2  2 3 3 3 1 3  0 1 0 0 1 0  1 1 1 0 0 1` |

**`MASK` appears in the selected `C*` for all ten classes — the only operator that
does.** Operator frequency across the ten: `MASK` 10, `CLIP` 4, `DAMP` 3, `PRUNE` 2,
`RANDOM_PRUNE` 2, `RESET` 2, `QUANTIZE` 1, `NOISE` 0. Three classes select `MASK`
alone (deer, dog, horse); five select it alone or with a single partner. This is a
*search outcome*, not a design choice — all operators were available at equal cost in
all ten runs — but with no operator ablation it remains convergent evidence rather
than a demonstrated property.

### 8.3 Hybrid variant — pure `C*` + one BN-frozen refinement step

**This is a different method and is reported separately, permanently.** The anchor's
method is gradient-free, so only the pure table is a like-for-like comparison with its
Table 1. Applied *outside* the evolutionary search:
one clipped gradient-**ascent** step on `D_f`, then one repair (descent) step on
`D_r`, with BatchNorm frozen.

Hyperparameters, identical for all nine attempts (`refinement.json`):

| | |
|---|---|
| forget step | SGD gradient ASCENT on cross-entropy over `D_f` |
| retain step | SGD gradient DESCENT on cross-entropy over `D_r` |
| forget lr / retain lr | 1e-4 / 1e-4 |
| batches per step | 8 |
| steps | one optimiser step per stage, on the mean gradient |
| BatchNorm | **FROZEN** — model held in `eval()` for both steps; `running_mean` / `running_var` / `num_batches_tracked` cannot update |
| max buffer movement | 1e-9 |
| movement budget (relative) | 0.02 (acceptance check threshold 0.0400) |
| max `D_r_test` drop | 0.01 |
| max loss ratio | 1.25 |
| max edit cost | 0.3 |
| seed | 42 |

**Nine eligible classes, nine accepted, zero rejected.** Airplane is a deliberate
no-op: pure `ACC_f` is already 0.00 at MIA 100.00.

| id | class | ACC_r | ACC_f | composite | MIA | S | param movement | BN movement | BN counters | `D_r_test` drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | 92.86 | 0.00 | 92.86 | 100.00 | 221.25 | — (no-op) | — | — | — |
| 1 | automobile | 92.60 | 5.50 | 87.51 | 95.80 | 178.42 | 0.000368 | 0.000000 | 0 | 0.000778 |
| 2 | bird | 94.90 | 12.50 | 83.04 | 93.70 | 1305.45 | 0.000343 | 0.000000 | 0 | 0.002222 |
| 3 | cat | 94.56 | 6.00 | 88.88 | 96.70 | 172.64 | 0.000303 | 0.000000 | 0 | 0.002444 |
| 4 | deer | 94.09 | 4.30 | 90.04 | 97.50 | 1262.49 | 0.000373 | 0.000000 | 0 | 0.001111 |
| 5 | dog | 95.57 | 1.90 | 93.75 | 99.50 | 4950.81 | 0.000357 | 0.000000 | 0 | 0.000556 |
| 6 | frog | 92.56 | 2.70 | 90.06 | 96.30 | 297.25 | 0.000420 | 0.000000 | 0 | −0.000333 |
| 7 | horse | 92.93 | 7.00 | 86.43 | 92.80 | 185.79 | 0.000394 | 0.000000 | 0 | 0.002111 |
| 8 | ship | 94.49 | 4.70 | 90.05 | 99.10 | 2832.90 | 0.000409 | 0.000000 | 0 | 0.000889 |
| 9 | truck | 92.68 | 30.90 | 64.04 | 79.10 | 107.42 | 0.000307 | 0.000000 | 0 | 0.002333 |

**Hybrid aggregate:** `ACC_r` 93.72 ± 1.12 · `ACC_f` **7.55 ± 8.87** ·
composite **86.66 ± 8.52** · MIA **95.05 ± 6.08** · `S` 1151.44 ± 1592.37.

**Six acceptance checks**, all six passed by all nine: (1) forget improved on
`D_f_test`; (2) `D_r_test` drop ≤ 0.010; (3) no utility collapse (retain losses ≤
1.25×); (4) edit cost ≤ 0.3; (5) parameter movement ≤ 0.0400; (6) **BatchNorm buffers
unchanged**.

> **Check 6 exists because of a real silent failure.** An earlier, unfrozen attempt on
> frog passed every weight-based guard and was reported as a success. It was not one:
> the model was left in training mode, and eight batches of `D_r` re-estimated the
> BatchNorm running statistics, undoing the operator edit while parameter movement,
> edit cost and retain accuracy all looked correct. Buffer movement is the only check
> that catches it, and it reads exactly 0.000000 on all nine accepted refinements.
> This belongs in the methodology chapter.

**Pure → hybrid, per class:**

| class | ΔACC_f | Δcomposite | ΔMIA | ΔACC_r |
|---|---:|---:|---:|---:|
| airplane | +0.00 | +0.00 | +0.00 | +0.0000 |
| automobile | −7.20 | +6.60 | +4.10 | −0.0778 |
| bird | −5.40 | +4.94 | +2.00 | −0.2222 |
| cat | −3.60 | +3.18 | +1.60 | −0.2444 |
| deer | −3.50 | +3.19 | +1.50 | −0.1111 |
| dog | −1.40 | +1.28 | +0.50 | −0.0556 |
| frog | −5.60 | +5.21 | +2.10 | **+0.0333** |
| horse | −2.80 | +2.41 | +1.70 | −0.2111 |
| ship | −9.30 | +8.71 | +2.10 | −0.0889 |
| truck | **−11.20** | **+10.24** | **+9.50** | −0.2333 |

| metric | pure | hybrid | change |
|---|---|---|---:|
| `ACC_r` (%) | 93.84 ± 1.15 | 93.72 ± 1.12 | **−0.12** |
| `ACC_f` (%) | 12.55 ± 11.57 | 7.55 ± 8.87 | **−5.00** |
| composite (%) | 82.09 ± 11.00 | 86.66 ± 8.52 | **+4.58** |
| anchor MIA (%) | 92.54 ± 8.62 | 95.05 ± 6.08 | **+2.51** |

Nine of ten improved on the composite; the tenth is the airplane no-op. No class
regressed on any headline metric. The standard deviation narrows on every metric —
the refinement helps the weak classes most.

### 8.4 Benchmark comparison

Rows 1–8 are **as reported** in Kodge et al. Table 1 (CIFAR-10 / ResNet-18, mean ± std
over all ten target classes) and were **not** re-measured in this harness. Rows 9–12
are measured here.

| method | numbers | grad-free | ACC_r ↑ | ACC_f ↓ | MIA ↑ |
|---|---|:---:|---|---|---|
| Original | reported | n/a | 94.89 ± 0.31 | 94.89 ± 2.75 | 0.03 ± 0.03 |
| Retraining (gold standard) | reported | n/a | 94.81 ± 0.52 | 0.00 | 100.00 ± 0.00 |
| NegGrad | reported | no | 69.89 ± 10.23 | 0.02 ± 0.04 | 0.00 |
| NegGrad+ | reported | no | 89.91 ± 1.41 | 0.94 ± 1.87 | 98.68 ± 1.42 |
| UNSIR (Tarun et al. 2023) | reported | no | 92.20 ± 0.72 | 10.89 ± 8.79 | 61.50 ± 25.86 |
| SCRUB (Kurmanji et al. 2023) | reported | no | 94.79 ± 0.63 | 0.00 | 0.00 |
| SSD (Foster et al. 2024) | reported | yes | 85.76 ± 25.76 | 4.37 ± 12.79 | 87.86 ± 31.21 |
| Kodge et al. 2024 (anchor) | reported | yes | 94.19 ± 0.50 | 0.03 ± 0.09 | 95.50 ± 14.23 |
| **Original `W_0`** | **measured** | n/a | **94.79 ± 0.29** | **94.79 ± 2.59** | **0.00 ± 0.00** |
| **Retraining `W_ref`** | **measured** | n/a | **95.06 ± 0.57** | **0.00 ± 0.00** | **100.00 ± 0.00** |
| **MED-US pure** | **measured** | yes | **93.84 ± 1.15** | **12.55 ± 11.57** | **92.54 ± 8.62** |
| **MED-US hybrid** | **measured** | no | **93.72 ± 1.12** | **7.55 ± 8.87** | **95.05 ± 6.08** |

**Gap to the anchor:** `ACC_r` −0.35, `ACC_f` **+12.52**, MIA −2.96.

**Evidence the two harnesses agree on shared baselines:** this project's own `W_0` and
`W_ref` land within **0.10** and **0.25** points of `ACC_r` of the paper's Original and
Retraining rows, with `ACC_f` and MIA agreeing to within 0.10 and 0.03. Indirect, and
it is the strongest such evidence available.

**A scepticism worth a paragraph:** in the same table SCRUB reaches `ACC_f` 0.00 with
MIA 0.00, and Retraining is pinned at exactly 100.00. A metric where the gold standard
sits at the ceiling has limited discriminative power. This project's own MIA AUC on
the same models sits near 0.52–0.63 — far closer to chance than the anchor MIA of
92–100 implies. The two numbers cannot both be describing the same thing.

### 8.5 Class-structure analysis (the founding measurement)

`experiments/analyse_class_structure.py`, measured on `W_0`, 5,000 samples per set,
seed 42, 90.3 s, against a null control built from two disjoint halves of `D_r`.
4,352 channels measured per class (4,355 rows including the 3-channel stem).

| class | median SNR | mean SNR | max SNR | % channels above noise floor | layer1 | layer2 | layer3 | layer4 | fc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ship | 16.09 | 21.12 | 47.54 | 91.22 | 16.09 | 1.77 | 5.06 | 35.12 | 47.54 |
| frog | 13.08 | 22.69 | 48.84 | 91.11 | 1.42 | 10.08 | 13.08 | 40.05 | 48.84 |
| airplane | 10.29 | 21.56 | 56.07 | 90.07 | 1.59 | 10.29 | 3.91 | 35.94 | 56.07 |
| cat | 10.11 | 25.29 | 74.23 | 85.36 | 1.28 | 1.36 | 10.11 | 39.48 | 74.23 |
| horse | 8.24 | 29.02 | 83.77 | 84.15 | — | — | — | — | — |
| truck | 7.82 | 25.01 | 68.26 | 88.72 | — | — | — | — | — |
| bird | 6.90 | 28.54 | 88.53 | 87.36 | — | — | — | — | — |
| dog | 5.17 | 21.48 | 62.02 | 85.71 | — | — | — | — | — |
| deer | 4.59 | 29.08 | 94.84 | 89.71 | — | — | — | — | — |
| automobile | 3.90 | 25.61 | 74.25 | 87.87 | 1.49 | 1.39 | 3.90 | 47.00 | 74.25 |

(Per-group SNR for every class is in `results/analysis/class_structure/per_class_groups.csv`
and `summary.json`; the four rows above are the ones quoted in the README.)

**Against the predecessor project's instance-level forget set: 0.55% of channels above
the noise floor, where the null control gives 1.00% by construction.** Fewer stood out
than chance alone produces. Class-level: 84.1%–91.2%. Three orders of magnitude. This
is the founding result.

**The follow-on hypothesis fails.** Correlations across the ten classes against pure
`ACC_f`:

| statistic | Pearson r | Spearman ρ |
|---|---:|---:|
| median SNR | **−0.037** | −0.03 |
| mean SNR | +0.206 | — |
| max SNR | +0.109 | — |
| % channels above noise | +0.057 | — |
| max inter-class similarity | **−0.084** | +0.018 |

All null. Truck is **sixth of ten** on median SNR and **fifth** on channels above the
floor, yet worst by far on forgetting. Automobile has the **least** structure of any
class (3.90) and forgets 3.3× better than truck. **Structure explains the regime, not
the ranking.**

### 8.6 Inter-class similarity (derived from the same artefact)

Cosine similarity between per-class channel-contrast vectors (4,355 dims), written to
`results/writeup_package/class_structure_similarity.csv`:

| | airpl | autom | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | — | 0.27 | 0.28 | 0.20 | 0.20 | 0.15 | 0.15 | 0.24 | **0.41** | 0.24 |
| **automobile** | 0.27 | — | 0.14 | 0.13 | 0.07 | 0.11 | 0.08 | 0.19 | 0.24 | **0.32** |
| **bird** | 0.28 | 0.14 | — | 0.21 | 0.29 | 0.16 | 0.27 | 0.19 | 0.21 | 0.06 |
| **cat** | 0.20 | 0.13 | 0.21 | — | 0.19 | **0.30** | 0.23 | 0.15 | 0.16 | 0.14 |
| **deer** | 0.20 | 0.07 | 0.29 | 0.19 | — | 0.20 | **0.30** | 0.25 | 0.22 | 0.14 |
| **dog** | 0.15 | 0.11 | 0.16 | **0.30** | 0.20 | — | 0.17 | 0.24 | 0.15 | 0.12 |
| **frog** | 0.15 | 0.08 | 0.27 | 0.23 | **0.30** | 0.17 | — | 0.13 | 0.18 | 0.08 |
| **horse** | 0.24 | 0.19 | 0.19 | 0.15 | **0.25** | 0.24 | 0.13 | — | 0.20 | 0.22 |
| **ship** | **0.41** | 0.24 | 0.21 | 0.16 | 0.22 | 0.15 | 0.18 | 0.20 | — | 0.29 |
| **truck** | 0.24 | **0.32** | 0.06 | 0.14 | 0.14 | 0.12 | 0.08 | 0.22 | 0.29 | — |

The matrix recovers the semantic grouping **without being told it** — vehicles with
vehicles, animals with animals — which is the evidence the measurement is meaningful.
**Truck's nearest neighbour is automobile (0.32), and the relation is mutual.**

But similarity does **not** predict difficulty (r = −0.08), and **airplane is decisive
against the simple version**: highest similarity of any class (0.41, with ship) and
`ACC_f` 0.00.

### 8.7 Truck failure analysis (inference only)

`experiments/analyse_truck_predictions.py` classifies the 1,000 held-out truck images
with four models. The pure `C*` has no checkpoint (the search records genomes, not
weights) so it was rebuilt by replaying its chromosome through the deterministic
operators; the rebuild reproduces the published `ACC_f` exactly (42.10 / 30.90), which
is the check that the figure describes the published models.

Full predicted-class distribution, % of 1,000 truck test images:

| predicted as | `W_0` | `W_ref` | `C*` pure | `C*` hybrid |
|---|---:|---:|---:|---:|
| airplane | 0.7 | 13.5 | 15.6 | 20.3 |
| **automobile** | 2.8 | **68.4** | **16.7** | **20.4** |
| bird | 0.1 | 0.3 | 3.1 | 3.3 |
| cat | 0.3 | 1.8 | 8.6 | 9.2 |
| deer | 0.0 | 0.3 | 0.7 | 0.6 |
| dog | 0.0 | 0.4 | 0.4 | 0.4 |
| frog | 0.0 | 0.4 | 3.2 | 3.7 |
| horse | 0.0 | 0.6 | 3.0 | 3.8 |
| ship | 0.7 | 14.3 | 6.6 | 7.4 |
| **truck** | **95.4** | **0.0** | **42.1** | **30.9** |

A model that never saw a truck sends 68.4% of them to automobile. Pure MED-US sends
only 16.7% there and leaves 42.1% still called truck; the refinement moves a further
11.2 points out, most arriving at automobile. **The failure is a partial move along
the truck–automobile axis, not a random scattering** — the reference's destination is
the same destination, reached less far. Two independent measurements (activations of
`W_0`; predictions of four models) agree on the pair.

### 8.8 Ablation studies

**None were run.** Explicitly absent: NSGA-II vs random search at equal evaluation
budget; operator families in isolation; `class_contrast` vs `magnitude` vs `random`
selection; population/generation budget sensitivity; `max_level` sensitivity; seed
variance. The selector ablation is *implemented* (`selection.py` supports all three
rules) but was never run as an experiment.

### 8.9 Timing / wall clock

Measured on the local machine (§9). All ten searches ran to completion, zero failures.

| class | search elapsed (s) | search (min) | evaluated | cache hits | front size |
|---|---:|---:|---:|---:|---:|
| airplane | 931.8 | 15.5 | 233 | 277 | 10 |
| automobile | 714.0 | 11.9 | 301 | 209 | 10 |
| bird | 498.6 | 8.3 | 280 | 230 | 10 |
| cat | 599.4 | 10.0 | 319 | 191 | 10 |
| deer | 383.6 | 6.4 | 212 | 298 | 10 |
| dog | 419.4 | 7.0 | 231 | 279 | 10 |
| frog | 494.9 | 8.2 | 235 | 275 | 10 |
| horse | 437.0 | 7.3 | 213 | 297 | 10 |
| ship | 468.1 | 7.8 | 230 | 280 | 10 |
| truck | 494.3 | 8.2 | 257 | 253 | 10 |

Mean per-individual evaluation time during search: 3.998 s for airplane (min 1.56,
max 12.871). Cache hit rates of 37–58% are substantial and worth reporting.

**Full-fidelity re-measurement** of the ten front members, per class:

| class | mean per member | total |
|---|---:|---:|
| frog | 65.3 s | 10.9 min |
| airplane | 64.5 s | 10.7 min |
| ship | 63.5 s | 10.6 min |
| truck | 59.5 s | 9.9 min |
| automobile | 59.4 s | 9.9 min |
| cat | 57.7 s | 9.6 min |
| deer | 56.6 s | 9.4 min |
| dog | 56.5 s | 9.4 min |
| bird | 56.0 s | 9.3 min |

**End-to-end per class**, from the sweep-driver log timestamps (search + full fidelity
+ baseline check + anchor metrics + figure): **~30–35 minutes**.
Successive classes in the driver were 30, 35, 31 and 32 minutes apart.

**Refinement:** ~4–5 minutes per class (eight classes completed between 00:40 and 01:13).

**Reference training:** ~2.32 h per class on a Kaggle Tesla T4, 200 epochs.
**`W_0` training:** ~5 h on the local GTX 1650 (README estimate).
**Class-structure analysis:** 90.3 s.

### 8.10 All plots in the repository

No `.pdf` or `.svg` exists. Seventeen `.png`, all 300 dpi.

**Per-class Pareto fronts (10):**
```
results/search/plan_a_airplane/pareto_front_plan_a_airplane.png
results/search/plan_a_automobile/pareto_front_plan_a_automobile.png
results/search/plan_a_bird/pareto_front_plan_a_bird.png
results/search/plan_a_cat/pareto_front_plan_a_cat.png
results/search/plan_a_deer/pareto_front_plan_a_deer.png
results/search/plan_a_dog/pareto_front_plan_a_dog.png
results/search/plan_a_frog/pareto_front_plan_a_frog.png
results/search/plan_a_horse/pareto_front_plan_a_horse.png
results/search/plan_a_ship/pareto_front_plan_a_ship.png
results/search/plan_a_truck/pareto_front_plan_a_truck.png
```
Plotted values beside each: `pareto_front_plot_data.csv` (frog uses the older
`pareto_front_plot_table.csv`, written by the deliberately frozen frog-specific
plotter — different columns, same purpose).

**Write-up figures (7):**
```
results/writeup_package/figures/pure_vs_hybrid_acc_f_by_class.png
results/writeup_package/figures/pure_vs_hybrid_acc_r_by_class.png
results/writeup_package/figures/pure_vs_hybrid_composite_by_class.png
results/writeup_package/figures/operator_frequency_selected_cstar.png
results/writeup_package/figures/benchmark_comparison.png
results/writeup_package/figures/truck_failure_analysis.png
results/writeup_package/figures/class_structure_analysis.png
```

**Figures that do not exist:** seed-variance, ablation, baseline-comparison and
runtime plots. Each needs an experiment that has not been run.

---

## 9. HARDWARE & SOFTWARE

### 9.1 Local machine (all searches, refinements, analyses, figures)

| item | value |
|---|---|
| GPU | **NVIDIA GeForce GTX 1650**, 4.0 GB VRAM, compute capability 7.5 (Turing, TU117), 16 SMs |
| CPU | **AMD Ryzen 5 3550H** with Radeon Vega Mobile Gfx — 4 cores / 8 threads |
| RAM | **13.9 GB** |
| OS | **Windows 11**, build 10.0.26200 |
| Python | **3.11.x** (CPython, `C:\Users\...\Python311`) |
| PyTorch | **2.5.1+cu121** |
| torchvision | **0.20.1** |
| CUDA runtime | 12.1 · cuDNN 90100 |

**AMP is deliberately disabled.** Measured on this GPU with ResNet-18 / CIFAR-10,
batch 64: fp32 440 samples/s with finite loss; fp16 AMP 165 samples/s with loss going
NaN inside `layer4`. The GTX 1650 is the one Turing part *without* tensor cores, so
fp16 has no fast path — 2.7× slower and numerically unstable. Documented in
`configs/base.yaml`.

### 9.2 Kaggle (reference-model training only)

From `results/reference_training/class*_kaggle_manifest.json`:

| item | value |
|---|---|
| GPU | **NVIDIA Tesla T4**, 14.56 GB, compute capability 7.5 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 · torchvision 0.25.0+cu128 |
| CUDA compiled | 12.8 · cuDNN 91002 |
| OS | Linux 6.12.90+ x86_64, glibc 2.35 |
| Wall clock | ~2.32 h per reference, 200 epochs |

### 9.3 `requirements.txt` (verbatim)

```
# MEDUS Class Unlearning -- dependencies.
#
# Torch is NOT pinned here. It is installed separately, because the correct
# build differs by platform:
#
#   local (Windows, GTX 1650, CUDA 12.1):
#       pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
#
#   Kaggle:
#       torch and torchvision are PREINSTALLED with a CUDA build matching the
#       assigned accelerator. Do not reinstall them -- pip will resolve a build
#       that does not match Kaggle's driver and every evaluation will silently
#       fall back to CPU, or fail to see the GPU at all.
#
# See kaggle/README_KAGGLE.md.

# --- scientific core -------------------------------------------------------
numpy>=1.26,<3
pandas>=2.0
scikit-learn>=1.3          # MIA attack model (diagnostic only)

# --- config, IO ------------------------------------------------------------
PyYAML>=6.0

# --- plotting (Pareto fronts, dissertation figures) ------------------------
matplotlib>=3.7

# --- testing ---------------------------------------------------------------
pytest>=8.0
```

### 9.4 `requirements-torch.txt` (verbatim)

```
# LOCAL ONLY. Installed from the CUDA 12.1 wheel index:
#   pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
#
# torch 2.5.1 / cu121 supports compute capability 7.5 (GTX 1650, Turing).
#
# DO NOT run this on Kaggle: torch is preinstalled there with a build matched to
# the assigned accelerator, and replacing it breaks GPU access.
torch==2.5.1
torchvision==0.20.1
```

### 9.5 Resolved versions in the working environment

| package | version |
|---|---|
| Python | 3.11 |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1 |
| numpy | ≥1.26 (installed) |
| scikit-learn | ≥1.3 (installed) |
| matplotlib | **3.10.8** |
| PyYAML | ≥6.0 (installed) |
| pytest | ≥8.0 (installed) |
| **pymoo** | **NOT INSTALLED — not a dependency** |

There is no `pyproject.toml`; the project is run from `src/` via
`sys.path.insert`, not installed as a package.

### 9.6 Determinism

`configs/base.yaml` sets `deterministic: true`, and deterministic algorithms are
enforced inside the evaluator so that evaluating the same chromosome twice yields
identical objectives. This is what makes the objective cache sound and what allowed
`C*` to be reconstructed from its chromosome months later and reproduce the published
`ACC_f` exactly.

---

## 10. KEY DESIGN DECISIONS

### 10.1 Why NSGA-II — and why *not* a library

The repository never compares NSGA-II with NSGA-III, and **no such comparison exists**
— do not claim one. What is documented is the choice of a hand-written NSGA-II over a
library implementation, and the reasoning is about the genome:

> "Written from scratch (Deb et al., 2002) rather than taken from a library, because
> the genome is not a real vector: it is five integer gene blocks with different
> bounds, latent genes behind a `b` mask, and an expensive, cache-able objective
> function. The standard SBX / polynomial-mutation operators assume continuous
> variables and do not apply."

A defensible NSGA-II-vs-NSGA-III argument you can make from the artefacts (as
reasoning, not as a cited result): NSGA-III's reference-direction machinery targets
many-objective problems (M ≥ 4); with M = 3 and a population of 10, crowding distance
is adequate and reference directions would be over-specified. If you make it, mark it
as design reasoning rather than a measured comparison.

The three NSGA-II pieces are written as **pure functions of an objective matrix**, so
they are unit-tested against hand-worked examples without touching a GPU, and the
algorithm never imports the evaluator — it receives an injected `evaluate` callback.

### 10.2 Why these operators

- **The library is eight operators, deliberately narrow.** The predecessor's table
  carried twenty-two across two families, most gradient-based and most demoted by the
  end. Carrying it forward would ship a table where two thirds of the entries are
  disabled.
- **`REINIT` and `SIGN_FLIP` are absent from the library, not disabled by config**, so
  no config edit can bring them back. They were the most destructive in calibration
  and dominated fronts full of wrecked models.
- **`PRUNE` and `RANDOM_PRUNE` are kept as controls, not as strong methods.** They pin
  themselves to data-free selection (`magnitude`, `random`). "If a forget-informed
  operator cannot beat this, the 'forget-informed' part is doing nothing — which is
  exactly what the instance-level project found."
- **`MASK` is the operator the project is really testing.** Its selector is the
  class-activation contrast, so it is the one that can exploit the class-specific
  channels the structure analysis found.
- **Intensity ladders were copied unchanged** from the predecessor's calibrated table:
  measured on the same architecture and dataset, so re-deriving them would discard
  verified work.
- **`configs/operators/lookup.yaml` is the only place a ratio, sigma, limit or bit
  count may be written down.** Nothing else in the codebase hard-codes an operator
  hyperparameter.

### 10.3 Why this dataset, model and class

- **CIFAR-10 / ResNet-18** is the anchor paper's primary setting, and the setting in
  which its Table 1 reports six competing methods. Matching it is what makes the
  comparison possible at all.
- **The CIFAR ResNet-18 variant** (3×3 stem, no max-pool) is the AMUN / SalUn /
  Unlearn-Sparse convention for 32×32 inputs.
- **All ten classes** were swept because the anchor's numbers are ten-class averages
  and a single-class result cannot be dropped into its table.
- **Frog as the first class** was chosen by measurement, not convention: its structure
  rises monotonically with depth (SNR 1.42 → 10.08 → 13.08 → 40.05 → 48.84 across
  layer1…fc) and it is the only class with **four** layer groups above SNR 10. Ship has
  a marginally higher median but its median is carried by `layer1` with a dead
  `layer2` — on a water-background class that is most likely low-level colour
  statistics, a shortcut cue that would make a strong result easy to dismiss.
- **Per-class configs are deliberately duplicated rather than inherited**, so each is a
  self-contained record of what produced that class's result: "A shared base would let
  an edit made for one class silently change another, and the ten rows would stop being
  comparable."

### 10.4 Decisions that changed during development

| what changed | from | to | why |
|---|---|---|---|
| Forget-set granularity | instance-level (random subset) | **class-level** (one whole class) | Exhaustive instance-level search found nothing: best `S` = 1.158 over 10,534 strategies, against ~932 for retraining. The structure measurement explained why (0.55% of channels above a noise floor where chance gives 1.00%) and predicted that a genuinely distinct `D_f` would behave differently. |
| `f1` | `\|L_f − L_f(ref)\|` — a scalar target on cross-entropy | **JS divergence between predictive distributions** | Many very different models share one loss value; a model that confidently relabels every frog as "cat" can hit the reference's loss exactly while behaving nothing like it. |
| `f3` | `KL(P_ref(D_f) ‖ P(D_f))` | **relative parameter-change norm** | The old `f3` duplicated `f2` (Spearman +0.74 vs +0.36); a nominally three-objective search was really two-objective. |
| `W_ref` selection | full-test accuracy | **`D_r_test` accuracy**, ties on `D_r_test` loss | Full-test is diluted *and backwards*: the forget logit is never positively trained but still fluctuates, so an epoch placing a few more frogs in class 6 scores higher — rewarding the reference for recognising what it must never have seen. |
| Composite definition | `ACC_r × (100 − ACC_f) × MIA` (literature-review draft) | **`ACC_r × (1 − ACC_f)`** | Read from the anchor's released source; **MIA is not a term in it**. The docstring in `anchor.py` records the correction. |
| Refinement BatchNorm | training mode | **frozen (eval mode), buffer movement checked** | An unfrozen attempt passed every weight-based guard while `D_r` batches silently re-estimated the running statistics and undid the operator edit. |
| `best_S` selection | plain `max()` | **finite-value guard** | `S` is `nan` for an identity edit and every comparison against `nan` is `False`, so `max()` reported the unedited model as most selective. Affected only the diagnostic row for deer, dog and horse; no headline number changed. |
| Pure vs hybrid reporting | (considered merging) | **permanently separate** | The anchor's method is gradient-free, so only the pure table is like-for-like. Merging would overstate what MED-US alone achieves by 5 points of `ACC_f`. |

### 10.5 Deliberately not carried over from the predecessor

Named in the README: the instance-level evaluator as the main evaluator; random-
instance splitting in any form; the `loss_kl` objective; full-test checkpoint
selection for the reference; the gradient-based operator families; and the four-way
objective-mode branching "that made it possible to run a search whose objectives did
not mean what the report said they meant."

### 10.6 What "forgotten" means in this project

**Not 0% accuracy on the forget class.** A model at zero has learned to actively avoid
the answer, which is its own detectable signature. `W_ref` never saw the class and
misclassifies it the way any naive model would. **The target is the gap to `W_ref`**,
reported beside every number. This is a substantive framing choice and should be
argued in the problem-formulation chapter.

---

## 11. EXISTING WRITING

Substantial prose already exists and should be mined rather than rewritten:

| file | size | what it contains |
|---|---:|---|
| `README.md` | 216 lines | The project's argument: why it exists, the structure measurement, the objectives, the operator library, the chromosome, the pipeline, the reference-selection reasoning, what "forgotten" means, what was and was not carried over |
| `docs/artifact_manifest.md` | 22.9 KB | Provenance of every artefact; the storage decision; the ten-class sweep; the hybrid section; the `best_S` defect and its fix |
| `claudedocs/research_anchor_paper_20260827.md` | 16.0 KB | The anchor-paper selection, a seven-paper comparison table, and the anchor's Table 1 transcribed |
| `results/literature_alignment/protocol_validation_report.md` | 12.7 KB | Protocol validation against the anchor |
| `results/writeup_package/results_chapter_notes.md` | — | Bullet points for every results section, all numbers traced |
| `results/writeup_package/key_numbers_summary.md` | — | Single reference sheet of every headline number |
| `results/writeup_package/limitations_future_work_notes.md` | — | Seven limitations, each with what it does not license and what would fix it |
| `results/writeup_package/missing_figures_status.md` | — | What was generated, what was not, and why |
| `results/writeup_package/figure_inventory.md` | — | Every figure, where it is, what it is for |
| `docs/dissertation_outline.html` | — | 8-chapter, 62-section plan with word budgets |
| Module docstrings | — | `objectives.py`, `nsga2.py`, `selection.py`, `genome.py`, `anchor.py`, `lookup.yaml` are all written as explanatory prose and are quotable near-verbatim |

### 11.1 README — the core argument (verbatim extracts)

> **Why this project exists.** The predecessor project (`Experimental_Studies_2`)
> searched **instance-level** unlearning, where `D_f` is a random subset of training
> images. It searched exhaustively and found nothing:
>
> | | |
> |---|---|
> | evaluated strategies | **10,534** |
> | operator families | 3 |
> | selectors compared | 5 |
> | objective formulations | 4 |
> | best selectivity `S` ever measured | **1.158** |
> | `S` for retraining without `D_f` | **~932** |
>
> `S ≈ 1` means damage to `D_f` and damage to `D_r` were empirically identical — no
> strategy was selective, whatever the operator, selector, objective or search
> algorithm.
>
> A direct measurement of the network explained why. Per-channel activation contrast
> between `D_f` and `D_r`, against a null control built from two disjoint halves of
> `D_r`:
>
> | forget set | channels above the noise floor |
> |---|---|
> | instance-level `D_f` | **0.55%** |
> | pure noise (by construction) | 1.00% |
>
> Fewer channels stood out than chance alone produces. `D_f` and `D_r` were the same
> ten classes, so the network used the same features — and the same weights — for
> both. There was no forget-specific structure for any weight edit to remove.
>
> **That explanation makes a falsifiable prediction:** make `D_f` genuinely different
> and the structure should appear. It does.

> **Every class shows 84–91% of channels above the noise floor, against 0.55% for
> instance-level. The structure exists.** This project tests whether the operators can
> exploit it.

> **What "forgotten" means here. Not 0% accuracy on frogs.** A model at zero has
> learned to actively avoid the answer, which is its own detectable signature. `W_ref`
> never saw a frog and misclassifies frogs the way any naive model would. The target
> is therefore the **gap to `W_ref`**, reported next to every number.

### 11.2 The pipeline, as documented

```
1.  build the class split          D_f_train / D_r_train / D_f_test / D_r_test
2.  train_class_reference.py       W_ref on D_r_train only
3.  analyse_class_structure.py     D_f vs D_r activations, with a null control
4.  run_plan_a.py                  MicroGA / NSGA-II over the safe operators
5.  evaluate_class_front.py        full-fidelity re-measurement
6.  refine_candidate.py            optional, outside the search
```

### 11.3 Reproducibility engineering worth a methodology subsection

- **Self-contained per-class configs** rather than inheritance, so an edit for one
  class cannot silently change another.
- **Checkpoint provenance by sha256**, with `CheckpointMetadata` sidecar JSON beside
  every `.pt`.
- **Byte-for-byte verification when merging a measured row into a committed table**:
  `measure_refined_anchor.py` copies existing rows as raw CSV text, never parsing and
  re-serialising them, then re-reads the file and compares every preserved line,
  restoring a backup and exiting non-zero on any mismatch. All nine merges verified
  clean.
- **Resumable search** with per-generation checkpointing, including RNG state — "the
  caller is responsible for having restored the RNG state onto `self.rng` first —
  without that the resumed run draws a different stream and is no longer the same
  experiment."
- **The sweep driver refuses to touch finished work**: classes with an existing
  `full_fidelity/front_full_fidelity.csv` are skipped unless named explicitly.
- **Splits byte-compared** against the version-controlled local split before any
  imported checkpoint was accepted.
- **65 tests**, all passing: `test_anchor_metrics.py`, `test_class_split.py`,
  `test_objectives.py`, `test_operators.py`.

---

## 12. KNOWN LIMITATIONS — state these, do not let a reader discover them

1. **One dataset, one architecture, one `W_0`.** CIFAR-10 / ResNet-18 only.
2. **One seed (42).** Every `±` is class-to-class spread, not run-to-run variance. Two
   distinct variances are unmeasured: *search* variance (cheap — `W_0` and `W_ref` are
   fixed and evaluation is deterministic, so only the sampler changes) and *training*
   variance (expensive — a new reference per class per seed).
3. **No baseline re-implemented.** The comparison is against published numbers under
   the anchor's protocol and inherits every difference between two implementations.
4. **`C*` is selected on the test set it is then reported on** — best-of-ten on the
   reported metric, with no held-out selection split.
5. **The anchor MIA appears saturated.** Retraining scores exactly 100.00; SCRUB
   reaches `ACC_f` 0.00 with MIA 0.00. This project's own MIA AUC on the same models
   sits near chance.
6. **No ablation of any kind was run.** The claim that the evolutionary search is
   necessary, and the `MASK`-in-all-ten observation, are both undefended by ablation.
7. **Truck is unexplained.** Structure magnitude is ruled out (r = −0.04); inter-class
   similarity is suggestive but not predictive (r = −0.08, airplane is the
   counterexample). Why truck is hard remains open.
8. **Runtime is unoptimised** — `num_workers: 0` during search, batch cap 3, no
   parallel population evaluation. Wall-clock comparisons against published runtimes
   would not be like-for-like.
9. **The hybrid is gradient-based** and needs `D_f`, `D_r` and an optimiser step at
   unlearning time, which removes part of the deployment argument for a gradient-free
   method.

---

*End of context dump. Every number above was read from the repository at commit
`207a874`; nothing is estimated or inferred except where explicitly labelled.*
