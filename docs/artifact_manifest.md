# Artifact manifest — class-level frog (CIFAR-10 label 6) unlearning

What exists, where it lives, whether git has it, and what would be lost if it disappeared.
Written during an artifact-safety pass; no model was trained, no search was run, no objective
changed, and no completed frog result was modified.

Sizes are bytes as measured on disk. "Tracked / ignored" is the state **after** this pass — the
`.gitignore` rewrite and the Git LFS setup described in §3 are part of it.

---

## 1. The reproducible chain

These five artifacts are the whole result. Remove any one and the frog experiment cannot be
recomputed or checked.

| artifact | path | size | tracked | ignored | disposition | why it matters |
|---|---|---:|---|---|---|---|
| **`W_0`** original model | `results/checkpoints/cifar10_resnet18_seed42_best.pt` | 89,489,966 | yes | no | **Git LFS** | The model every edit starts from and `f3` is measured against. Every objective value in the project is relative to these exact weights. |
| `W_0` provenance | `results/checkpoints/cifar10_resnet18_seed42_best.json` | 509 | yes | no | commit normally | Seed, epoch, metrics, torch version. Lets a reader detect a swapped checkpoint. |
| **`W_ref`** retain-only reference | `results/checkpoints/class6_frog_reference_best_dr.pt` | 89,490,794 | yes | no | **Git LFS** | Trained on `D_r` only, never saw a frog. `f1` is a JS divergence *against this model*, and it is the anchor protocol's gold-standard row. Irreplaceable without a full retrain. |
| `W_ref` provenance | `results/checkpoints/class6_frog_reference_best_dr.json` | 1,277 | yes | no | commit normally | Records that selection was on `D_r_test`, not full test accuracy — the distinction the config warns about. |
| **`C*`** pure gradient-free | `results/checkpoints/class6_frog_C_star_pure_gradient_free.pt` | 44,775,426 | yes | no | **Git LFS** | The headline pure result. Created by this pass — see §2. |
| `C*` provenance | `results/checkpoints/class6_frog_C_star_pure_gradient_free.json` | 1,011 | yes | no | commit normally | Carries the chromosome and the front row it was replayed from. |
| **`C*_refined_bn_frozen`** | `results/search/plan_a_frog_bn_frozen_refined/refined_best.pt` | 44,772,340 | yes | no | **Git LFS** | The accepted BatchNorm-frozen refinement — `ACC_f` 2.70 %, the best forgetting we have. |
| the class split | `results/splits/cifar10_class6_frog.json` | 397,993 | yes | no | commit normally | The exact index partition. Every accuracy in the project is defined on it. |

### `C*` did not exist as a checkpoint until this pass

The Plan A search recorded genomes, not weights, so the headline **pure gradient-free** result
existed only as row `front_position = 8` of `front_full_fidelity.csv` plus the operator code needed
to replay it. Every other model in the result set had a `.pt`. A search of this repo and, read-only,
of `Experimental_Studies`, `Experimental_Studies_2`, `MEDUS_CHECKPOINT_TRANSFER`,
`MEDUS_MODEL_BACKUPS` and `UnlearningCode-CIPHAR` found no `C*` checkpoint anywhere — those folders
hold only instance-level originals and retain references. No old file or old git state was modified.

`experiments/save_cstar_checkpoint.py` replays the recorded chromosome through the same
deterministic weight-surgery operators against the same `W_0`, writes the checkpoint, then **reloads
it from disk** and re-measures. Verifying the reloaded file rather than the in-memory model is the
point: what needs proving is that the file is right. On any mismatch the script deletes the
checkpoint and its sidecar and exits non-zero.

Result — every recorded value reproduced **exactly**, delta `0.00e+00` on all eight:

| metric | reloaded | recorded on front row |
|---|---:|---:|
| `f1` JS to `W_ref` | 0.36370449 | 0.36370449 |
| `f2` retain train loss | 0.01890546 | 0.01890546 |
| `f3` edit cost | 0.14505422 | 0.14505422 |
| `D_f_test` accuracy | 0.08300000 | 0.08300000 |
| `D_r_test` accuracy | 0.92522222 | 0.92522222 |
| `D_f_train` accuracy | 0.07240000 | 0.07240000 |
| `D_r_train` accuracy | 0.99431111 | 0.99431111 |
| `S` selectivity | 281.76403633 | 281.76403633 |

Anchor protocol on the reloaded checkpoint: `ACC_r` 92.5222 %, `ACC_f` 8.3000 %, composite
84.8429 %, MIA 94.2000 % — identical to the `C_star` row already in
`results/literature_alignment/frog_anchor_metrics.csv`. Nothing about the completed frog result
changed; it simply has a file now.

---

## 2. Search and result records

| artifact | path | size | tracked | ignored | disposition | why it matters |
|---|---|---:|---|---|---|---|
| full-fidelity Pareto front | `results/search/plan_a_frog/full_fidelity/front_full_fidelity.csv` | 7,088 | yes | no | commit normally | Holds every front member's **chromosome**, which is what makes `C*` replayable at all. Row `front_position = 8` is `C*`. |
| screening Pareto front | `results/search/plan_a_frog/pareto_front.csv` | 2,810 | yes | no | commit normally | The search-stage front, before full-fidelity re-measurement. |
| search summary | `results/search/plan_a_frog/summary.json` | 1,150 | yes | no | commit normally | Run configuration and generation history. |
| baselines | `results/search/plan_a_frog/full_fidelity/baselines.json` | 795 | yes | no | commit normally | `W_0` and `W_ref` measured at full fidelity; `S` is a ratio against these. |
| final objectives | `results/search/plan_a_frog/final_objectives.json` | 4,314 | yes | no | commit normally | The five-row reported table (`W_0`, `W_ref`, best-`S`, `C*`, refined). |
| refinement record | `results/search/plan_a_frog_bn_frozen_refined/refinement.json` | 4,563 | yes | no | commit normally | The accepted refinement, with the four acceptance conditions. |
| refinement summary | `results/search/plan_a_frog_bn_frozen_refined/refinement_summary.md` | 6,674 | yes | no | commit normally | Prose account of the accepted run. |
| refined provenance | `results/search/plan_a_frog_bn_frozen_refined/refined_best.json` | 1,311 | yes | no | commit normally | Metrics and the BatchNorm-frozen note. |
| rejected refinement | `results/search/plan_a_frog_refined/refinement.json` | 3,807 | yes | no | commit normally | The **rejected** first attempt that exposed the BatchNorm defect. A negative result worth keeping. |
| earlier refinement summary | `results/search/plan_a_frog/refinement_summary.md` | 6,055 | yes | no | commit normally | Companion to the above. |
| Pareto figure | `results/search/plan_a_frog/pareto_front_plan_a_frog.png` | 353,540 | yes | no | commit normally | Dissertation figure. |
| Pareto plot table | `results/search/plan_a_frog/pareto_front_plot_table.csv` | 1,758 | yes | no | commit normally | The numbers behind the figure. |
| reference training log | `results/class6_frog_reference_log.csv` | 16,112 | yes | no | commit normally | Per-epoch record of the `W_ref` retrain. |

## 3. Anchor-protocol reports

| artifact | path | size | tracked | ignored | disposition | why it matters |
|---|---|---:|---|---|---|---|
| anchor metrics (machine) | `results/literature_alignment/frog_anchor_metrics.csv` | 1,847 | yes | no | commit normally | `ACC_r`, `ACC_f`, composite, anchor MIA for all four models, plus our own columns. |
| anchor metrics (full) | `results/literature_alignment/frog_anchor_metrics.json` | 4,269 | yes | no | commit normally | Same, with loader sizes and MIA shadow-set sizes. |
| anchor metrics (readable) | `results/literature_alignment/frog_anchor_metrics.md` | 3,878 | yes | no | commit normally | The table against the anchor's own Table 1. |
| protocol validation | `results/literature_alignment/protocol_validation_report.md` | 12,746 | yes | no | commit normally | Where each formula came from in the anchor's code, and the `W_0`/`W_ref` sanity check. |
| literature review | `claudedocs/research_anchor_paper_20260827.md` | — | yes | no | commit normally | Why this anchor, and the correction to the composite formula. |

## 4. Deliberately external or ignored

Nothing here is needed to reproduce a published number.

| artifact | path | size | ignored | disposition | why |
|---|---|---:|---|---|---|
| `W_ref` final-epoch | `results/checkpoints/class6_frog_reference_final.pt` | 44,774,358 | yes | **keep external** | Training bookkeeping. The config points at `_best_dr`; nothing reads this. |
| `W_ref` latest | `results/checkpoints/class6_frog_reference_latest.pt` | 89,490,606 | yes | **keep external** | Resume-from checkpoint. Same reason. |
| `W_0` duplicate | `dist/medus_class_weights/cifar10_resnet18_seed42_best.pt` | 89,489,966 | yes | **keep external** | Byte-identical copy of `W_0` staged for the Kaggle upload. `dist/` is a build directory. |
| evaluation history | `results/search/plan_a_frog/evaluation_history.csv` | 99,178 | yes | keep external | Every evaluation of the run, including failures. Useful for post-hoc analysis, too noisy for the repo. |
| smoke run | `results/search/plan_a_frog_smoke/` | — | yes | keep external | Pipeline smoke test, not a result. |
| console logs | `results/*.out` | — | yes | keep external | Reproducible by re-running; the JSON/CSV records are authoritative. |
| CIFAR-10 | `data/` | ~340 MB | yes | keep external | Public dataset, downloaded on demand. |

## 5. Storage decisions

**Git LFS, four files, 268,528,526 bytes (~256 MiB).** `.gitattributes` tracks `*.pt`, `*.pth`,
`*.ckpt` and `*.safetensors`; datasets and cache directories are **not** tracked by LFS. GitHub's
free tier allows 1 GiB of LFS storage and 1 GiB/month of bandwidth, so this uses roughly a quarter
of the allowance and every fresh clone spends ~256 MiB of it.

Two notes for later:

* `W_0` and `W_ref` are 85 MiB each because they carry optimiser and scheduler state alongside the
  weights; `C*` and the refined model are 43 MiB because they carry weights only. Re-saving the two
  large ones weights-only would halve the LFS footprint. Not done here — it would rewrite files that
  every existing result was measured against.
* The `.gitignore` rewrite this pass required is subtle. `results/checkpoints/` and
  `results/search/` were directory exclusions, and **git cannot re-include a file whose parent
  directory is excluded**, so they became `results/checkpoints/*` and `results/search/*` with
  explicit `!` negations for the four checkpoints and their JSON sidecars. Verified with
  `git check-ignore` in both directions: the four are visible, and `*_final.pt`, `*_latest.pt`,
  the `dist/` duplicate, `evaluation_history.csv`, the smoke run, the `.out` logs and `data/` all
  remain ignored.

---

## 6. Ship (class 8) — imported, deliberately **not** in Git LFS

**The ship `W_ref` exists locally.** It was trained on Kaggle (Tesla T4, torch 2.10.0+cu128,
2.29 h, 200 epochs, seed 42), downloaded as **`reference_outputs_trial_ship.zip`**, validated by
`kaggle/reference_training/validate_reference_zip.py` (verdict `PASS`), and installed by
`kaggle/reference_training/import_reference_zip.py`.

Checkpoint **sha256 begins `852fc08e`** (full:
`852fc08e8cb0c2882996e3d63702c21f8c7e96b013315f277549d3474bd13c5e`), byte-identical to the member
inside the zip. The zip itself is sha256 `2badb411…`. Measured from its final location: `D_f_test`
accuracy **0.0000** (loss 10.3602), `D_r_test` accuracy **0.9502** (loss 0.1986), selected epoch
**181** of 200.

| artifact | path | size | tracked | ignored | disposition |
|---|---|---:|---|---|---|
| **ship `W_ref`** | `results/checkpoints/class8_ship_reference_best_dr.pt` | 89,491,299 | **no** | **yes** | **external for now — see below** |
| provenance sidecar | `results/checkpoints/class8_ship_reference_best_dr.json` | 1,238 | no | yes | external, follows the `.pt` |
| training log | `results/class8_ship_reference_log.csv` | 16,037 | no | yes | external, follows the `.pt` |
| environment snapshot | `results/class8_ship_reference_environment.json` | 386 | no | yes | external, follows the `.pt` |
| training summary | `results/reference_training/class8_ship_training_summary.md` | 1,038 | yes | no | commit normally |
| Kaggle run manifest | `results/reference_training/class8_ship_kaggle_manifest.json` | 1,045 | yes | no | commit normally |
| validation summary | `results/reference_training/reference_validation_summary.csv` | — | yes | no | commit normally |
| baseline wiring check | `results/reference_training/ship_reference_validation.md` | — | yes | no | commit normally |
| the ship split | `results/splits/cifar10_class8_ship.json` | 397,993 | yes | no | commit normally |
| the source zip | `reference_outputs_trial_ship.zip` | 83,505,420 | no | **no** | **keep external — do not commit** |

### Why the ship checkpoint is not in LFS

**This is a deliberate hold, not an oversight.** The frog chain — `W_0`, frog `W_ref`, `C*` and
`C*_refined_bn_frozen`, 256 MiB — is already in LFS and stays there. Adding ship would take that to
~341 MiB, and the eight remaining references would take it past **1 GiB**, which is the entire free
GitHub allowance, with bandwidth spent again on every clone.

So the decision for reference checkpoints beyond frog is **pending**. The realistic options, for
when there is something to decide with:

1. **All ten in LFS** (~1.0–1.1 GiB) — needs a paid LFS data pack.
2. **Weights-only re-save first.** These files are 85 MiB because they carry optimiser and scheduler
   state; the weights alone are ~43 MiB. Re-saving the nine *new* references weights-only would put
   all ten under ~600 MiB. Safe for the new ones — unlike `W_0` and frog `W_ref`, nothing has been
   measured against a specific byte layout of a file that does not exist yet.
3. **References stay external** — kept in the Kaggle output zips and a local backup, with only the
   metadata, logs, checksums and validation reports in git. Cheapest, and the checksums make the
   external files verifiable.

Until that is decided, the ship `.pt`, its sidecar, its log and its environment file remain
git-ignored, and `reference_outputs_trial_ship.zip` is not committed either. What *is* committed is
everything needed to recognise the right file if it turns up again: the sha256, the split, the
validation verdict and the training log summary.

---

## 7. The other eight references — imported 2026-08-28/29, all external

All ten CIFAR-10 classes now have a validated retain-only reference. Eight arrived from Kaggle
in seven zips (Aditya's carried two classes), were validated by
`kaggle/reference_training/validate_reference_zip.py` and installed by
`import_reference_zip.py`. Frog was trained locally and never packaged, so it was validated in
place against the same thresholds and recorded with `source_zip = "(local training, never
packaged)"`.

| id | class | `D_f_test` | `D_r_test` | epoch | sha256 | source zip |
|---:|---|---:|---:|---:|---|---|
| 0 | airplane | 0.0000 | 0.9516 | 135 | `283da38e314e` | `reference_outputs_yash_airplane.zip` |
| 1 | automobile | 0.0000 | 0.9476 | 191 | `5d07d03e6655` | `reference_outputs_yash_automobile.zip` |
| 2 | bird | 0.0000 | 0.9526 | 159 | `3750f7f565c1` | `reference_outputs_yash_bird.zip` |
| 3 | cat | 0.0000 | 0.9612 | 195 | `60ba8b8bf71a` | `reference_outputs_pragati_cat.zip` |
| 4 | deer | 0.0000 | 0.9457 | 189 | `cca3b78022c8` | `reference_outputs_pragati_deer.zip` |
| 5 | dog | 0.0000 | 0.9576 | 151 | `f2bb585dbc49` | `reference_outputs_pragati_dog.zip` |
| 6 | frog | 0.0000 | 0.9459 | 163 | `c44f3f99e8a3` | (local, never packaged) |
| 7 | horse | 0.0000 | 0.9423 | 162 | `478d67d102fe` | `reference_outputs_aditya.zip` |
| 8 | ship | 0.0000 | 0.9502 | 181 | `852fc08e8cb0` | `reference_outputs_trial_ship.zip` |
| 9 | truck | 0.0000 | 0.9514 | 174 | `52c6e8d7c132` | `reference_outputs_aditya.zip` |

Every one is 200 log epochs, seed 42, split 5000 / 45000 / 1000 / 9000, and every split inside
every zip was byte-compared against the version-controlled local split before import.

### Storage: unchanged, and now the pressure is real

**All nine non-frog references are external and git-ignored, and so is frog's.** Only `W_0`,
frog's `W_ref`, frog's `C*` and frog's refined checkpoint are in Git LFS — 256 MiB of the free
1 GiB allowance. Adding the nine new references at ~85 MiB each would be **~765 MiB more**,
taking the repository past the free tier on its own.

The three options from section 5 stand, and option 2 is now clearly the right one to cost out:
these nine files are 85 MiB because they carry optimiser and scheduler state, and a weights-only
re-save would put all ten references at ~430 MiB total. Unlike `W_0` and frog's `W_ref`, nothing
has been measured against a specific byte layout of the eight new files, so re-saving them is
safe. **Still pending explicit approval — nothing has been added to LFS.**

What is committed for each: the sha256, the split, the validation verdict and thresholds, the
per-class training summary and the Kaggle run manifest. Enough to recognise the right file, not
enough to reconstruct it. The source zips (~83 MB each, ~750 MB total) are ignored via
`reference_outputs_*.zip` and are the only copy of the weights outside `results/checkpoints/`.
