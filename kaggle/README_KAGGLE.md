# Running MED-US Class Unlearning on Kaggle

Kaggle is the **overflow** environment, not the primary one. The local GTX 1650
(4 GB) runs the pipeline end to end; Kaggle exists for work that does not fit in
that budget — repeat seeds, larger populations, and validation on a second
forget class. See [When Kaggle is worth it](#when-kaggle-is-worth-it).

---

## Before you start

**Use the finished `_best_dr` W_ref checkpoint.** `train_class_reference.py`
rewrites `class6_frog_reference_best_dr.pt` every time `D_r_test` improves, so
the file exists and loads correctly from epoch 1 onwards. Existence does **not**
mean training finished. `scripts/package_for_kaggle.py` reads the checkpoint's
own epoch metadata and refuses to stage a live one — trust that check rather
than the file's presence.

**Never use `_best.pt` as W_ref.** For a retain-only reference, full-test
accuracy is the wrong selection criterion: 1,000 of the 10,000 test images are
frogs the model never trained on, so the number is diluted, and an epoch that
happens to classify more frogs correctly scores *higher* — rewarding the
reference for recognising the thing it must never have seen. Only `_best_dr.pt`
is selected on `D_r_test`.

---

## 1. Build the upload bundles

Locally, after W_ref training finishes:

```bash
python scripts/package_for_kaggle.py --require-reference
```

This writes to `dist/`:

| artefact | contents | size |
|---|---|---|
| `medus_class_code.zip` | source, configs, experiments, tests | ~130 KB |
| `medus_class_weights/` | `W_0`, `W_ref` (`_best_dr`), class split | ~171 MB |
| `kaggle_manifest.json` | file list, SHA-256 of every weight, exclusions | — |

`--require-reference` makes the script exit non-zero if W_ref is missing or
still live, so it cannot quietly produce a half-complete bundle.

They are split into two bundles deliberately: weights are ~85 MB each and change
on a different schedule from the code. Bundling them together would mean
re-uploading 171 MB to fix a config comment.

---

## 2. Required input datasets

Create **two** Kaggle Datasets:

| dataset slug | upload | notes |
|---|---|---|
| `medus-class-code` | `dist/medus_class_code.zip` | Kaggle auto-extracts the zip |
| `medus-class-weights` | contents of `dist/medus_class_weights/` | three files, flat |

CIFAR-10 itself is **not** uploaded. The notebook downloads it via torchvision
(requires *Internet: On*). If you prefer to keep the notebook offline, add any
CIFAR-10 Kaggle dataset as a third input and point `data.root` at it.

Attach both datasets to the notebook, and set **Accelerator: GPU** (T4 or P100)
and **Internet: On**.

---

## 3. Expected folder layout

Kaggle mounts inputs read-only at `/kaggle/input/<slug>/`. The notebook copies
the project into `/kaggle/working/`, which is the only writable location:

```
/kaggle/input/medus-class-code/MEDUS_Class_Unlearning/    # read-only
/kaggle/input/medus-class-weights/                        # read-only
    cifar10_resnet18_seed42_best.pt
    class6_frog_reference_best_dr.pt
    cifar10_class6_frog.json

/kaggle/working/MEDUS_Class_Unlearning/                   # writable
    src/  configs/  experiments/  tests/
    results/
        checkpoints/          <- weights copied here by the notebook
        splits/               <- split copied here
        search/               <- run outputs land here
    data/cifar10/             <- downloaded
```

**Why the project must be copied, not run from `/kaggle/input`:** `PROJECT_ROOT`
is derived as `Path(__file__).resolve().parents[3]` from
`src/medus_class/utils/config.py`, and every relative path in every config
resolves against it. Running from the read-only input mount would make
`PROJECT_ROOT` read-only, and the first attempt to write `results/` would fail.

---

## 4. Run commands

The notebook `kaggle_run_plan_a.ipynb` does all of this in order. To run by hand:

```bash
cd /kaggle/working/MEDUS_Class_Unlearning

# 1. sanity: the suite must be green before anything expensive
python -m pytest tests/ -q

# 2. objective smoke -- identity scores 0 on f1 and f3; f2/f3 independence
python experiments/smoke_objectives.py --config search/kaggle_class_frog_main.yaml

# 3. search smoke -- population 4, one generation
python experiments/run_plan_a.py --config search/kaggle_class_frog_smoke.yaml

# 4. the main search
python experiments/run_plan_a.py --config search/kaggle_class_frog_main.yaml

# 5. full-fidelity re-measurement -- nothing is reportable before this
python experiments/evaluate_class_front.py \
    --config search/kaggle_class_frog_main.yaml \
    --front results/search/kaggle_class_frog_main/pareto_front.csv
```

Download `results/search/` from the notebook output when finished.

---

## 5. Configs

| config | population | generations | purpose |
|---|---|---|---|
| `search/kaggle_class_frog_smoke.yaml` | 4 | 1 | mechanical check |
| `search/kaggle_class_frog_main.yaml` | 10 | 50 | the real run |

Both **inherit `search/plan_a_frog.yaml`**, so the objective set, operator
library, level cap and selector are the same objects as the local run rather
than copies. Anything that must match across environments cannot drift; a
deliberate difference would be visible as an override in the Kaggle file.

What legitimately differs:

| setting | local | Kaggle | why |
|---|---|---|---|
| `num_workers` | 0 | 2 | Windows re-spawns DataLoader workers per iteration; Linux forks cheaply |
| `data.root` | local copy | downloaded | no 341 MB in the bundle |
| `results_dir` | `results/search/plan_a_frog` | `.../kaggle_class_frog_main` | keeps the two runs' outputs apart |

Objectives, identical in both:

```
f1 = JS( P_ref(D_f) || P_cand(D_f) )     bounded by ln 2
f2 = L_r                                  retain loss
f3 = ||θ − θ₀||₂ / ||θ₀||₂                edit cost
```

---

## 6. Expected runtime

Measured locally on a GTX 1650; a Kaggle T4 is roughly 3–4× faster.

| stage | local | Kaggle (est.) |
|---|---|---|
| tests | 11 s | ~10 s |
| objective smoke | ~2 min | ~40 s |
| search smoke (4 × 1) | ~1 min | ~20 s |
| main search (10 × 50) | ~6 min | ~2 min |
| full-fidelity front | ~11 min | ~4 min |

The search itself is cheap — the operators are gradient-free, so a candidate
costs a handful of forward passes. **Kaggle's value is not speed on one run**;
it is running many.

Kaggle's 9-hour session limit is not a constraint for any of the above. It
*would* be for training a W_ref from scratch (~4.3 h locally), which is the one
job worth moving there if a second forget class is needed.

---

## When Kaggle is worth it

| use | worth it? | reasoning |
|---|---|---|
| repeating Plan A | marginal | ~6 min locally; only worth it for many seeds at once |
| larger search (bigger population, more generations) | **yes** | the clearest win — local runs are small to stay same-day |
| another class as validation | **yes** | needs a second ~4.3 h W_ref, which dominates the local GPU for a day |
| backing up full-fidelity evaluation | no | ~11 min locally, and it must match the run it evaluates |

---

## Troubleshooting

**`FileNotFoundError: reference checkpoint not found`** — the weights dataset is
not attached, or the notebook's copy step was skipped. Check
`results/checkpoints/` under the working copy.

**CUDA not available** — Accelerator is set to None. The notebook asserts this
early rather than silently running on CPU, where the search would take hours.

**Objectives come back `nan`** — usually a mismatched W_ref: a checkpoint for a
different class, or a corrupt download. Compare SHA-256 against
`kaggle_manifest.json`.

**The split is for the wrong class** — `get_or_create_class_split` raises rather
than proceeding, and names the class it found. Delete
`results/splits/cifar10_class6_frog.json` and let it rebuild, or re-copy it.

**Do not reinstall torch.** Kaggle preinstalls a build matched to the assigned
accelerator; `pip install torch` resolves a different one and breaks GPU access.
`requirements.txt` deliberately omits it.
