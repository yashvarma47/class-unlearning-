# Kaggle reference training — `W_ref` for the class-level sweep

For **Yash**. The version to hand to Pragati and Aditya is
[`FRIEND_INSTRUCTIONS.md`](FRIEND_INSTRUCTIONS.md), which is deliberately shorter and
stricter.

---

## What this is for

The 10-class sweep needs one retain-only reference per class: a ResNet-18 trained on the
45,000 CIFAR-10 images that are *not* class *c*, so it has genuinely never seen that
class. `f1` is a Jensen–Shannon divergence measured **against** this model, and it is the
anchor protocol's gold-standard `Retraining` row, so every number in the sweep depends on
these ten models being trained identically.

Nine are missing. Class 6 (frog) is done, committed, and **must never be retrained** —
every published frog result was measured against the checkpoint already in the repo. The
driver refuses class 6 outright.

Locally each reference costs **~4 hours** on the GTX 1650 (measured: 14,527 s over 200
epochs). On a Kaggle T4/P100 it should be **45–75 minutes**. That is why this exists.

---

## The plan, in order

1. **Ship trial (now).** You run class 8 alone on Kaggle and validate the zip locally.
2. **Nothing else until it passes.** No parallel training, no other classes.
3. **Parallel (later, only after ship passes).** Pragati and Aditya get their assignment
   files and `FRIEND_INSTRUCTIONS.md`.

Assignments, once we get there:

| person | classes |
|---|---|
| Yash | airplane 0, automobile 1, bird 2 |
| Pragati | cat 3, deer 4, dog 5 |
| Aditya | horse 7, truck 9 |
| — | ship 8 comes from this trial; frog 6 is already done |

---

## Running the ship trial

### 1. Build and upload the code dataset

```bash
python scripts/package_for_kaggle.py --profile reference
```

Upload `dist/medus_class_code.zip` as a Kaggle Dataset named **`medus-class-code`**.
Kaggle auto-extracts it. **No weights dataset is needed** — reference training starts
from random initialisation and builds its own split per class.

`--profile reference` matters. Without it the script builds a *search* bundle and stages
`dist/medus_class_weights/` containing `W_0` and a class split — and the class it picks is
whatever `--forget-class` says, defaulting to 6. A reference-training upload that arrives
carrying a frog split next to cat/deer/dog is confusing at best and, if anyone wires it in,
wrong. The reference profile ships no weights directory at all, so there is nothing to
mis-wire.

### 2. Create the notebook

Upload `kaggle/reference_training/train_references_kaggle.ipynb`, then set:

* **Accelerator:** GPU (T4 or P100)
* **Internet:** On (torchvision downloads CIFAR-10)
* **Input:** attach `medus-class-code`

### 3. Run it

Cell 0 already says `ASSIGNMENT = "trial_ship.yaml"`. **Leave it.** Run all cells.

Cell 4 is a dry run that trains nothing and prints the safety block — assigned person,
class id, class name, the four split sizes, seed, epochs, GPU, output path and the
selection rule. Read it. If the class is not 8 (ship) or any split size is not
5000 / 45000 / 1000 / 9000, stop.

Cell 5 does the training (~45–75 min). Cell 6 builds the zip.

### 4. Download

From the Kaggle **Output** panel, download:

```
reference_outputs_trial_ship.zip
```

### 5. Validate locally

```bash
python kaggle/reference_training/validate_reference_zip.py \
    --zip reference_outputs_trial_ship.zip --expect-class 8
```

Writes `results/reference_training/reference_validation_summary.csv` and prints
`RESULT: PASS` or `RESULT: FAIL`. Exit code is non-zero on failure.

---

## What validation actually checks

Nothing in the zip is taken on trust — least of all the claim that the model never saw
the class. That one is verified by **running the model**, not by reading its sidecar.

| check | why |
|---|---|
| zip extracts cleanly | corrupt transfer |
| `_best_dr.pt` present | the only checkpoint anything reads |
| loads with `strict=True` | doubles as the architecture check: a different ResNet fails to load |
| metadata class id **and** name match | catches a file renamed by hand |
| split sizes 5000 / 45000 / 1000 / 9000 | catches a wrong or stale split |
| measured `D_f_test` accuracy ≤ 0.05 | **the one that matters** — a high value means forget-class images leaked into training, and every objective computed against this reference would be quietly wrong |
| measured `D_r_test` accuracy ≥ 0.90 | catches a collapsed run |
| training log present, 200 epochs | catches a truncated run |

Thresholds for reference: the finished frog reference measures `D_f_test` **0.0000** and
`D_r_test` **0.9459**. `MIN_RETAIN_TEST_ACC = 0.90` is deliberately loose — it catches
"training collapsed", not "half a point below frog".

---

## Files here

| file | what it is |
|---|---|
| `train_references_kaggle.ipynb` | the Kaggle notebook; one editable line |
| `train_references.py` | the driver — validates the assignment, prints the safety block, calls the trainer |
| `package_outputs.py` | builds the download zip from an allowlist |
| `validate_reference_zip.py` | local validation of a downloaded zip |
| `trial_ship.yaml` | the trial: class 8 only |
| `classes_yash.yaml` | airplane 0, automobile 1, bird 2 |
| `classes_pragati.yaml` | cat 3, deer 4, dog 5 |
| `classes_aditya.yaml` | horse 7, truck 9 |

The driver does **not** contain a training loop. It shells out to
`experiments/train_class_reference.py` — the same unmodified script, the same recipe,
that produced the frog reference. A second implementation would drift, and the ten models
would stop being comparable.

---

## Safety properties worth knowing

* **No "train everything" path.** The driver iterates the assignment's `classes` list and
  nothing else. A missing or empty list is an error.
* **Class 6 is refused** with an explicit message, in `FROZEN_CLASSES`.
* **Class name must match class id.** The name is redundant with the id on purpose — a
  mismatch means the YAML was hand-edited and one of the two is wrong.
* **An existing `_best_dr.pt` stops the run** unless `--overwrite` is passed.
* **Split sizes are checked twice** — in the driver before anything starts, and again
  inside `train_class_reference.py` before the first epoch.
* **The zip is an allowlist**, so `_latest.pt`, `_final.pt`, CIFAR-10 and caches cannot
  travel by accident.

---

## Outputs, per class

```
class<ID>_<name>_reference_best_dr.pt      the reference
class<ID>_<name>_reference_best_dr.json    provenance sidecar
class<ID>_<name>_training_log.csv          per-epoch history
class<ID>_<name>_training_summary.md       readable summary
class<ID>_<name>_environment.json          torch / CUDA / GPU snapshot
cifar10_class<ID>_<name>.json              the split, so it can be re-verified
manifest.json                              what was requested and produced
```

---

## Local dry run (no training)

Safe to run on the laptop — it builds the split and prints the safety block, then stops:

```bash
python kaggle/reference_training/train_references.py \
    --assignment kaggle/reference_training/trial_ship.yaml --dry-run
```
