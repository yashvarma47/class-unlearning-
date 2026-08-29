# W_ref availability -- all ten CIFAR-10 classes

Generated 2026-08-29T16:13:29+00:00 by `experiments/summarise_references.py`.

**10 of 10 classes have a usable retain-only reference.**

A class counts as available only when the checkpoint is present at `results/checkpoints/class<ID>_<name>_reference_best_dr.pt` **and** a `PASS` row names it in `reference_validation_summary.csv`. A PASS row alone proves a zip was validated once; it does not prove the file is still where every config expects it.

| id | class | W_ref | `D_f_test` acc | `D_r_test` acc | epoch | log epochs | seed | sha256 | source zip |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 0 | airplane | **yes** | 0.0000 | 0.9516 | 135 | 200 | 42 | `283da38e314e` | `reference_outputs_yash_airplane.zip` |
| 1 | automobile | **yes** | 0.0000 | 0.9476 | 191 | 200 | 42 | `5d07d03e6655` | `reference_outputs_yash_automobile.zip` |
| 2 | bird | **yes** | 0.0000 | 0.9526 | 159 | 200 | 42 | `3750f7f565c1` | `reference_outputs_yash_bird.zip` |
| 3 | cat | **yes** | 0.0000 | 0.9612 | 195 | 200 | 42 | `60ba8b8bf71a` | `reference_outputs_pragati_cat.zip` |
| 4 | deer | **yes** | 0.0000 | 0.9457 | 189 | 200 | 42 | `cca3b78022c8` | `reference_outputs_pragati_deer.zip` |
| 5 | dog | **yes** | 0.0000 | 0.9576 | 151 | 200 | 42 | `f2bb585dbc49` | `reference_outputs_pragati_dog.zip` |
| 6 | frog | **yes** | 0.0000 | 0.9459 | 163 | 200 | 42 | `c44f3f99e8a3` | `(local training, never packaged)` |
| 7 | horse | **yes** | 0.0000 | 0.9423 | 162 | 200 | 42 | `478d67d102fe` | `reference_outputs_aditya.zip` |
| 8 | ship | **yes** | 0.0000 | 0.9502 | 181 | 200 | 42 | `852fc08e8cb0` | `reference_outputs_trial_ship.zip` |
| 9 | truck | **yes** | 0.0000 | 0.9514 | 174 | 200 | 42 | `52c6e8d7c132` | `reference_outputs_aditya.zip` |

## Storage

Every one of these checkpoints is **external and git-ignored**. Ten references at ~85 MiB each is ~850 MiB, which alone would consume most of the free 1 GiB Git LFS allowance on top of the 256 MiB the frog chain already uses. The storage decision is still open -- see section 6 of `docs/artifact_manifest.md`. What *is* tracked is enough to identify the right file: the sha256 above, the split, the validation verdict and the per-class training summary.

## Complete

All ten classes have a validated reference. The 10-class sweep is no longer blocked on reference training.

