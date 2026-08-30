# Table: retain-only reference models (`W_ref`), all ten classes

Generated 2026-08-30T20:45:08+00:00 by `experiments/build_writeup_package.py` from committed artefacts. Nothing here was recomputed, re-measured or re-run.

Each `W_ref` is a ResNet-18 trained from scratch on `D_r_train` only -- the 45,000
CIFAR-10 training images that are not the forget class. It never saw a single image
of that class, so it is the retraining gold standard against which every unlearned
model in this dissertation is measured.

| id | class | verdict | epoch | `D_f_test` | `D_r_test` | full test | sha256 |
|---:|:---|:---:|---:|---:|---:|---:|:---|
| 0 | airplane | **PASS** | 135 | 0.0000 | 0.9516 | 0.8564 | `283da38e314e` |
| 1 | automobile | **PASS** | 191 | 0.0000 | 0.9476 | 0.8528 | `5d07d03e6655` |
| 2 | bird | **PASS** | 159 | 0.0000 | 0.9526 | 0.8573 | `3750f7f565c1` |
| 3 | cat | **PASS** | 195 | 0.0000 | 0.9612 | 0.8651 | `60ba8b8bf71a` |
| 4 | deer | **PASS** | 189 | 0.0000 | 0.9457 | 0.8511 | `cca3b78022c8` |
| 5 | dog | **PASS** | 151 | 0.0000 | 0.9576 | 0.8618 | `f2bb585dbc49` |
| 6 | frog | **PASS** | 163 | 0.0000 | 0.9459 | 0.8513 | `c44f3f99e8a3` |
| 7 | horse | **PASS** | 162 | 0.0000 | 0.9423 | 0.8481 | `478d67d102fe` |
| 8 | ship | **PASS** | 181 | 0.0000 | 0.9502 | 0.8552 | `852fc08e8cb0` |
| 9 | truck | **PASS** | 174 | 0.0000 | 0.9514 | 0.8563 | `52c6e8d7c132` |

`D_f_test` accuracy is **0.0000 for all ten**, which is the correctness condition: a
model that never trained on a class must not classify it. It is diagnostic only and
never influenced checkpoint selection -- the best epoch was chosen on `D_r_test`
accuracy with `D_r_test` loss as tie-breaker.

`D_r_test` accuracy: **0.9506 +/- 0.0057** over the ten references
(min 0.9423, max 0.9612).

Full-test accuracy sits near 0.85 for every class because one class in ten is held
out by construction; it is not a utility number and should not be read as one.

Protocol, identical for all ten: 200 epochs, seed 42, split 5,000 / 45,000 / 1,000 / 9,000,
every split byte-compared against the version-controlled local split before import.
