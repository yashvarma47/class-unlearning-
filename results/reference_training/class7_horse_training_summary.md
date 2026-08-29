# W_ref for class 7 (horse)

Retain-only reference: trained on `D_r_train` only, the 45,000 CIFAR-10 training images that are **not** horse. It never saw a single horse.

| | |
|---|---|
| assigned to | Aditya |
| forget class | 7 (horse) |
| epochs | 200 |
| seed | 42 |
| GPU | Tesla T4 |
| wall clock | 2.32 h |
| selection rule | best checkpoint selected by D_r_test accuracy, with D_r_test loss as tie-breaker (D_f_test is logged every epoch but NEVER influences selection) |

## Selected checkpoint

| | |
|---|---|
| epoch | 162 |
| `D_r_test` accuracy | 0.9423 |
| `D_r_test` loss | 0.2382 |
| `D_f_test` accuracy | 0.0000 (diagnostic only -- near zero is CORRECT) |
| `D_f_test` loss | 9.9951 |
| full test accuracy | 0.8481 (diluted by the held-out class; not a utility number) |

File: `class7_horse_reference_best_dr.pt`

## Split

| set | size |
|---|---:|
| `D_f_train` (excluded from training) | 5,000 |
| `D_r_train` (the training set) | 45,000 |
| `D_f_test` (diagnostic) | 1,000 |
| `D_r_test` (selection criterion) | 9,000 |
