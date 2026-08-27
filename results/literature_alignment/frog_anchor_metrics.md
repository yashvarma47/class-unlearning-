# Frog (class 6) under the anchor protocol

Anchor: Kodge, Saha & Roy. Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting. TMLR 07/2024.  
Paper: https://openreview.net/forum?id=BmI5p6wBi0  
Code: https://github.com/sangamesh-kodge/class_forgetting

Forget class: **6 (frog)**. Config: `search/plan_a_frog.yaml`. Loader sizes: `{'forget_train': 5000, 'retain_train': 45000, 'forget_eval': 5000, 'retain_eval': 45000, 'forget_test': 1000, 'retain_test': 9000}`.

Measured by `experiments/report_anchor_metrics.py`; no model was trained and no search was run. `C*` was rebuilt from the chromosome stored on its Pareto-front row and checked against that row's recorded objectives before being scored.

## Anchor-protocol metrics

`ACC_r` and `ACC_f` are **test-set** accuracies in percent. `composite = ACC_r x (1 - ACC_f)`, their `metric_function`. `MIA` is their `SVC_MIA`: an RBF SVC on true-class confidence, fit on `D_r_train` (member) against `D_f_train` (non-member) and scored as the fraction of `D_f_test` it calls non-member. Higher `ACC_r` is better, lower `ACC_f` is better, higher `MIA` is better.

| model | ACC_r (%) | ACC_f (%) | composite (%) | MIA (%) |
|---|---|---|---|---|
| `W_0` (original) | 94.51 | 97.30 | 2.55 | 0.00 |
| `W_ref` (retain-only reference) | 94.59 | 0.00 | 94.59 | 100.00 |
| `C*` (pure MED-US, front #8) | 92.52 | 8.30 | 84.84 | 94.20 |
| `C*_refined_bn_frozen` | 92.56 | 2.70 | 90.06 | 96.30 |

The MIA used the full shadow sets, as the anchor does (no subsampling).

## The anchor's own Table 1, for comparison

CIFAR-10 / ResNet-18, mean +/- std over all 10 target classes (https://arxiv.org/html/2312.00761v4). Our rows above are a **single class**, so they are not yet commensurable with these; see `protocol_validation_report.md`.

| method | ACC_r | ACC_f | MIA |
|---|---|---|---|
| Original | 94.89 +/- 0.31 | 94.89 +/- 2.75 | 0.03 +/- 0.03 |
| Retraining (gold standard) | 94.81 +/- 0.52 | 0 | 100 +/- 0 |
| NegGrad | 69.89 +/- 10.23 | 0.02 +/- 0.04 | 0 |
| NegGrad+ | 89.91 +/- 1.41 | 0.94 +/- 1.87 | 98.68 +/- 1.42 |
| Tarun et al. 2023 (UNSIR) | 92.20 +/- 0.72 | 10.89 +/- 8.79 | 61.5 +/- 25.86 |
| Kurmanji et al. 2023 (SCRUB) | 94.79 +/- 0.63 | 0 | 0 |
| Foster et al. 2024 (SSD) | 85.76 +/- 25.76 | 4.37 +/- 12.79 | 87.86 +/- 31.21 |
| Kodge et al. 2024 (the anchor's own method) | 94.19 +/- 0.50 | 0.03 +/- 0.09 | 95.5 +/- 14.23 |

## Our objectives and diagnostics (unchanged)

Kept as extra columns exactly as previously reported. `f3` is undefined for `W_ref`, which is an independently trained model rather than an edit of `W_0`; its distance from `W_0` is recorded separately in the CSV as `distance_from_W0_not_an_edit`.

| model | f1 JS to W_ref | f2 retain train loss | f3 edit cost | S | our MIA AUC |
|---|---|---|---|---|---|
| `W_0` (original) | 0.690076 | 0.001110 | 0.000000 | n/a | 0.5479 |
| `W_ref` (retain-only reference) | 0.000000 | 0.000893 | -- | -47001.4226 | 0.4960 |
| `C*` (pure MED-US, front #8) | 0.363704 | 0.018905 | 0.145054 | 281.7640 | 0.5237 |
| `C*_refined_bn_frozen` | 0.358538 | 0.022433 | 0.145066 | 297.2504 | 0.5235 |

### Accuracies and losses on all four splits

| model | D_f train acc | D_f train loss | D_f test acc | D_f test loss | D_r train acc | D_r train loss | D_r test acc | D_r test loss | KL to W_ref (diag) |
|---|---|---|---|---|---|---|---|---|---|
| `W_0` (original) | 1.0000 | 0.0011 | 0.9730 | 0.1127 | 1.0000 | 0.0011 | 0.9451 | 0.2079 | 8.9473 |
| `W_ref` (retain-only reference) | 0.0000 | 10.1997 | 0.0000 | 10.1853 | 1.0000 | 0.0009 | 0.9459 | 0.2191 | 0.0000 |
| `C*` (pure MED-US, front #8) | 0.0724 | 5.0153 | 0.0830 | 5.2857 | 0.9943 | 0.0189 | 0.9252 | 0.2597 | 2.4616 |
| `C*_refined_bn_frozen` | 0.0276 | 6.3395 | 0.0270 | 6.5671 | 0.9933 | 0.0224 | 0.9256 | 0.2666 | 2.4993 |

