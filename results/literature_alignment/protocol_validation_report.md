# Protocol validation: MED-US class unlearning against the anchor paper

**Anchor.** Sangamesh Kodge, Gobinda Saha, Kaushik Roy. *Deep Unlearning: Fast and Efficient
Gradient-free Class Forgetting.* Transactions on Machine Learning Research, 07/2024.
Paper: <https://openreview.net/forum?id=BmI5p6wBi0> · arXiv: <https://arxiv.org/abs/2312.00761> ·
Code: <https://github.com/sangamesh-kodge/class_forgetting> (branch `master`).

**Scope of this report.** Evaluation layer only. No model was trained, no search was run, no
objective was changed, and no previously reported frog number moved. Four already-existing models
were re-scored under the anchor's metrics: `W_0`, `W_ref`, the pure gradient-free `C*`, and the
corrected `C*_refined_bn_frozen`.

---

## 1. Exact metric definitions used

Every formula below was read out of the anchor's released source, not out of the paper's prose.
That distinction mattered: our own literature review had recorded the composite score as
`ACC_r × (100 − ACC_f) × MIA`, and the code says otherwise.

### `ACC_r` — retained-class test accuracy

Micro-accuracy over all test samples whose label is **not** the forget class, in percent. The anchor
computes it in `utils.py::test` by building a confusion matrix over one combined loader and pooling
the nine retain classes:
`retain_acc = Σ_{i∉F} cm[i,i] / Σ_{i∉F} cm[i,:]`. We hold `D_r_test` as a separate loader already,
so we read it directly; the two are the same number.

**Which split.** Test, not train. Two pieces of evidence: their Table 1 `Original` row reports
`ACC_f = 94.89 ± 2.75`, which cannot be a train-set number (a converged model is at ~100 % on its
own training data — our `W_0` measures exactly `1.0000` on `D_f_train`); and `scripts/our_cifar10.sh`
runs `main.py` without `--val-set-mode`, so the reported pass is the `"Final Test Set"` evaluation
over `test_loader`.

### `ACC_f` — forget-class test accuracy

The same quantity restricted to the forget class: `forget_acc = Σ_{i∈F} cm[i,i] / Σ_{i∈F} cm[i,:]`,
in percent. Measured on `D_f_test` (1 000 frog test images).

### `composite` — the anchor's `metric`

```
utils.py::metric_function(x, y) = x * (1 - y)      # x = retain_acc, y = forget_acc, both in [0,1]
                                                   # logged as 100 * metric
demo.py                          retain_acc * (100 - forget_acc) / 10000   # same thing, % scale
```

**The MIA is not a factor in it.** This is the correction to our earlier literature review, which
is now annotated in `claudedocs/research_anchor_paper_20260827.md`.

### `MIA` — the anchor's `SVC_MIA`

`utils.py::SVC_MIA` / `SVC_fit_predict`, wired in `main.py` (the `--do-mia` branch, lines ~1044-1071):

```python
SVC_MIA(shadow_train = train_retain_loader,   # D_r_train  -> label 1 (member)
        shadow_test  = train_forget_loader,   # D_f_train  -> label 0 (non-member)
        target_train = None,
        target_test  = test_forget_loader,    # D_f_test   -> the set the score is read off
        model        = model)
```

Three details that a paraphrase loses, and all three were wrong in our first written description:

1. **The feature is the true-class probability**, `torch.gather(prob, 1, target)` — the probability
   the model assigns to the sample's ground-truth label. It is *not* the max softmax that our own
   AUC-based MIA uses.
2. **The attacker is fit on retain-train against forget-train**, and then scored on a third,
   disjoint set. It is not "forget-train members versus forget-test non-members".
3. **The reported number is `1 − mean(predict(D_f_test))`** — the fraction of forget-class *test*
   images the attacker calls non-members, in percent. Higher is better.

Classifier: `sklearn.svm.SVC(C=3, gamma='auto', kernel='rbf')`.

### Our metrics, unchanged

`f1` (JS to `W_ref` on `D_f`), `f2` (retain train loss), `f3` (relative edit cost), `S` (selectivity),
our MIA AUC, and accuracy + loss on all four splits are all carried through as extra columns. None of
their definitions were touched.

---

## 2. Where each metric is implemented

| metric | implementation |
|---|---|
| `ACC_r`, `ACC_f` | `src/medus_class/evaluation/metrics.py::evaluate` over `loaders.retain_test` / `loaders.forget_test`, assembled by `src/medus_class/evaluation/anchor.py::anchor_metrics_from_accuracies` |
| `composite` | `src/medus_class/evaluation/anchor.py::anchor_composite` |
| anchor `MIA` | `src/medus_class/evaluation/anchor.py::anchor_mia` → `true_class_confidence` + `svc_membership_accuracy` |
| `f1` JS to `W_ref` | `src/medus_class/evaluation/objectives.py::js_to_reference` (unchanged) |
| `f2` retain loss | `src/medus_class/evaluation/metrics.py::evaluate` on `loaders.retain_eval` (unchanged) |
| `f3` edit cost | `src/medus_class/evaluation/objectives.py::relative_parameter_delta` (unchanged) |
| `S` selectivity | `src/medus_class/evaluation/objectives.py::selectivity` (unchanged) |
| our MIA AUC | `src/medus_class/evaluation/privacy.py::compute_mia_auc` (unchanged) |
| driver | `experiments/report_anchor_metrics.py` |
| table renderer | `experiments/write_anchor_markdown.py` |
| tests | `tests/test_anchor_metrics.py` (17 tests) |

`C*` has no checkpoint of its own — the search stored genomes, not weights — so
`report_anchor_metrics.py::rebuild_candidate` re-decodes the chromosome recorded on its Pareto-front
row and re-executes the same deterministic, gradient-free operators against the same `W_0`. It then
refuses to report anything unless the recomputed `f1`, `f2`, `f3`, `D_f_test` accuracy and
`D_r_test` accuracy all match that row to `1e-4`. They did.

---

## 3. Do `W_0` and `W_ref` behave as the anchor protocol expects?

Yes. This is the check that decides whether our implementation is theirs.

| model | `ACC_r` | `ACC_f` | `MIA` | anchor's expectation |
|---|---|---|---|---|
| `W_0` (original) | 94.51 | 97.30 | **0.00** | their `Original` row: 94.89 ± 0.31 / 94.89 ± 2.75 / **0.03 ± 0.03** |
| `W_ref` (retain-only) | 94.59 | **0.00** | **100.00** | their `Retraining` row: 94.81 ± 0.52 / **0** / **100 ± 0** |

* `W_ref` reproduces the gold-standard row **exactly** on the two values that validate the
  implementation: `ACC_f = 0` and `MIA = 100`. A retrained model's forget-test confidences look like
  its forget-train confidences, so the attacker calls all 1 000 of them non-members. Had our MIA
  wiring been wrong — the wrong feature, the wrong fit sets, or the sign of the final `1 - mean(...)`
  — this number would not have come out at 100.
* `W_0` scores `MIA = 0.00` against their `0.03 ± 0.03`, i.e. inside their reported spread.
* `W_0`'s `ACC_f = 97.30` sits above their `94.89 ± 2.75` because theirs is a mean over all ten
  classes and ours is frog alone; frog is one of the easier CIFAR-10 classes for this model.
* `ACC_r` is ~0.3 points below theirs for both baselines, consistent with our `W_0` simply being a
  slightly weaker CIFAR-10 ResNet-18 (94.51 % vs 94.89 %) rather than with any protocol difference.

**Conclusion: the anchor protocol is correctly implemented.**

---

## 4. Where the frog result actually stands

| model | `ACC_r` ↑ | `ACC_f` ↓ | composite ↑ | `MIA` ↑ |
|---|---|---|---|---|
| `W_0` | 94.51 | 97.30 | 2.55 | 0.00 |
| `W_ref` (gold) | 94.59 | 0.00 | 94.59 | 100.00 |
| `C*` (pure gradient-free) | 92.52 | 8.30 | 84.84 | 94.20 |
| `C*_refined_bn_frozen` | 92.56 | 2.70 | 90.06 | 96.30 |
| *anchor's own method (10-class mean)* | *94.19 ± 0.50* | *0.03 ± 0.09* | *~94.16* | *95.5 ± 14.23* |

Two things this makes precise, and they point in opposite directions:

* **On privacy we are already competitive.** `C*_refined_bn_frozen` scores `MIA = 96.30` against the
  anchor's `95.5 ± 14.23` and NegGrad+'s `98.68 ± 1.42`, and it beats SSD (`87.86 ± 31.21`), UNSIR
  (`61.5 ± 25.86`) and SCRUB (`0`) outright under this attack. Purely gradient-free `C*` is at 94.20.
* **On the accuracy pair we are not.** `ACC_f = 2.70` is ~90× the anchor's `0.03`, and `ACC_r =
  92.56` is ~1.6 points below their `94.19`. The composite lands at 90.06 against their ~94.16.
  Under their headline metric we are behind every method in Table 1 except UNSIR and NegGrad.

**Is the result ready to compare?** The *measurement* is; the *experiment* is not. Their Table 1
rows are means ± std over **all ten target classes**, and ours is a single class with n = 1. A
one-class number cannot be placed in that table, and we do not yet know whether `ACC_f = 2.70` is a
frog-specific artefact or a systematic MED-US ceiling.

---

## 5. What must happen before the 10-class sweep

1. **Train the nine remaining retain-only references** `W_ref^(c)` for `c ≠ 6`. Unavoidable: `f1` is
   defined against `W_ref`, and the anchor's `Retraining` row needs one per class anyway. This is
   the dominant cost of the sweep.
2. **Generalise the split and config plumbing beyond class 6.** `results/splits/cifar10_class6_frog.json`
   and `configs/search/plan_a_frog.yaml` are single-class artefacts; `build_class_split` already
   takes any label, so this is config/pathing work, not new science.
3. **Decide the seed count.** The anchor reports mean ± std over 10 classes. Three seeds per class
   (30 runs) gives us both a per-class std and a cross-class one; one seed per class gives only the
   latter and matches their protocol more literally. Recommend 3 — the SSD row's ±25.76 shows how
   much a single unstable run can distort a 10-class mean.
4. **Freeze the reporting path.** `report_anchor_metrics.py` currently hard-codes the frog front and
   `--front-position 8`; the sweep version needs to take a class and locate that class's front.
5. **Do not enlarge the search yet.** A bigger 50×100 search that moves `ACC_f` from 2.70 to ~1.5 is
   still ~50× behind the anchor, while an unmeasured 10-class result cannot be published at all.
   Comparability first.

---

## 6. Mismatches and open uncertainties

Recorded rather than papered over. None of these invalidate section 3, but each is a place where our
numbers could differ from theirs for a reason that is not the method.

1. **Their released MIA call may score the wrong model.** `main.py` line ~1070 passes `model=model`
   to `SVC_MIA`, not `unlearn_model` (which is what line 888 assigns). Whether this is a bug or
   simply relies on the unlearning function mutating `model` in place depends on the method. If it
   is a bug, their published `MIA` column may describe the pre-unlearning model for some rows. We
   score the unlearned model explicitly. **Unresolved — worth an email to the authors before the
   sweep is written up.**
2. **`exp` vs `softmax` on the model output.** Their `collect_prob` calls `torch.exp(model(data))`
   because their networks end in `log_softmax`; our ResNet-18 returns raw logits, so
   `true_class_confidence` applies `softmax`. Same quantity, different porting; documented in
   `anchor.py`.
3. **Confusion matrix vs separate loaders.** They derive `ACC_r`/`ACC_f` from one confusion matrix
   over a combined loader; we read them from `D_r_test` and `D_f_test` loaders that were already
   separate. Arithmetically identical, but it means we never cross-check against a confusion matrix.
   Adding a per-class confusion matrix would also feed the CMIA critique
   (<https://arxiv.org/html/2506.20893>) about *where* forgotten samples get routed.
4. **`scripts/our_cifar10.sh` ships `vgg11_bn`, not `resnet18`.** The ResNet-18 CIFAR-10 numbers in
   Table 1 come from the same script with `--arch resnet18`; the exact `--our-alpha-r` /
   `--our-alpha-f` / `--our-samples` used for that architecture are not in the shipped script. Only
   matters if we later reproduce *their method*, not for our metrics.
5. **Composite for `W_ref` is `94.59`, not `100`.** Correct and expected — the composite is bounded
   above by `ACC_r`, so the gold standard cannot score 100. Flagging it so nobody reads it as a bug.
6. **`S` is `nan` for `W_0` and `−47001` for `W_ref`.** Also expected and pre-existing: `S` is a
   ratio of deltas against `W_0`, so it is undefined for `W_0` itself, and `W_ref` is not an edit of
   `W_0` at all. Unchanged from `final_objectives.json`.
7. **No subsampling was used.** The SVC was fit on the full 45 000 + 5 000 shadow set, as the anchor
   does (`subsampled = False` in every row). The `--max-shadow-per-group` escape hatch exists but was
   not needed — the fit takes ~70 s.

---

## Files produced

* `results/literature_alignment/frog_anchor_metrics.csv`
* `results/literature_alignment/frog_anchor_metrics.json`
* `results/literature_alignment/frog_anchor_metrics.md`
* `results/literature_alignment/protocol_validation_report.md` (this file)
