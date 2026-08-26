"""Forward-pass evaluation metrics.

Everything here runs under ``torch.no_grad()`` with the model in ``eval()``
mode. That is a supervisor requirement -- objective values are computed by
forward passes only -- and it also matters for correctness in two ways that are
easy to get wrong:

* ``eval()`` makes BatchNorm use its stored running statistics instead of the
  batch statistics. In ``train()`` mode the measured accuracy would depend on
  how samples happened to be grouped into batches, so the same model would score
  differently on the same data.
* ``no_grad()`` prevents the evaluation itself from building a graph, which on a
  4 GB card is the difference between fitting and not fitting.

A single pass produces everything the three objectives need: aggregate loss and
accuracy for ``f1``/``f2``, and the per-sample loss and confidence that the MIA
in :mod:`medus.evaluation.privacy` consumes for ``f3``. Evaluating twice would
double the dominant cost of an SEC call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class EvalOutput:
    """Result of one forward-pass evaluation over a loader.

    Attributes
    ----------
    loss:
        Mean cross-entropy over the split.
    accuracy:
        Top-1 accuracy in ``[0, 1]``.
    n_samples:
        Number of samples evaluated.
    per_sample_loss:
        Cross-entropy per sample. Low loss on a training sample is the classic
        membership signal.
    per_sample_confidence:
        Max softmax probability per sample -- the second membership signal.
    per_sample_correct:
        Boolean correctness per sample.
    """

    loss: float
    accuracy: float
    n_samples: int
    per_sample_loss: np.ndarray
    per_sample_confidence: np.ndarray
    per_sample_correct: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    collect_per_sample: bool = True,
) -> EvalOutput:
    """Evaluate ``model`` over ``loader`` with forward passes only.

    Parameters
    ----------
    collect_per_sample:
        If ``False``, the per-sample arrays come back empty. Set it to ``False``
        for splits whose per-sample scores are not needed (``D_r``), to avoid
        holding 45 000 floats for a diagnostic.

    Raises
    ------
    ValueError
        If the loader is empty -- silently returning ``nan`` here would surface
        much later as an unexplained objective value.

    Notes
    -----
    The model's training mode is restored on exit, so calling this between two
    operators of one chromosome cannot change what the next operator sees.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    losses: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    corrects: list[np.ndarray] = []

    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            sample_loss = F.cross_entropy(logits, labels, reduction="none")
            predictions = logits.argmax(dim=1)
            correct = predictions == labels

            total_loss += float(sample_loss.sum().item())
            total_correct += int(correct.sum().item())
            total_samples += labels.size(0)

            if collect_per_sample:
                losses.append(sample_loss.detach().cpu().numpy())
                confidences.append(
                    F.softmax(logits, dim=1).max(dim=1).values.detach().cpu().numpy()
                )
                corrects.append(correct.detach().cpu().numpy())
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("cannot evaluate an empty loader")

    empty = np.array([], dtype=np.float32)
    return EvalOutput(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        n_samples=total_samples,
        per_sample_loss=np.concatenate(losses) if losses else empty,
        per_sample_confidence=np.concatenate(confidences) if confidences else empty,
        per_sample_correct=np.concatenate(corrects) if corrects else empty.astype(bool),
    )


def forgetting_objective(forget_accuracy: float) -> float:
    """``obj1`` -- the forgetting objective, **minimised**.

    ``obj1 = forget_acc``: the accuracy of the unlearned model on ``D_f``.
    Lower is better, because the model should no longer classify the forgotten
    samples correctly.

    Beware the sign convention. The identity chromosome scores ``obj1 = 1.0``,
    which is the **worst** possible value -- the model still remembers ``D_f``
    perfectly. It is not "good forgetting". Report
    :func:`forgetting_score` when a human needs to read the number.
    """
    return float(forget_accuracy)


def uniform_predictor_loss(num_classes: int) -> float:
    """Cross-entropy of a maximally uncertain predictor: ``ln(C)``.

    A model that outputs a uniform distribution over ``C`` classes has
    ``-log(1/C) = log(C)`` cross-entropy on every sample, regardless of the
    label. For CIFAR-10 that is ``ln(10) = 2.3026``.

    This is the natural ceiling for a *forgetting* objective. Below it the model
    is uncertain; above it the model is confidently **wrong** -- it has moved
    probability mass onto a specific incorrect class. Those are different
    behaviours, and only the first is what unlearning should produce: the
    retrained reference, which genuinely never saw ``D_f``, sits at
    ``L_f = 0.1873``, i.e. 8% of this value.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    return math.log(num_classes)


def forget_loss_objective(forget_loss: float, cap: float | None = None) -> float:
    """``f1`` in ``loss_kl`` mode -- the negated forget loss, optionally capped.

    ``f1 = -min(L_f, cap)``, minimised. Minimising the negative maximises the
    forget-set loss, which is the supervisor-specified direction.

    Why the cap exists
    ------------------
    Cross-entropy is unbounded above, so an uncapped ``-L_f`` has no minimum and
    the search chases it without limit. The first ``loss_kl`` pilot measured the
    consequence directly: best ``f1`` reached ``-1.8e35``, 21 of 119 evaluations
    were numerically exploded, and 13 of the 20 final Pareto-front members were
    such models -- legitimately non-dominated, since nothing beats them on f1.

    It also destroyed the per-generation normalisation. With one outlier setting
    the range, all 98 numerically sane individuals normalised to bit-identical
    ``f1`` values, so the objective stopped separating real candidates at all.

    Capping at :func:`uniform_predictor_loss` fixes both while keeping the
    supervisor's formula intact -- it is a ceiling on ``L_f``, not a different
    objective. Re-scored against the same pilot data, the ``f1`` range becomes
    ``-2.3026 .. -0.0008`` and the count of distinct normalised values rises
    from 1 to 29.

    Parameters
    ----------
    cap:
        Maximum ``L_f`` counted. ``None`` disables capping, reproducing the
        original unbounded behaviour -- kept so the recorded negative result
        stays reproducible.

    Notes
    -----
    Capping is deliberately NOT clamping the loss itself: ``forget_loss`` is
    still recorded raw on the result, so how far past the cap a candidate went
    remains visible in the diagnostics even though the objective ignores it.
    """
    if cap is None:
        return -float(forget_loss)
    if cap <= 0:
        raise ValueError(f"forget-loss cap must be positive, got {cap}")
    return -float(min(forget_loss, cap))


def memorisation_gap_objective(
    forget_accuracy: float, reference_forget_accuracy: float
) -> float:
    """``obj1`` in ``gap_to_reference`` mode -- distance from the retrained model.

    ``obj1 = |forget_acc - reference_forget_acc|``, minimised.

    Why this mode exists
    --------------------
    :func:`forgetting_objective` drives ``forget_acc`` towards **zero**. Phase 8
    measured what a model that never trained on ``D_f`` actually scores on
    ``D_f``, by retraining ResNet-18 on the 45 000 retain samples only:

    ==========================  ==========  =========
    Model                       forget_acc  test_acc
    ==========================  ==========  =========
    Original (memorised)        1.0000      0.9479
    Retrained reference         0.9438      0.9441
    ==========================  ==========  =========

    The reference scores **0.9438** on data it has never seen -- indistinguishable
    from its own test accuracy (they differ by 0.0003). It classifies those
    images correctly because it learned the classes from other examples; it
    simply has no memory of these particular ones. Its MIA AUC is 0.5024, i.e.
    chance.

    So the correct target is ``forget_acc ~= 0.944``, not 0. Driving to zero
    demands the model be *actively wrong* about those images, which is well
    beyond what retraining achieves and is reachable only by damaging the
    network.

    What this fixes, and what it does not
    -------------------------------------
    **Fixes:** under :func:`forgetting_objective` a destroyed model scored
    ``obj1 ~= 0.1`` -- the *best* value in the population. Here it scores
    ``|0.1 - 0.944| = 0.844``, the worst. That inversion is the point of this
    mode.

    **Does not fix:** the coupling between ``obj1`` and ``obj2``. Simulating this
    objective over the 120 Phase 7 pilot individuals moves
    ``corr(obj1, obj2)`` from ``-0.9992`` to ``+0.9966`` -- the sign flips, the
    magnitude does not. Because ~78% of candidates fall *below* the reference,
    the absolute value is inactive for them and ``obj1 = ref - forget_acc``
    tracks ``obj2 = 1 - forget_acc`` almost exactly.

    That coupling is a property of the operator library, not of this objective:
    no strategy currently available forgets ``D_f`` without losing general
    accuracy by the same amount. A different objective definition cannot
    manufacture a trade-off the operators cannot physically produce. What the
    objectives *do* separate is ``obj3``: ``corr(obj1, obj3) = -0.81``, so
    privacy leakage falls as the model is damaged, giving a genuine
    fidelity-versus-privacy trade-off in place of the previous meaningless
    one-dimensional line.

    Absolute value, not one-sided
    -----------------------------
    Over-forgetting is penalised too, deliberately. A model scoring 0.20 where
    the reference scores 0.944 is conspicuously *wrong* on exactly the forgotten
    samples, and "suspiciously bad here" is itself a membership signal -- an
    attacker learns those samples were in the training set. One-sided
    ``max(0, forget_acc - ref)`` would score such a model 0, i.e. perfect.

    Parameters
    ----------
    reference_forget_accuracy:
        Accuracy on ``D_f`` of a model trained without ``D_f``. Must be measured
        on the **same** ``D_f`` subset this objective is computed on -- see
        :class:`medus.evaluation.sec.SEC`, which measures it rather than
        hard-coding it, for the same reason ``original_test_acc`` is measured.

    Raises
    ------
    ValueError
        If ``reference_forget_accuracy`` is outside ``[0, 1]``. An accuracy
        cannot be, and a silently wrong reference would bias every objective
        value in the run by a constant.
    """
    if not 0.0 <= reference_forget_accuracy <= 1.0:
        raise ValueError(
            f"reference_forget_accuracy must be in [0, 1], got "
            f"{reference_forget_accuracy}"
        )
    return float(abs(forget_accuracy - reference_forget_accuracy))


@torch.no_grad()
def reference_logits(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> torch.Tensor:
    """Cache a fixed model's logits over a loader, in loader order.

    The retrained reference never changes, so its outputs are computed once at
    SEC construction and reused by every ``f3`` evaluation. Recomputing them per
    chromosome would double the cost of the objective for no benefit.

    The loader must be deterministic and unshuffled -- ``forget_eval`` is -- or
    the cached rows would not line up with the candidate's rows.
    """
    was_training = model.training
    model.eval()
    chunks: list[torch.Tensor] = []
    try:
        for images, _ in loader:
            chunks.append(model(images.to(device, non_blocking=True)).detach())
    finally:
        model.train(was_training)
    return torch.cat(chunks)


@torch.no_grad()
def kl_to_reference(
    model: nn.Module,
    loader: DataLoader,
    cached_reference_logits: torch.Tensor,
    device: str = "cpu",
) -> float:
    """``obj3`` in ``loss_kl`` mode -- KL divergence from the retrained reference.

    .. math::

        f_3 = \\mathrm{KL}(P_r \\| P)
            = \\frac{1}{|\\mathcal{D}|}\\sum_{x}\\sum_{c}
              P_r(c \\mid x)\\,\\log\\frac{P_r(c \\mid x)}{P(c \\mid x)}

    where :math:`P_r` is the retrained reference's predictive distribution and
    :math:`P` the candidate's, both softmax over the ``C`` classes. Evaluated on
    ``D_f`` -- the only region where the two models are supposed to differ.

    Direction matters. This is the **forward** KL with the reference first, which
    is *mass-covering*: it grows sharply wherever the candidate assigns low
    probability to a class the reference is confident about. A destroyed model,
    which spreads probability roughly uniformly or confidently onto the wrong
    class, is therefore penalised hard. The reverse direction ``KL(P || P_r)``
    would be mode-seeking and far more forgiving of exactly that failure.

    Zero if and only if the candidate's predictive distribution matches the
    reference's everywhere on the evaluation set -- so the retrained model itself
    scores exactly 0.

    Numerics
    --------
    Computed from log-softmax rather than ``log(softmax(...))``. Softmax outputs
    are never exactly zero, so the divergence is finite, but taking the log of a
    tiny probability directly loses precision where this objective is most
    sensitive.

    Raises
    ------
    ValueError
        If the cached reference logits do not match the loader's sample count,
        which would mean the two models were compared on different data.
    """
    was_training = model.training
    model.eval()

    total = 0.0
    seen = 0
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            candidate_logits = model(images)

            reference_slice = cached_reference_logits[seen : seen + images.size(0)]
            if reference_slice.size(0) != images.size(0):
                raise ValueError(
                    f"cached reference logits exhausted at sample {seen}: the "
                    f"cache holds {cached_reference_logits.size(0)} rows but the "
                    f"loader yielded more. The two must cover the same samples."
                )

            log_p_ref = F.log_softmax(reference_slice.to(candidate_logits.device), dim=1)
            log_p_cand = F.log_softmax(candidate_logits, dim=1)
            p_ref = log_p_ref.exp()

            # sum_c P_r * (log P_r - log P), summed over classes then samples
            total += float((p_ref * (log_p_ref - log_p_cand)).sum().item())
            seen += images.size(0)
    finally:
        model.train(was_training)

    if seen == 0:
        raise ValueError("cannot compute KL over an empty loader")
    if seen != cached_reference_logits.size(0):
        raise ValueError(
            f"loader yielded {seen} samples but the reference cache holds "
            f"{cached_reference_logits.size(0)}; the two models would be "
            f"compared on different data"
        )
    return total / seen


def normalise_objectives(
    objectives: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    """Per-generation min-max normalisation, applied per objective.

    .. math::

        f'_i = \\frac{f_i - f_i^{\\min}}{f_i^{\\max} - f_i^{\\min}}

    Supervisor-specified. Two things worth being clear about:

    1. **It cannot change the Pareto fronts.** Min-max is a strictly increasing
       transform of each objective independently, and Pareto dominance is
       invariant under such transforms. The fronts are identical with or without
       it. Its real value is making objectives of wildly different scale
       readable side by side -- which matters here because ``f1 = -L_f`` is
       unbounded while ``f2`` and ``f3`` are not.
    2. **Zero range is not an error.** When every individual scores the same on
       an objective, the range is 0 and the objective carries no discriminating
       information for that generation. Those entries map to 0.0 rather than
       dividing by zero. MED-US hits this routinely -- ``obj2`` was exactly 0.0
       for many individuals in early generations of the Phase 7 pilot.

    Values are **not** comparable across generations: the same normalised 0.5
    means different things in different generations, because the range is
    recomputed each time. Raw objective values are what get recorded.
    """
    if not objectives:
        return []

    n_objectives = len(objectives[0])
    columns = list(zip(*objectives))
    bounds = []
    for column in columns:
        finite = [v for v in column if math.isfinite(v)]
        if finite:
            bounds.append((min(finite), max(finite)))
        else:
            bounds.append((0.0, 0.0))

    normalised: list[tuple[float, ...]] = []
    for row in objectives:
        scaled = []
        for index in range(n_objectives):
            low, high = bounds[index]
            spread = high - low
            scaled.append(0.0 if spread <= 0.0 else (row[index] - low) / spread)
        normalised.append(tuple(scaled))
    return normalised


def forget_loss_band_objective(forget_loss: float, target: float) -> float:
    """``f1 = |L_f - target|`` -- forget as much as the reference did, no more.

    The capped objective ``-min(L_f, ln C)`` asks for the largest possible
    forget loss and stops rewarding further only at the ceiling. A search under
    it marches to that ceiling: in the round-2 ablation, 8 of 10 winners in two
    arms sat exactly at ln(C) with retain loss ABOVE their forget loss, i.e.
    destroyed rather than unlearned.

    This asks a different question. The retrained reference -- a model that
    genuinely never saw D_f -- has forget loss 0.1873. That is what successful
    unlearning looks like. Overshooting it is not better forgetting, it is
    damage, and this objective penalises it symmetrically.

    Re-scoring the 132 saved full-fidelity candidates under this objective cut
    median retain loss from 0.8469 to 0.0108 and wrecked candidates from 24 to
    5. It could not change any candidate's selectivity, because S is a function
    of the losses rather than of f1 -- which is precisely why it has to be
    tested as a SEARCH objective rather than only as a re-ranking.

    Minimised, like every MED-US objective. Zero is exact agreement with the
    reference.
    """
    if target <= 0:
        raise ValueError(f"forget-loss target must be positive, got {target}")
    return abs(float(forget_loss) - float(target))


def forgetting_score(forget_accuracy: float) -> float:
    """Human-readable forgetting, ``1 - forget_acc``. **Higher is better.**

    Never use this as the NSGA-II objective -- NSGA-II minimises, so it would
    invert the search.
    """
    return float(1.0 - forget_accuracy)


def utility_objective(test_accuracy: float, original_test_accuracy: float) -> float:
    """``obj2`` -- normalised utility loss, **minimised**.

    ``obj2 = max(0, original_test_acc - test_acc) / original_test_acc``

    Normalising by the original accuracy makes the objective comparable across
    models and datasets whose baseline accuracies differ (ResNet-18 on CIFAR-10
    versus VGG19 on Tiny ImageNet). Clamping at zero means a strategy that
    *improves* test accuracy scores 0 rather than a negative value, which would
    otherwise dominate the Pareto front for a reason unrelated to unlearning.

    Raises
    ------
    ValueError
        If ``original_test_accuracy`` is not positive.
    """
    if original_test_accuracy <= 0:
        raise ValueError(
            f"original_test_accuracy must be > 0, got {original_test_accuracy}"
        )
    return float(max(0.0, original_test_accuracy - test_accuracy) / original_test_accuracy)
