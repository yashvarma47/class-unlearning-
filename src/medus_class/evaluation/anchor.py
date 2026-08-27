"""The anchor protocol: Kodge, Saha & Roy (TMLR 2024) class-forgetting metrics.

Why this module exists
----------------------
The supervisor asked that the class-level experiments follow a published
experimental study so our numbers can be compared with someone else's. The study
chosen is

    Sangamesh Kodge, Gobinda Saha, Kaushik Roy.
    "Deep Unlearning: Fast and Efficient Gradient-free Class Forgetting."
    Transactions on Machine Learning Research, 07/2024.
    https://openreview.net/forum?id=BmI5p6wBi0
    code: https://github.com/sangamesh-kodge/class_forgetting (branch ``master``)

It is the only shortlisted study whose primary setting is ours -- CIFAR-10,
ResNet-18, one whole class forgotten, gradient-free, no retraining -- and whose
Table 1 already carries Retrain, NegGrad, NegGrad+, UNSIR, SCRUB and SSD for the
same setting. To place a MED-US row in that table we must measure what they
measured, the way they measured it.

Nothing here replaces our own objectives. ``f1``/``f2``/``f3``, ``S`` and the
AUC-based MIA in :mod:`medus_class.evaluation.privacy` are untouched and stay in
every report as extra columns. This module only adds their three numbers.

What the anchor actually computes -- read from their code, not from the paper
-----------------------------------------------------------------------------
Every formula below was taken from their released source rather than inferred
from the paper's prose, because two of the three differ from what a reader would
guess.

``ACC_r`` / ``ACC_f``
    ``utils.py::test`` builds a confusion matrix over the evaluation loader and
    reads accuracy off it, pooling the nine retain classes into one number and
    the one forget class into another. That is plain micro-accuracy over each
    group of samples, so it is reproduced here from our already-separate
    ``D_r_test`` / ``D_f_test`` loaders. Reported in percent.

``metric`` (the composite)
    ``utils.py::metric_function(x, y) = x * (1 - y)`` with ``x = retain_acc``
    and ``y = forget_acc`` as fractions, logged as ``100 * metric``. ``demo.py``
    writes the identical quantity on a percentage scale as
    ``retain_acc * (100 - forget_acc) / 10000``.

    **The MIA is not a term in it.** An earlier draft of our literature review
    recorded the composite as ``ACC_r x (100 - ACC_f) x MIA``; that was wrong,
    and this docstring is the correction.

``MIA``
    ``utils.py::SVC_MIA``, wired in ``main.py`` (``do_mia`` branch) as::

        SVC_MIA(shadow_train = train_retain_loader,   # members,     label 1
                shadow_test  = train_forget_loader,   # non-members, label 0
                target_train = None,
                target_test  = test_forget_loader,
                model        = model)

    Three details that a paraphrase loses:

    * The feature is the probability the model assigns to the sample's
      **ground-truth label** (``torch.gather(prob, 1, target)``), not the max
      softmax our own MIA uses. One scalar per sample.
    * The attacker is **fit** on retain-train (member) against forget-train
      (non-member), and then **scored** on a third, disjoint set, ``D_f_test``.
      It is not "forget-train members versus forget-test non-members".
    * The reported number is ``1 - mean(predict(D_f_test))``: the fraction of
      forget-class *test* images the attacker calls non-members. Higher is
      better. A retrained model scores 100 because its forget-test confidences
      look exactly like its forget-train confidences; the original model scores
      ~0 because they look like retain-train members.

    The classifier is ``sklearn.svm.SVC(C=3, gamma='auto', kernel='rbf')``.

One porting difference is unavoidable and is handled here rather than silently:
their ``collect_prob`` calls ``torch.exp`` on the model output because their
networks end in ``log_softmax``. Our :class:`~medus_class.models.ResNetCIFAR`
returns raw logits, so :func:`true_class_confidence` applies ``softmax``. The
resulting probabilities are the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from torch.utils.data import DataLoader

#: The attacker, exactly as ``utils.py::SVC_fit_predict`` constructs it.
ANCHOR_SVC_KWARGS: dict[str, Any] = {"C": 3, "gamma": "auto", "kernel": "rbf"}


def anchor_composite(retain_accuracy: float, forget_accuracy: float) -> float:
    """The anchor's composite score, ``ACC_r x (1 - ACC_f)``, in percent.

    Both inputs are fractions in ``[0, 1]``; the return is on a 0-100 scale to
    match the ``100 * metric`` their logging reports.

    Source: ``utils.py::metric_function``. Deliberately a two-term product --
    the MIA does not appear in it.

    Raises
    ------
    ValueError
        If either accuracy is outside ``[0, 1]``. Passing percentages here is
        the easy mistake, and it would silently produce a number ~100x too large
        rather than an error.
    """
    for name, value in (("retain_accuracy", retain_accuracy),
                        ("forget_accuracy", forget_accuracy)):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be a fraction in [0, 1], got {value}. This "
                f"function takes fractions and returns percent."
            )
    return 100.0 * float(retain_accuracy) * (1.0 - float(forget_accuracy))


@torch.no_grad()
def true_class_confidence(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> np.ndarray:
    """``p(y_true | x)`` for every sample in ``loader``, in loader order.

    The anchor's membership feature. Equivalent to their
    ``torch.gather(collect_prob(...), 1, labels[:, None])``, with ``softmax``
    standing in for their ``exp`` because our model emits logits rather than
    log-probabilities -- see the module docstring.

    Returns a 1-D ``float64`` array, which is what
    :func:`svc_membership_accuracy` expects.
    """
    was_training = model.training
    model.eval()
    chunks: list[np.ndarray] = []
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = F.softmax(model(images), dim=1)
            picked = probabilities.gather(1, labels[:, None]).squeeze(1)
            chunks.append(picked.detach().cpu().numpy().astype(np.float64))
    finally:
        model.train(was_training)

    if not chunks:
        raise ValueError("cannot collect confidences over an empty loader")
    return np.concatenate(chunks)


@dataclass
class AnchorMiaResult:
    """Outcome of the anchor's SVC membership attack.

    Attributes
    ----------
    mia:
        The reported number, in percent: the fraction of ``target_nonmember``
        samples the attacker labels non-member. Higher is better; a retrained
        reference should approach 100.
    n_shadow_member / n_shadow_nonmember:
        Sizes actually used to fit the attacker, **after** any subsampling.
    subsampled:
        ``True`` if ``max_shadow_per_group`` reduced either shadow group. The
        anchor does no subsampling; when this is ``True`` the number is a
        deviation from their protocol and must be reported as one.
    """

    mia: float
    n_shadow_member: int
    n_shadow_nonmember: int
    n_target_nonmember: int
    subsampled: bool
    seed: int | None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_mia": self.mia,
            "n_shadow_member": self.n_shadow_member,
            "n_shadow_nonmember": self.n_shadow_nonmember,
            "n_target_nonmember": self.n_target_nonmember,
            "subsampled": self.subsampled,
            "seed": self.seed,
            **self.notes,
        }


def svc_membership_accuracy(
    shadow_member: np.ndarray,
    shadow_nonmember: np.ndarray,
    target_nonmember: np.ndarray,
    max_shadow_per_group: int | None = None,
    seed: int | None = 42,
) -> AnchorMiaResult:
    """Fit the anchor's SVC attacker and score it on ``target_nonmember``.

    Reproduces ``utils.py::SVC_fit_predict`` for the call the anchor actually
    makes -- ``target_train=None``, so only the non-member branch contributes
    and the returned mean is over a single term:

    .. math::

        \\mathrm{MIA} = 1 - \\frac{1}{|T|}\\sum_{x \\in T} \\hat{y}(x)

    where ``T`` is ``target_nonmember`` (``D_f_test``) and ``\\hat{y}`` is the
    fitted SVC's 0/1 prediction.

    Parameters
    ----------
    shadow_member:
        Confidences on ``D_r_train``. Labelled 1.
    shadow_nonmember:
        Confidences on ``D_f_train``. Labelled 0.
    target_nonmember:
        Confidences on ``D_f_test``. The set the score is read off.
    max_shadow_per_group:
        Cap on each shadow group before fitting. ``None`` (default) reproduces
        the anchor exactly. An RBF SVC is O(n^2) in the fit, and the faithful
        CIFAR-10 call passes 45 000 + 5 000 one-dimensional points, so a cap is
        provided for machines where that is not affordable -- but any run that
        uses it is no longer the anchor's protocol and
        :attr:`AnchorMiaResult.subsampled` records that.
    seed:
        Seeds the subsample only. The SVC fit itself is deterministic.

    Raises
    ------
    ValueError
        If any of the three groups is empty. A one-class fit or an empty score
        set would return a meaningless number rather than fail.
    """
    member = np.asarray(shadow_member, dtype=np.float64).reshape(-1)
    nonmember = np.asarray(shadow_nonmember, dtype=np.float64).reshape(-1)
    target = np.asarray(target_nonmember, dtype=np.float64).reshape(-1)

    for name, array in (("shadow_member", member),
                        ("shadow_nonmember", nonmember),
                        ("target_nonmember", target)):
        if array.size == 0:
            raise ValueError(f"{name} is empty; the anchor MIA needs all three groups")

    subsampled = False
    if max_shadow_per_group is not None:
        rng = np.random.RandomState(seed)
        if member.size > max_shadow_per_group:
            member = member[rng.choice(member.size, max_shadow_per_group, replace=False)]
            subsampled = True
        if nonmember.size > max_shadow_per_group:
            nonmember = nonmember[
                rng.choice(nonmember.size, max_shadow_per_group, replace=False)
            ]
            subsampled = True

    features = np.concatenate([member, nonmember]).reshape(-1, 1)
    labels = np.concatenate([np.ones(member.size), np.zeros(nonmember.size)])

    classifier = SVC(**ANCHOR_SVC_KWARGS)
    classifier.fit(features, labels)

    predicted = classifier.predict(target.reshape(-1, 1))
    mia = 100.0 * float(1.0 - predicted.mean())

    return AnchorMiaResult(
        mia=mia,
        n_shadow_member=int(member.size),
        n_shadow_nonmember=int(nonmember.size),
        n_target_nonmember=int(target.size),
        subsampled=subsampled,
        seed=seed if subsampled else None,
    )


def anchor_mia(
    model: nn.Module,
    retain_train_loader: DataLoader,
    forget_train_loader: DataLoader,
    forget_test_loader: DataLoader,
    device: str = "cpu",
    max_shadow_per_group: int | None = None,
    seed: int | None = 42,
) -> AnchorMiaResult:
    """The anchor's MIA end to end, wired as ``main.py``'s ``do_mia`` branch.

    The loader roles are not interchangeable, so they are named rather than
    positional in intent: ``retain_train_loader`` supplies members,
    ``forget_train_loader`` supplies non-members, and the score is read off
    ``forget_test_loader``, which the attacker has never seen.

    All three must be clean, unshuffled evaluation loaders. Augmented loaders
    would make the confidences depend on the random crop drawn.
    """
    return svc_membership_accuracy(
        shadow_member=true_class_confidence(model, retain_train_loader, device),
        shadow_nonmember=true_class_confidence(model, forget_train_loader, device),
        target_nonmember=true_class_confidence(model, forget_test_loader, device),
        max_shadow_per_group=max_shadow_per_group,
        seed=seed,
    )


@dataclass
class AnchorMetrics:
    """The three numbers a row of the anchor's Table 1 holds, plus provenance.

    Accuracies and the composite are in **percent**, matching the paper's
    tables. ``mia`` is ``None`` when the attack was not run.
    """

    acc_r: float
    acc_f: float
    composite: float
    mia: float | None = None
    mia_detail: AnchorMiaResult | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "anchor_ACC_r": self.acc_r,
            "anchor_ACC_f": self.acc_f,
            "anchor_composite": self.composite,
            "anchor_MIA": self.mia,
        }
        if self.mia_detail is not None:
            detail = self.mia_detail.to_dict()
            detail.pop("anchor_mia", None)
            row.update(detail)
        return row


def anchor_metrics_from_accuracies(
    retain_test_accuracy: float,
    forget_test_accuracy: float,
    mia_result: AnchorMiaResult | None = None,
) -> AnchorMetrics:
    """Assemble :class:`AnchorMetrics` from fractional test accuracies.

    Kept separate from any model or loader so the arithmetic is testable on its
    own, and so a row can be built for a candidate whose accuracies were already
    measured elsewhere (the Pareto front CSV holds them) without a second pass
    over CIFAR-10.
    """
    return AnchorMetrics(
        acc_r=100.0 * float(retain_test_accuracy),
        acc_f=100.0 * float(forget_test_accuracy),
        composite=anchor_composite(retain_test_accuracy, forget_test_accuracy),
        mia=None if mia_result is None else mia_result.mia,
        mia_detail=mia_result,
    )


__all__ = [
    "ANCHOR_SVC_KWARGS",
    "AnchorMetrics",
    "AnchorMiaResult",
    "anchor_composite",
    "anchor_metrics_from_accuracies",
    "anchor_mia",
    "svc_membership_accuracy",
    "true_class_confidence",
]
