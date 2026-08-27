"""The anchor protocol must be the anchor's protocol, not a paraphrase of it.

Every assertion here pins a formula against the released source of

    Kodge, Saha & Roy, "Deep Unlearning: Fast and Efficient Gradient-free Class
    Forgetting", TMLR 07/2024 -- https://github.com/sangamesh-kodge/class_forgetting

rather than against the paper's prose. That distinction is the whole point: our
own literature review first recorded the composite score as
``ACC_r x (100 - ACC_f) x MIA``, which their code does not compute, and only
reading ``utils.py`` caught it. A test suite that encoded the prose would have
passed on the wrong metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation.anchor import (  # noqa: E402
    ANCHOR_SVC_KWARGS,
    anchor_composite,
    anchor_metrics_from_accuracies,
    svc_membership_accuracy,
    true_class_confidence,
)


# --- the composite ------------------------------------------------------------


def test_composite_is_the_two_term_product_from_their_code():
    """``metric_function(x, y) = x * (1 - y)``, logged as ``100 * metric``."""
    assert anchor_composite(0.9481, 0.0) == pytest.approx(94.81, abs=1e-9)
    assert anchor_composite(0.9419, 0.0003) == pytest.approx(
        100 * 0.9419 * (1 - 0.0003), abs=1e-9
    )


def test_composite_matches_the_percentage_form_used_in_their_demo():
    """``demo.py`` writes ``retain_acc*(100-forget_acc)/10000`` on a % scale."""
    retain_pct, forget_pct = 92.52, 2.70
    assert anchor_composite(retain_pct / 100, forget_pct / 100) == pytest.approx(
        100 * retain_pct * (100 - forget_pct) / 10000, abs=1e-9
    )


def test_composite_does_not_involve_the_mia():
    """A regression guard on the error the literature review actually made.

    If someone reintroduces an MIA factor, the composite for a model with a
    perfect retain score and perfect forgetting stops being 100.
    """
    assert anchor_composite(1.0, 0.0) == pytest.approx(100.0)


@pytest.mark.parametrize("retain, forget", [(94.81, 0.0), (0.9, 2.7), (-0.1, 0.5)])
def test_composite_rejects_percentages_and_out_of_range_values(retain, forget):
    with pytest.raises(ValueError):
        anchor_composite(retain, forget)


def test_metrics_assembly_reports_percentages():
    metrics = anchor_metrics_from_accuracies(
        retain_test_accuracy=0.9252, forget_test_accuracy=0.083
    )
    assert metrics.acc_r == pytest.approx(92.52)
    assert metrics.acc_f == pytest.approx(8.30)
    assert metrics.composite == pytest.approx(100 * 0.9252 * (1 - 0.083))
    assert metrics.mia is None


# --- the membership attack ----------------------------------------------------


def test_attacker_is_the_svc_they_configured():
    assert ANCHOR_SVC_KWARGS == {"C": 3, "gamma": "auto", "kernel": "rbf"}


def test_mia_is_100_when_forget_test_looks_like_a_non_member():
    """The retrained-reference case.

    A model that never saw the forget class scores it as badly on ``D_f_test``
    as on ``D_f_train``, so the attacker -- fit to call low confidence
    "non-member" -- calls every forget-test sample a non-member.
    """
    result = svc_membership_accuracy(
        shadow_member=np.full(400, 0.99),      # D_r_train: memorised
        shadow_nonmember=np.full(200, 0.01),   # D_f_train: never seen
        target_nonmember=np.full(100, 0.01),   # D_f_test: also never seen
    )
    assert result.mia == pytest.approx(100.0)
    assert result.subsampled is False


def test_mia_is_0_when_forget_test_still_looks_like_a_member():
    """The original-model case: the anchor reports ~0.03 for it."""
    result = svc_membership_accuracy(
        shadow_member=np.full(400, 0.99),
        shadow_nonmember=np.full(200, 0.01),
        target_nonmember=np.full(100, 0.98),
    )
    assert result.mia == pytest.approx(0.0)


def test_mia_reads_off_the_target_set_not_the_shadow_sets():
    """Same attacker, different target: the score must follow the target.

    Guards the wiring detail that a paraphrase loses -- the SVC is fit on
    retain-train against forget-train and then scored on a third, disjoint set.
    """
    shared = {"shadow_member": np.full(300, 0.95),
              "shadow_nonmember": np.full(300, 0.05)}
    member_like = svc_membership_accuracy(target_nonmember=np.full(50, 0.95), **shared)
    nonmember_like = svc_membership_accuracy(target_nonmember=np.full(50, 0.05), **shared)
    assert nonmember_like.mia > member_like.mia


def test_subsampling_is_recorded_as_a_deviation():
    result = svc_membership_accuracy(
        shadow_member=np.random.RandomState(0).uniform(0.8, 1.0, 5000),
        shadow_nonmember=np.random.RandomState(1).uniform(0.0, 0.2, 5000),
        target_nonmember=np.full(100, 0.05),
        max_shadow_per_group=500,
    )
    assert result.subsampled is True
    assert result.n_shadow_member == 500
    assert result.n_shadow_nonmember == 500
    assert result.seed == 42


@pytest.mark.parametrize("empty", ["shadow_member", "shadow_nonmember",
                                   "target_nonmember"])
def test_mia_refuses_an_empty_group(empty):
    groups = {"shadow_member": np.full(10, 0.9),
              "shadow_nonmember": np.full(10, 0.1),
              "target_nonmember": np.full(10, 0.1)}
    groups[empty] = np.array([])
    with pytest.raises(ValueError):
        svc_membership_accuracy(**groups)


# --- the membership feature ---------------------------------------------------


class FixedLogits(nn.Module):
    """Returns a constant logit vector, so the softmax is known in closed form."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(x.size(0), -1)


def test_confidence_is_the_true_class_probability_not_the_max():
    """Their feature is ``gather(prob, 1, target)``, which is not ``max``.

    With label 0 and the mass on class 2, the true-class probability is small
    while the max is large. Reading the max here would silently substitute our
    own MIA's feature for theirs.
    """
    logits = torch.tensor([0.0, 0.0, 5.0])
    model = FixedLogits(logits)
    expected = torch.softmax(logits, dim=0)[0].item()

    loader = DataLoader(
        TensorDataset(torch.zeros(6, 3), torch.zeros(6, dtype=torch.long)),
        batch_size=4, shuffle=False,
    )
    confidences = true_class_confidence(model, loader)

    assert confidences.shape == (6,)
    assert confidences == pytest.approx(expected)
    assert expected < 0.05  # i.e. nowhere near the max softmax


def test_confidence_restores_training_mode():
    """Evaluation must not leave the model switched out of train()."""
    model = FixedLogits(torch.zeros(3))
    model.train()
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
        batch_size=2, shuffle=False,
    )
    true_class_confidence(model, loader)
    assert model.training is True
