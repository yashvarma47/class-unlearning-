"""The objectives must mean what their names say.

The predecessor project's central lesson was that an objective can look
reasonable, run without error, and still not measure the thing it is named
after -- its f3 was a second copy of f2 for four rounds of experiments before
anyone measured the correlation. These tests pin the properties each objective
is relied upon for.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.evaluation.objectives import (  # noqa: E402
    JS_MAX_NATS,
    js_to_reference,
    kl_to_reference,
    reference_logits,
    relative_parameter_delta,
    selectivity,
)


class TinyNet(nn.Module):
    """Smallest thing with a weight and a forward pass."""

    def __init__(self, out_features: int = 4, scale: float = 1.0) -> None:
        super().__init__()
        self.fc = nn.Linear(8, out_features)
        with torch.no_grad():
            self.fc.weight.mul_(scale)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def loader():
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


# --- f1: Jensen-Shannon ---------------------------------------------------

def test_js_is_zero_against_itself(loader):
    """A model identical to the reference must score exactly 0."""
    model = TinyNet()
    cached = reference_logits(model, loader)
    assert js_to_reference(model, loader, cached) == pytest.approx(0.0, abs=1e-9)


def test_js_is_positive_for_a_different_model(loader):
    torch.manual_seed(1)
    reference, candidate = TinyNet(), TinyNet()
    cached = reference_logits(reference, loader)
    assert js_to_reference(candidate, loader, cached) > 0.0


def test_js_is_bounded_by_ln2(loader):
    """The bound is why one destroyed candidate cannot flatten a generation."""
    reference = TinyNet(scale=1.0)
    # Wildly different logits -> near-disjoint predictive distributions.
    extreme = TinyNet(scale=500.0)
    cached = reference_logits(reference, loader)

    value = js_to_reference(extreme, loader, cached)
    assert 0.0 <= value <= JS_MAX_NATS + 1e-9
    assert JS_MAX_NATS == pytest.approx(math.log(2.0))


def test_js_is_symmetric(loader):
    """Unlike the KL, swapping the two models must not change the value."""
    torch.manual_seed(2)
    a, b = TinyNet(), TinyNet()

    forward = js_to_reference(b, loader, reference_logits(a, loader))
    backward = js_to_reference(a, loader, reference_logits(b, loader))
    assert forward == pytest.approx(backward, rel=1e-6)


def test_js_stays_finite_where_kl_explodes(loader):
    """A confidently-wrong candidate must produce a usable f1, not an infinity."""
    reference = TinyNet(scale=1.0)
    confident = TinyNet(scale=2000.0)
    cached = reference_logits(reference, loader)

    js = js_to_reference(confident, loader, cached)
    kl = kl_to_reference(confident, loader, cached)
    assert math.isfinite(js)
    assert js <= JS_MAX_NATS + 1e-9
    # The KL is what JS was chosen over; it is free to be enormous.
    assert kl > js


def test_js_rejects_a_mismatched_cache(loader):
    """Comparing two models on different data must fail loudly."""
    model = TinyNet()
    cached = reference_logits(model, loader)[:4]      # too few rows
    with pytest.raises(ValueError):
        js_to_reference(model, loader, cached)


# --- f3: edit cost ---------------------------------------------------------

def test_edit_cost_is_zero_for_an_untouched_model():
    model = TinyNet()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert relative_parameter_delta(model, state) == pytest.approx(0.0, abs=1e-12)


def test_edit_cost_grows_with_the_edit():
    model = TinyNet()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    with torch.no_grad():
        model.fc.weight.mul_(1.01)
    small = relative_parameter_delta(model, state)

    with torch.no_grad():
        model.fc.weight.mul_(1.50)
    large = relative_parameter_delta(model, state)

    assert 0.0 < small < large


def test_edit_cost_is_scale_invariant():
    """Relative, not absolute: doubling every weight and doubling the edit with
    it must give the same cost. This is what stops the biggest layer group from
    dominating the objective."""
    small = TinyNet(scale=1.0)
    big = TinyNet(scale=100.0)
    big.load_state_dict({k: v * 100.0 for k, v in small.state_dict().items()})

    small_state = {k: v.detach().clone() for k, v in small.state_dict().items()}
    big_state = {k: v.detach().clone() for k, v in big.state_dict().items()}

    with torch.no_grad():
        small.fc.weight.mul_(1.1)
        big.fc.weight.mul_(1.1)

    assert relative_parameter_delta(small, small_state) == pytest.approx(
        relative_parameter_delta(big, big_state), rel=1e-6
    )


def test_edit_cost_ignores_buffers_by_default():
    """BatchNorm running statistics are not edits and must not be charged for."""
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Move a running statistic only -- no weight touched.
    with torch.no_grad():
        model[1].running_mean.add_(5.0)

    assert relative_parameter_delta(model, state) == pytest.approx(0.0, abs=1e-12)


# --- selectivity -----------------------------------------------------------

def test_selectivity_matches_its_definition():
    # gained 2.0 on forget, paid 0.5 on retain
    assert selectivity(2.5, 1.0, 0.5, 0.5) == pytest.approx(4.0)


def test_selectivity_is_one_for_indiscriminate_damage():
    """S ~ 1 is the instance-level result this project exists to escape."""
    assert selectivity(1.5, 1.5, 0.5, 0.5) == pytest.approx(1.0)


def test_selectivity_is_infinite_for_free_forgetting():
    """Forget loss up, retain loss unmoved: the target outcome, not an error."""
    assert selectivity(2.0, 0.5, 0.5, 0.5) == math.inf


def test_selectivity_is_nan_when_nothing_happened():
    assert math.isnan(selectivity(0.5, 0.5, 0.5, 0.5))
