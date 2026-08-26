"""The three objectives, and the diagnostics recorded beside them.

.. math::

    f_1 = \\mathrm{JS}\\big(P_{ref}(D_f)\\;\\|\\;P(D_f)\\big)
    \\qquad
    f_2 = L_r
    \\qquad
    f_3 = \\frac{\\lVert\\theta - \\theta_0\\rVert_2}{\\lVert\\theta_0\\rVert_2}

All three are minimised. A candidate scoring ``(0, L_r^{ref}, small)`` is
indistinguishable from the retrained reference on the forget class, has paid
nothing on retain, and got there with a small edit.

Both departures from the predecessor project are deliberate.

**f1 matches a distribution, not a loss.** The instance-level project's final
objective was ``|L_f - L_f(reference)|``, a scalar target on the cross-entropy.
Cross-entropy on ``D_f`` is a one-number summary of a ten-dimensional
distribution, and many very different models share a value: a model that
confidently relabels every frog as "cat" can hit the reference's loss exactly
while behaving nothing like it. Under class unlearning that slack is the whole
question, so ``f1`` compares the predictive distributions directly.

**f3 is an edit cost, not a second reference term.** That project used
``f2 = L_r`` together with ``f3 = KL(P_ref(D_f) || P(D_f))``, and the two behaved
as near-duplicates -- both punish a damaged model, so a nominally
three-objective search was really a two-objective one wearing three labels. A
measured rank correlation against ``f2`` on random candidates: ``+0.74`` for that
KL, against ``+0.36`` for the parameter-change norm used here. Edit cost never
reads the data, so it is orthogonal by construction, and among candidates with
equal losses it prefers the one that needed less surgery -- which is also what
makes a result defensible as *unlearning* rather than as retraining by another
route.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#: Upper bound of the Jensen-Shannon divergence in nats, attained only when the
#: two distributions have disjoint support.
JS_MAX_NATS = math.log(2.0)


@torch.no_grad()
def reference_logits(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> torch.Tensor:
    """Cache a fixed model's logits over a loader, in loader order.

    The reference never changes, so its outputs are computed once when the
    evaluator is built and reused by every candidate. The loader must be
    deterministic and unshuffled -- ``forget_eval`` is -- or the cached rows
    would not line up with the candidate's.
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
def js_to_reference(
    model: nn.Module,
    loader: DataLoader,
    cached_reference_logits: torch.Tensor,
    device: str = "cpu",
) -> float:
    """``f1`` -- Jensen-Shannon divergence from the reference on ``D_f``.

    .. math::

        \\mathrm{JS}(P \\| Q)
            = \\tfrac{1}{2}\\mathrm{KL}(P \\| M)
            + \\tfrac{1}{2}\\mathrm{KL}(Q \\| M),
        \\qquad M = \\tfrac{1}{2}(P + Q)

    Three properties motivate JS over the KL the predecessor used:

    * **Symmetric.** ``KL(P_ref || P)`` punishes a candidate for being unsure
      where the reference is confident but barely notices the reverse. We want
      the two distributions to *agree*, so neither one-sided direction is right.
    * **Bounded** by ``ln 2``. The forward KL is unbounded, and one destroyed
      candidate can produce a value large enough to flatten every other
      individual's ``f1`` to nothing under min-max normalisation.
    * **Finite everywhere.** ``KL(P_ref || P)`` diverges as the candidate's
      probability on a reference-supported class approaches zero; the mixture
      ``M`` always covers both.

    Computed from log-softmax with ``logaddexp`` rather than by adding
    probabilities and taking a log, which loses precision in the tail where this
    objective is most sensitive.
    """
    was_training = model.training
    model.eval()

    total, seen = 0.0, 0
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            candidate = model(images)

            reference_slice = cached_reference_logits[seen : seen + images.size(0)]
            if reference_slice.size(0) != images.size(0):
                raise ValueError(
                    f"cached reference logits exhausted at sample {seen}: the "
                    f"cache holds {cached_reference_logits.size(0)} rows but the "
                    f"loader yielded more."
                )

            log_p = F.log_softmax(reference_slice.to(candidate.device), dim=1)
            log_q = F.log_softmax(candidate, dim=1)
            log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)

            p, q = log_p.exp(), log_q.exp()
            kl_p_m = (p * (log_p - log_m)).sum(dim=1)
            kl_q_m = (q * (log_q - log_m)).sum(dim=1)
            total += float((0.5 * kl_p_m + 0.5 * kl_q_m).sum().item())
            seen += images.size(0)
    finally:
        model.train(was_training)

    if seen == 0:
        raise ValueError("cannot compute JS divergence over an empty loader")
    if seen != cached_reference_logits.size(0):
        raise ValueError(
            f"loader yielded {seen} samples but the reference cache holds "
            f"{cached_reference_logits.size(0)}; the two models would be "
            f"compared on different data"
        )
    return total / seen


@torch.no_grad()
def kl_to_reference(
    model: nn.Module,
    loader: DataLoader,
    cached_reference_logits: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Forward ``KL(P_ref || P)`` on ``D_f``. **Diagnostic only, never f1 or f3.**

    Retained so the claim that it duplicates ``f2`` can be re-measured from any
    run's own output rather than taken on trust from the predecessor project.
    """
    was_training = model.training
    model.eval()

    total, seen = 0.0, 0
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            candidate = model(images)
            reference_slice = cached_reference_logits[seen : seen + images.size(0)]

            log_p_ref = F.log_softmax(reference_slice.to(candidate.device), dim=1)
            log_p_cand = F.log_softmax(candidate, dim=1)
            total += float(
                (log_p_ref.exp() * (log_p_ref - log_p_cand)).sum().item()
            )
            seen += images.size(0)
    finally:
        model.train(was_training)

    if seen == 0:
        raise ValueError("cannot compute KL over an empty loader")
    return total / seen


@torch.no_grad()
def relative_parameter_delta(
    model: nn.Module,
    original_state: Mapping[str, torch.Tensor],
    weights_only: bool = True,
) -> float:
    """``f3`` -- how much of the network the edit moved, relative to its scale.

    A *relative* norm, not an absolute one. Absolute movement is dominated by
    whichever layer group holds the most parameters -- ``layer4`` carries roughly
    two thirds of ResNet-18 -- so an absolute cost would price a large edit to
    ``fc`` as nearly free and a small edit to ``layer4`` as expensive, which is a
    statement about the architecture rather than about the strategy.

    Parameters
    ----------
    weights_only:
        Restrict to ``*.weight`` tensors (default). BatchNorm running statistics
        are buffers, not edits, and counting them would charge a strategy for the
        model merely having seen data.
    """
    moved_sq = 0.0
    base_sq = 0.0
    current = model.state_dict()

    for name, original in original_state.items():
        if weights_only and not name.endswith("weight"):
            continue
        if not torch.is_floating_point(original) or name not in current:
            continue
        theta = current[name].detach().to(original.device, torch.float64)
        theta_0 = original.detach().to(torch.float64)
        moved_sq += float((theta - theta_0).pow(2).sum().item())
        base_sq += float(theta_0.pow(2).sum().item())

    if base_sq <= 0.0:
        raise ValueError("original parameters have zero norm; cannot normalise")
    return math.sqrt(moved_sq) / math.sqrt(base_sq)


def selectivity(
    forget_loss: float,
    retain_loss: float,
    original_forget_loss: float,
    original_retain_loss: float,
) -> float:
    """``S = (forget loss gained) / (retain loss paid)``.

    Carried over unchanged from the instance-level project so the two settings
    stay directly comparable. There, ``S`` never exceeded **1.158** across 10 534
    evaluated strategies, against roughly **932** for retraining from scratch --
    damage was almost exactly indiscriminate.

    Returns ``inf`` when a candidate raises the forget loss while paying nothing
    measurable on retain. That is the outcome the project is looking for, and it
    would otherwise be a division by zero rather than a failure.
    """
    gained = forget_loss - original_forget_loss
    paid = retain_loss - original_retain_loss
    if abs(paid) < 1e-9:
        return float("inf") if gained > 0 else float("nan")
    return gained / paid


__all__ = [
    "JS_MAX_NATS",
    "reference_logits",
    "js_to_reference",
    "kl_to_reference",
    "relative_parameter_delta",
    "selectivity",
]
