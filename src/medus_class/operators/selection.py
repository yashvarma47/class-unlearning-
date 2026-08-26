"""Which connections should a gradient-free operator touch?

Every operator in the gradient-free family acts on "selected connections". This
module decides which. It is the part of the new library that makes unlearning
*targeted* rather than merely destructive, and it is therefore the component the
whole direction depends on.

Why selection has to consult D_f
--------------------------------
The full-fidelity re-evaluation of the 50 x 100 search measured selectivity

    S = (forget-loss gained) / (retain-loss paid)

at **1.002 median, 1.047 best** across the front: damage was exactly
indiscriminate. An operator that chooses its targets from the weights alone --
by magnitude, by sign, at random -- has no mechanism to distinguish D_f from
D_r, so it can only degrade the network generically and S ~ 1 becomes
structural rather than merely observed. The supervisor confirmed the
requirement directly: *"the connections to change must be chosen using the
forget set D_f; otherwise, the operators will damage the forgotten and retained
data equally."*

How this stays gradient-free
----------------------------
Importance is estimated from **forward passes only**. No loss is formed, no
``backward()`` is called, and ``torch.no_grad()` wraps the whole measurement.
The criterion is the activation-aware one used by Wanda:

.. math::

    \\mathrm{importance}(W_{ij}) = |W_{ij}| \\cdot \\|a_j\\|_2

i.e. a weight matters if it is large *and* the input it reads is active. That
is computed separately over D_f and D_r, and the two are subtracted:

.. math::

    \\mathrm{score}(W_{ij}) = |W_{ij}| \\cdot (\\hat a^{f}_j - \\hat a^{r}_j)

Connections scoring high are those the forget set drives and the retain set does
not. This is the same principle DAMP uses -- identify forget-specific structure
in closed form, then edit the weights -- without ever computing a derivative.

Activation norms are root-mean-square rather than raw sums, so D_f and D_r
contribute on the same scale even though the loaders hold very different
numbers of samples (512 against 45 000 in the search configuration).

Alternatives are kept for comparison
------------------------------------
``magnitude`` and ``random`` selection are implemented alongside so the claim
"forget-informed selection is what buys selectivity" can be *tested* rather than
asserted. That comparison is the natural ablation for the dissertation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import torch
import torch.nn as nn

from medus_class.models.layer_groups import LayerGroup
from medus_class.utils.seeding import derive_seed

#: Selection rules. ``class_contrast`` is the production rule; the other two are
#: DATA-FREE ABLATIONS and exist so the contribution of forget-informed
#: selection can be measured rather than assumed. PRUNE and RANDOM_PRUNE pin
#: themselves to ``magnitude`` and ``random`` respectively for that reason.
SelectionRule = Literal["class_contrast", "magnitude", "random"]

SELECTION_RULES: tuple[str, ...] = ("class_contrast", "magnitude", "random")

#: Module types that own a 2-D-or-more weight worth selecting within. BatchNorm
#: is excluded deliberately: its parameters are per-channel scales and shifts,
#: so "connections" is not a meaningful unit there, and editing them is a
#: blunter intervention than any operator in this family intends.
SELECTABLE_MODULES = (nn.Conv2d, nn.Linear)


@dataclass
class ConnectionMask:
    """Which coordinates of which parameters an operator may modify.

    Keyed by fully-qualified parameter name, matching ``LayerGroup``'s
    ``parameter_names``, so an operator can look up the mask for each tensor it
    is about to write to. A parameter absent from the mapping is not selected at
    all and must be left untouched.
    """

    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    rule: str = "class_contrast"
    ratio: float = 0.0

    @property
    def n_selected(self) -> int:
        return int(sum(int(m.sum()) for m in self.masks.values()))

    @property
    def n_total(self) -> int:
        return int(sum(m.numel() for m in self.masks.values()))

    def for_parameter(self, name: str) -> torch.Tensor | None:
        return self.masks.get(name)

    def summary(self) -> dict[str, Any]:
        total = self.n_total
        return {
            "rule": self.rule,
            "ratio": round(self.ratio, 4),
            "selected": self.n_selected,
            "total": total,
            "fraction": round(self.n_selected / total, 5) if total else 0.0,
        }


def _target_modules(
    model: nn.Module, group: LayerGroup
) -> list[tuple[str, nn.Module]]:
    """The Conv2d/Linear modules inside ``group``, in model order."""
    wanted = set(group.module_names)
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, SELECTABLE_MODULES):
            continue
        if any(name == w or name.startswith(f"{w}.") for w in wanted):
            found.append((name, module))
    return found


@torch.no_grad()
def _input_activation_norms(
    model: nn.Module,
    modules: list[tuple[str, nn.Module]],
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> dict[str, torch.Tensor]:
    """Root-mean-square input activation per input channel/feature.

    Forward hooks only -- the model is never asked for a gradient. Returns one
    vector per module, of length ``in_channels`` (Conv2d) or ``in_features``
    (Linear).

    RMS rather than a sum so that loaders of very different size are directly
    comparable: D_f holds 512 samples in the search configuration and D_r holds
    45 000, and an unnormalised sum would make every retain score larger purely
    by virtue of sample count.
    """
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            if x.dim() == 4:                      # Conv2d: (N, C, H, W)
                squared = x.pow(2).sum(dim=(0, 2, 3))
                n = x.shape[0] * x.shape[2] * x.shape[3]
            else:                                 # Linear: (N, F)
                squared = x.pow(2).sum(dim=0)
                n = x.shape[0]
            if name in sums:
                sums[name] += squared.float()
                counts[name] += n
            else:
                sums[name] = squared.float()
                counts[name] = n
        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()  # freeze BatchNorm statistics: measuring must not modify
    try:
        for images, _labels in batches:
            model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {
        name: torch.sqrt(sums[name] / max(counts[name], 1))
        for name in sums
    }


def _broadcast_to_weight(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Line the per-input-unit activation vector up with the weight tensor."""
    if weight.dim() == 4:                       # Conv2d (out, in, kh, kw)
        return activation.view(1, -1, 1, 1).expand_as(weight)
    if weight.dim() == 2:                       # Linear (out, in)
        return activation.view(1, -1).expand_as(weight)
    raise ValueError(f"unexpected weight shape {tuple(weight.shape)}")


def _take_batches(loader, cap: int | None):
    """At most ``cap`` batches. Selection is a measurement, not training."""
    for index, batch in enumerate(loader):
        if cap is not None and index >= cap:
            return
        yield batch


@torch.no_grad()
def select_connections(
    model: nn.Module,
    context: Any,
    group: LayerGroup,
    ratio: float,
    rule: SelectionRule = "class_contrast",
    largest: bool = True,
) -> ConnectionMask:
    """Choose the ``ratio`` fraction of connections in ``group`` to modify.

    Parameters
    ----------
    ratio:
        Fraction of the group's selectable weights to mark, in ``[0, 1]``.
        Comes from the chromosome's intensity level via the lookup table.
    largest:
        Take the highest-scoring connections (default) or the lowest. ``PRUNE``
        is defined as switching off *weak* connections, so it passes ``False``
        with the ``magnitude`` rule; everything else takes the top.
    rule:
        ``class_contrast`` -- highest ``|W| * (rms_f - rms_r)``: connections
        responding more to the forget class than to the retained classes. The
        production rule, and the only one that can be selective.
        ``magnitude`` -- largest ``|W|``. Data-free ablation.
        ``random`` -- uniform. The null model.

    Notes
    -----
    Bias vectors and BatchNorm parameters are never selected: "connection" is a
    weight-matrix notion, and the operators in this family are defined on it.
    A group with no Conv2d or Linear module therefore yields an empty mask, and
    operators must treat that as a no-op rather than an error.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"selection ratio must be in [0, 1], got {ratio}")

    modules = _target_modules(model, group)
    masks: dict[str, torch.Tensor] = {}
    if not modules or ratio <= 0.0:
        return ConnectionMask(masks=masks, rule=rule, ratio=ratio)

    scores: dict[str, torch.Tensor] = {}

    if rule == "class_contrast":
        cap = getattr(context, "batch_cap", None)
        forget = _input_activation_norms(
            model, modules, _take_batches(context.loaders.forget_eval, cap),
            context.device,
        )
        retain = (
            _input_activation_norms(
                model, modules, _take_batches(context.loaders.retain_eval, cap),
                context.device,
            )
        )
        for name, module in modules:
            weight = module.weight.detach()
            a_f = forget.get(name)
            if a_f is None:                       # module never ran; skip it
                continue
            score = weight.abs() * _broadcast_to_weight(a_f, weight)
            a_r = retain.get(name)
            if a_r is not None:
                score = score - weight.abs() * _broadcast_to_weight(a_r, weight)
            scores[f"{name}.weight"] = score

    elif rule == "magnitude":
        for name, module in modules:
            scores[f"{name}.weight"] = module.weight.detach().abs()

    elif rule == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(context.seed, "select", group.name))
        for name, module in modules:
            weight = module.weight.detach()
            scores[f"{name}.weight"] = torch.rand(
                weight.shape, generator=generator
            ).to(weight.device)

    else:
        raise ValueError(f"unknown selection rule {rule!r}; expected one of {SELECTION_RULES}")

    if not scores:
        return ConnectionMask(masks=masks, rule=rule, ratio=ratio)

    # One global threshold across the whole group rather than per-module
    # quantiles: a per-module threshold would force every module to give up the
    # same fraction, which contradicts the point of the exercise. If the forget
    # set is concentrated in one convolution, that convolution should supply
    # most of the selected connections.
    flat = torch.cat([s.flatten() for s in scores.values()])
    k = int(round(ratio * flat.numel()))
    if k <= 0:
        return ConnectionMask(masks={n: torch.zeros_like(s, dtype=torch.bool)
                                     for n, s in scores.items()},
                              rule=rule, ratio=ratio)
    k = min(k, flat.numel())
    threshold = torch.topk(flat, k, largest=largest, sorted=True).values[-1]

    for name, score in scores.items():
        chosen = score >= threshold if largest else score <= threshold
        # Protected channels carry -inf and must stay unselected even when the
        # requested ratio exceeds the number of eligible channels.
        masks[name] = chosen & torch.isfinite(score)

    # topk ties can select slightly more than k; that is harmless and preferable
    # to breaking ties arbitrarily, which would make the mask depend on memory
    # layout rather than on the data.
    return ConnectionMask(masks=masks, rule=rule, ratio=ratio)
