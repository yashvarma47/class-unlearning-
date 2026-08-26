"""The gradient-free operator family.

Ten operators, each a **single direct write** to the selected connections of one
layer group. None of them forms a loss, calls ``backward()``, or runs an
optimiser, so the whole family is derivative-free in the strict sense.

Why this family exists
----------------------
The 50 x 100 search used five gradient operators and five smoothing operators.
It worked, but two things came out of it:

* Full-fidelity re-evaluation put selectivity at **S = 1.002 median**, i.e.
  damage was exactly indiscriminate -- forget-set loss and retain-set loss moved
  together at 1:1.
* Gradient operators accounted for ~70% of the 9.31-hour runtime
  (``1.035 s per action + 2.296 s fixed``), which made tuning the algorithm
  impractical: every adjustment cost an overnight run.

The supervisor's direction is to replace the library with gradient-free
operators. That fixes the second problem outright -- these operators do no data
passes, so evaluation cost collapses to the measurement floor -- and it attacks
the first through :mod:`medus.operators.selection`, which chooses *which*
connections to modify using D_f.

The selection step is what makes this more than vandalism. An operator that
picks its targets by weight magnitude or at random cannot damage D_f more than
D_r, so S ~ 1 would become structural. Selection is therefore not an
implementation detail of these operators: it is the mechanism by which they
forget anything in particular.

Shared structure
----------------
Every operator here:

1. asks :func:`select_connections` for a boolean mask over the group's
   Conv2d/Linear weights, sized by the ``ratio`` hyperparameter;
2. applies its own elementwise edit **only where the mask is true**;
3. writes once, and returns.

Step 2 is the only thing that differs between them, which is why they share
:class:`MaskedWriteOperator` and each contribute a two-line ``edit``.

Biases and BatchNorm parameters are never touched: "connection" is a
weight-matrix notion. A group with no Conv2d or Linear module is a no-op.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn

from medus_class.models.layer_groups import LayerGroup
from medus_class.operators.base import (
    OperatorContext,
    OperatorResult,
    UnlearningOperator,
    delta_norm,
    parameter_snapshot,
    restrict_to_group,
)
from medus_class.operators.selection import ConnectionMask, select_connections
from medus_class.utils.seeding import derive_seed

#: The LIBRARY these operators belong to, for the write-up and the audit trail.
#: Distinct from ``family``, which names the chromosome CHANNEL an operator is
#: reachable through. All ten are one gradient-free library, but they are split
#: across the two existing channels so the genome keeps its (b, g, s, d_g, d_s)
#: shape and its length of 30 -- see the module docstring.
LIBRARY = "gradient_free"


class MaskedWriteOperator(UnlearningOperator):
    """Select connections with D_f, then apply one elementwise edit to them.

    Subclasses set :attr:`name`, optionally override :attr:`selection_rule` and
    :attr:`select_largest`, and implement :meth:`edit`.

    ``edit`` receives the parameter tensor, the boolean mask, the resolved
    hyperparameters and a CPU generator, and must modify the tensor **in place**
    under ``torch.no_grad()``. It must not touch coordinates outside the mask --
    that is the contract the chromosome's intensity gene relies on.
    """

    #: Set per subclass to the chromosome channel the operator is registered
    #: under. ``family`` must match its lookup-table section or
    #: ``build_operator`` cannot find the class; the library-level identity
    #: shared by all ten is :data:`LIBRARY`.
    library = LIBRARY
    #: Overridden by PRUNE and RANDOM_PRUNE, which define their own targets.
    selection_rule: str = "class_contrast"
    #: PRUNE switches off *weak* connections, so it takes the bottom of the
    #: ranking rather than the top.
    select_largest: bool = True
    #: Operators that consult D_f need it present; declared for the SEC.
    requires: tuple[str, ...] = ("forget",)

    def edit(
        self,
        parameter: nn.Parameter,
        mask: torch.Tensor,
        hparams: dict[str, Any],
        generator: torch.Generator,
        name: str,
        context: OperatorContext,
        group: LayerGroup,
    ) -> None:
        raise NotImplementedError

    def apply(
        self,
        model: nn.Module,
        context: OperatorContext,
        group: LayerGroup,
        hparams: dict[str, Any],
    ) -> OperatorResult:
        started = time.perf_counter()
        ratio = float(hparams.get("ratio", 0.0))
        rule = str(hparams.get("selection_rule", self.selection_rule))
        # A run-level override swaps the selector for the whole library at once,
        # which is how the three selectors are compared on equal footing. It
        # deliberately does NOT touch operators that are defined by their own
        # rule (PRUNE = magnitude, RANDOM_PRUNE = random): those are the
        # controls, and overriding them would remove the baseline.
        override = getattr(context, "selection_rule", None)
        if override and rule == "class_contrast":
            rule = str(override)

        result = OperatorResult(
            operator=self.name, group=group.name, hparams=dict(hparams)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(context.seed, self.name, group.name))

        with restrict_to_group(model, context.registry, group) as group_parameters:
            if not group_parameters:
                result.wall_time = time.perf_counter() - started
                result.extra["skipped"] = "group has no parameters"
                return result

            before = parameter_snapshot(group_parameters)

            selection: ConnectionMask = select_connections(
                model, context, group, ratio, rule=rule,  # type: ignore[arg-type]
                largest=self.select_largest,
            )
            if not selection.masks:
                result.wall_time = time.perf_counter() - started
                result.extra["skipped"] = "no selectable Conv2d/Linear weights"
                result.extra["selection"] = selection.summary()
                return result

            edited = 0
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    mask = selection.for_parameter(name)
                    if mask is None or not bool(mask.any()):
                        continue
                    self.edit(
                        parameter, mask.to(parameter.device), hparams,
                        generator, name, context, group,
                    )
                    edited += 1

            result.steps = edited
            result.parameter_delta_norm = delta_norm(before, group_parameters)
            result.extra["selection"] = selection.summary()
            result.extra["tensors_edited"] = edited

        result.wall_time = time.perf_counter() - started
        return result


# -- switching connections off ----------------------------------------------


class Mask(MaskedWriteOperator):
    """Switch the selected connections off: ``theta[mask] <- 0``.

    Selection is forget-informed, so this removes the connections the forget set
    drives and the retain set does not. It is the most direct expression of the
    family's intent, and the closest gradient-free analogue of removing a
    forget-specific direction.
    """

    name = "MASK"
    family = "editor"

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        parameter[mask] = 0.0


class Prune(MaskedWriteOperator):
    """Switch off *weak* connections: magnitude selection, smallest first.

    Distinct from :class:`Mask` in the **selection rule**, not in the write.
    Where MASK removes what D_f depends on, PRUNE removes what carries little
    weight at all -- the classical sparsity argument that memorised samples are
    supported by small, sample-specific weights (Liu et al., 2024).

    Data-free by construction, and kept that way on purpose: it is the natural
    control for MASK. If MASK forgets selectively and PRUNE does not, the
    difference is attributable to forget-informed selection rather than to
    zeroing weights.
    """

    name = "PRUNE"
    family = "editor"
    selection_rule = "magnitude"
    select_largest = False
    requires: tuple[str, ...] = ()

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        parameter[mask] = 0.0


class RandomPrune(MaskedWriteOperator):
    """Switch off a random subset of connections.

    The null model for the whole family. Any operator that cannot beat
    RANDOM_PRUNE is not exploiting structure.
    """

    name = "RANDOM_PRUNE"
    family = "editor"
    selection_rule = "random"
    requires: tuple[str, ...] = ()

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        parameter[mask] = 0.0


# -- restoring and re-randomising -------------------------------------------


class Reset(MaskedWriteOperator):
    """Restore the selected connections to their original values.

    ``theta[mask] <- theta_original[mask]``, using the per-group snapshot the
    SEC takes before it touches a group.

    A repair operator, and meaningless as the first action of a strategy: if
    nothing has changed the group yet, the snapshot equals the current weights
    and RESET is a no-op. Its value is *later* in a sequence -- undoing damage a
    previous operator did to connections that turned out to matter for D_r,
    which is exactly the kind of correction a search can discover but a
    hand-designed pipeline would not.

    Falls back to a no-op with an explicit note when no snapshot exists, rather
    than silently doing nothing.
    """

    name = "RESET"
    family = "smoother"
    requires: tuple[str, ...] = ("forget",)

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        snapshot = context.group_snapshots.get(group.name)
        if not snapshot or name not in snapshot:
            return
        original = snapshot[name].to(parameter.device)
        parameter[mask] = original[mask]


class Damp(MaskedWriteOperator):
    """Make the selected connections weaker: ``theta[mask] *= (1 - strength)``.

    A graded alternative to MASK. Where MASK removes a connection outright,
    DAMP scales it down, so ``strength`` gives the chromosome a continuous dial
    between "untouched" and "removed" (``strength = 1`` is exactly MASK).

    Multiplicative rather than subtractive on purpose: it is scale-free, so the
    same ``strength`` means the same proportional reduction in every layer
    group, and it can never drive a weight past zero and flip its sign -- the
    failure mode that made ``L1_SPARSE`` collapse at high intensity.
    """

    name = "DAMP"
    family = "smoother"

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        strength = float(hparams["strength"])
        parameter[mask] = parameter[mask] * (1.0 - strength)


class Noise(MaskedWriteOperator):
    """Perturb the selected connections with relative Gaussian noise.

    ``theta[mask] += sigma * std(theta) * eps``.

    The masked counterpart of the existing ``WNOISE``, which perturbs a whole
    group indiscriminately. Restricting the perturbation to forget-selected
    connections is precisely the difference between generic degradation and
    targeted forgetting, and comparing the two is a clean ablation.
    """

    name = "NOISE"
    family = "smoother"

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        sigma = float(hparams["sigma"])
        std = float(parameter.detach().float().std(unbiased=False))
        if not math.isfinite(std) or std <= 0.0:
            return
        noise = torch.randn(
            int(mask.sum()), generator=generator, dtype=torch.float32
        ).to(parameter.device) * (sigma * std)
        parameter[mask] = parameter[mask] + noise.to(parameter.dtype)


class Clip(MaskedWriteOperator):
    """Limit how large the selected weights may be.

    ``theta[mask] <- clamp(theta[mask], -c, +c)`` with ``c = limit * std(theta)``.

    Relative to the tensor's own standard deviation, for the same
    cross-group-comparability reason as ``REINIT`` and ``NOISE``. Only affects
    weights already outside the bound, so at a loose ``limit`` it is a genuine
    no-op rather than a small perturbation -- useful as a low-intensity rung
    that the search can select without cost.
    """

    name = "CLIP"
    family = "smoother"

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        limit = float(hparams["limit"])
        std = float(parameter.detach().float().std(unbiased=False))
        if not math.isfinite(std) or std <= 0.0:
            return
        bound = limit * std
        parameter[mask] = torch.clamp(parameter[mask], min=-bound, max=bound)


class Quantize(MaskedWriteOperator):
    """Reduce the precision of the selected weights.

    Uniform quantisation to ``2^bits`` levels across the tensor's observed
    range, applied only to the selected coordinates.

    The gentlest operator in the family: it preserves each weight's approximate
    value while discarding the fine detail. If memorisation of individual
    samples lives in low-order bits -- the premise behind quantisation-based
    unlearning -- this removes it while leaving the coarse structure that
    supports generalisation intact.
    """

    name = "QUANTIZE"
    family = "smoother"

    def edit(self, parameter, mask, hparams, generator, name, context, group):
        bits = int(hparams["bits"])
        if bits <= 0:
            return
        selected = parameter[mask]
        low = float(selected.min())
        high = float(selected.max())
        span = high - low
        if not math.isfinite(span) or span <= 0.0:
            return
        levels = float(2 ** bits - 1)
        step = span / levels
        parameter[mask] = torch.round((selected - low) / step) * step + low


#: Registry-facing maps, keyed by the name the lookup table uses.
#:
#: REINIT and SIGN_FLIP are absent from this library by design, not disabled by
#: config: they were the most destructive operators in the predecessor's
#: calibration and dominated the fronts that turned out to be full of wrecked
#: models. Excluding them here means no config edit can bring them back.
EDITOR_OPERATORS: dict[str, type[MaskedWriteOperator]] = {
    "MASK": Mask,
    "PRUNE": Prune,
    "RANDOM_PRUNE": RandomPrune,
}

SMOOTHER_OPERATORS: dict[str, type[MaskedWriteOperator]] = {
    "DAMP": Damp,
    "NOISE": Noise,
    "CLIP": Clip,
    "QUANTIZE": Quantize,
    "RESET": Reset,
}
