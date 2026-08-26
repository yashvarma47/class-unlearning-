"""The unlearning-operator interface.

Every operator implements one method::

    apply(model, context, group, hparams) -> OperatorResult

and must obey one rule: **it may only change parameters inside ``group``.**
That rule is what makes a layer-wise chromosome meaningful. If an operator
leaked changes into other groups, the genes for those groups would no longer
describe what happened to them.

Two distinct leaks have to be closed, and only the first is obvious:

1. **Gradient leak.** Handled by ``requires_grad`` plus an optimiser
   constructed over the group's parameters only.

2. **BatchNorm statistics leak.** ``running_mean`` / ``running_var`` are updated
   during the *forward* pass whenever a BN module is in training mode --
   ``requires_grad`` does not prevent it, and no optimiser is involved. A
   gradient step on ``layer4`` would therefore silently shift the BN statistics
   of every earlier layer and change their behaviour. ``restrict_to_group``
   puts every module outside the active group into eval mode for the duration
   of the operator, which closes this.

Operators are stateless and must not assume they run first: a chromosome may
apply a gradient operator and then a smoothing operator to the same group, and
different groups are processed in sequence on the same model object.
"""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.nn as nn

from medus_class.data import ClassLoaders
from medus_class.models.layer_groups import LayerGroup, LayerGroupRegistry
from medus_class.utils.seeding import derive_seed


@dataclass
class OperatorContext:
    """Everything an operator needs beyond the model itself.

    Attributes
    ----------
    loaders:
        ``D_f`` / ``D_r`` / ``D_T`` loaders for the current split.
    registry:
        Layer-group registry for the model being modified.
    device:
        Device string, e.g. ``"cuda:0"``.
    num_classes:
        Needed by operators that resample labels (RL, boundary methods).
    seed:
        Base seed; operators derive their own stream from it so that two
        evaluations of the same chromosome behave identically.
    batch_cap:
        Optional maximum number of batches per epoch. This is the primary SEC
        cost lever: a full pass over ``D_r`` (45 000 samples) per operator per
        group would make a population evaluation unaffordable on a 4 GB GPU.
        ``None`` means no cap.
    """

    loaders: ClassLoaders
    registry: LayerGroupRegistry
    device: str = "cpu"
    num_classes: int = 10
    seed: int = 42
    batch_cap: int | None = None
    #: Channel statistics the images were normalised with. Boundary Shrink needs
    #: them to move perturbations back into pixel space: FGSM's epsilon and the
    #: ``clamp(0, 1)`` are only meaningful on raw images, not on normalised
    #: tensors.
    normalize_mean: tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    normalize_std: tuple[float, ...] = (0.2470, 0.2435, 0.2616)
    #: Run-level selector override. When set, it replaces the lookup table's
    #: rule for every operator whose rule is ``class_contrast`` -- i.e. the
    #: forget-informed ones. PRUNE and RANDOM_PRUNE keep their own rules,
    #: because magnitude and random selection are what those operators ARE;
    #: overriding them would delete the two controls the comparison rests on.
    selection_rule: str | None = None
    #: Instance-DAMP channel scores, computed once when the SEC is built and
    #: reused for every chromosome. They describe the ORIGINAL model, so
    #: recomputing them after each edit would measure a different network than
    #: the one being unlearned from. Keyed by layer-group name; empty unless a
    #: selection rule needs them.
    instance_scores: dict[str, Any] = field(default_factory=dict)
    #: Per-group weight snapshots taken by the SEC *before* it touches a group.
    #: SalUn needs them to restore non-salient coordinates to their
    #: pre-gradient-operator values. Keyed by layer-group name.
    group_snapshots: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def take_snapshot(self, model: nn.Module, group: LayerGroup) -> None:
        """Record a group's weights so a later operator can restore from them."""
        state = model.state_dict()
        self.group_snapshots[group.name] = {
            name: state[name].detach().clone() for name in group.parameter_names
        }

    def normalization_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` shaped ``(1, C, 1, 1)`` on the context device."""
        mean = torch.tensor(self.normalize_mean, device=self.device).view(1, -1, 1, 1)
        std = torch.tensor(self.normalize_std, device=self.device).view(1, -1, 1, 1)
        return mean, std

    def loader_for(self, requirement: str):
        """Return the training loader named by an operator's ``requires`` entry."""
        if requirement == "forget":
            return self.loaders.forget_train
        if requirement == "retain":
            return self.loaders.retain_train
        raise KeyError(f"unknown data requirement '{requirement}'")


@dataclass
class OperatorResult:
    """What an operator did, for logging and for the SEC's audit trail."""

    operator: str
    group: str
    hparams: dict[str, Any]
    steps: int = 0
    mean_loss: float = float("nan")
    wall_time: float = 0.0
    parameter_delta_norm: float = 0.0
    diverged: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "group": self.group,
            "hparams": self.hparams,
            "steps": self.steps,
            "mean_loss": self.mean_loss,
            "wall_time": round(self.wall_time, 3),
            "parameter_delta_norm": self.parameter_delta_norm,
            "diverged": self.diverged,
            **({"extra": self.extra} if self.extra else {}),
        }


@contextmanager
def restrict_to_group(
    model: nn.Module, registry: LayerGroupRegistry, group: LayerGroup
) -> Iterator[list[nn.Parameter]]:
    """Restrict all modification to ``group`` for the duration of the block.

    Yields the group's parameters, ready to hand to an optimiser.

    On entry:
      * every parameter outside ``group`` gets ``requires_grad = False``;
      * every module outside ``group`` is switched to eval mode, freezing
        BatchNorm running statistics and disabling dropout;
      * modules inside ``group`` are switched to train mode.

    On exit the previous ``requires_grad`` flags and training modes are
    restored, so operators compose without leaking state into each other.
    """
    previous_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    previous_training = {name: module.training for name, module in model.named_modules()}

    group_parameter_names = set(group.parameter_names)
    group_module_names = set(group.module_names)

    try:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in group_parameter_names)

        for name, module in model.named_modules():
            in_group = any(
                name == module_name or name.startswith(f"{module_name}.")
                for module_name in group_module_names
            )
            module.train(in_group)

        yield [
            parameter
            for name, parameter in model.named_parameters()
            if name in group_parameter_names
        ]
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(previous_requires_grad[name])
        for name, module in model.named_modules():
            module.train(previous_training[name])


def parameter_snapshot(parameters: list[nn.Parameter]) -> list[torch.Tensor]:
    """Detached copies, used to measure how far an operator moved the weights."""
    return [parameter.detach().clone() for parameter in parameters]


def delta_norm(
    before: list[torch.Tensor], parameters: list[nn.Parameter]
) -> float:
    """L2 norm of the total parameter change, as a single diagnostic number.

    A norm of exactly zero means the operator did nothing -- usually a sign the
    intensity level is too low to matter, or that the group was frozen.
    """
    total = torch.zeros((), dtype=torch.float64)
    for old, new in zip(before, parameters):
        total += (new.detach().double() - old.double()).pow(2).sum().cpu()
    return float(total.sqrt())


class UnlearningOperator(ABC):
    """Base class for every unlearning operator.

    Subclasses declare their identity and data requirements as class
    attributes, and implement :meth:`apply`.
    """

    #: Short name, matching ``configs/operators/lookup.yaml``.
    name: str = "base"
    #: ``"gradient"`` or ``"smoothing"``.
    family: str = "gradient"
    #: Which data splits the operator needs: ``"forget"`` and/or ``"retain"``.
    requires: tuple[str, ...] = ()

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        context: OperatorContext,
        group: LayerGroup,
        hparams: dict[str, Any],
    ) -> OperatorResult:
        """Modify ``model`` in place, restricted to ``group``.

        Parameters
        ----------
        model:
            The model being unlearned. Modified in place; the SEC is
            responsible for having cloned the original beforehand.
        context:
            Loaders, registry, device, seed and the batch cap.
        group:
            The only layer group this call may change.
        hparams:
            Concrete hyperparameters resolved from the chromosome's ordinal
            intensity level via the operator lookup table.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} family={self.family!r}>"


class GradientOperatorBase(UnlearningOperator):
    """Shared training loop for the gradient-based operators.

    GA, FT and RL differ only in *which* data they use and *what loss* they
    compute; the surrounding loop -- restrict to the group, build an SGD
    optimiser over that group, iterate for ``epochs``, respect ``batch_cap``,
    abort on a non-finite loss -- is identical, so it lives here once.
    """

    family = "gradient"
    #: Loaders to iterate over, in order, within each epoch. RL needs two
    #: (``("forget", "retain")``) because the reference implementation performs a
    #: random-label pass over the forget set followed by a normal pass over the
    #: retain set.
    data_sources: tuple[str, ...] = ("forget",)
    #: Loaders combined into ONE loss per step, rather than iterated one after
    #: another. When set, ``data_sources`` is ignored and
    #: :meth:`compute_joint_loss` is called instead of :meth:`compute_loss`.
    #:
    #: The distinction is the whole point of NegGrad+: descending
    #: ``alpha * CE(retain) - (1 - alpha) * CE(forget)`` in a single backward
    #: pass is *not* the same as a gradient-descent step on the retain set
    #: followed by an ascent step on the forget set. The joint gradient can
    #: cancel where the two objectives conflict; sequential steps cannot.
    #:
    #: The first entry drives the epoch length; the rest are cycled, matching
    #: the reference's ``zip(r_loader, cycle(f_loader))``.
    joint_sources: tuple[str, ...] = ()

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        context: OperatorContext,
        generator: torch.Generator,
        source: str,
    ) -> torch.Tensor:
        """Return the scalar loss to descend for one batch.

        ``source`` names the loader the batch came from, so an operator with
        several data sources can treat them differently (RL randomises labels on
        the forget pass only).
        """

    def compute_joint_loss(
        self,
        model: nn.Module,
        batches: dict[str, tuple[torch.Tensor, torch.Tensor]],
        context: OperatorContext,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Return one scalar loss combining a batch from each joint source.

        ``batches`` maps each name in :attr:`joint_sources` to its
        ``(images, labels)``, already moved to the device. Only operators that
        set :attr:`joint_sources` need to implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets joint_sources but does not implement "
            f"compute_joint_loss()"
        )

    def prepare(
        self,
        model: nn.Module,
        context: OperatorContext,
        group: LayerGroup,
        hparams: dict[str, Any],
        group_parameters: list[nn.Parameter],
    ) -> None:
        """Optional hook, run once before the training loop begins.

        Called with the model still **unmodified**, which is what Boundary
        Shrink needs: it takes a frozen copy here to generate adversarial labels
        from, so the labels come from the model as it was at the start of the
        operator rather than one that is drifting under its own updates.

        ``group_parameters`` is stashed on the instance too, for operators whose
        loss depends on the group's weights (the L1-sparse penalty).

        Storing state on ``self`` is safe: ``registry.build_operator`` constructs
        a fresh operator instance for every decoded action, so no state outlives
        a single ``apply`` call.
        """
        return None

    def cleanup(self) -> None:
        """Optional hook, run after the loop. Frees anything ``prepare`` held."""
        return None

    def apply(
        self,
        model: nn.Module,
        context: OperatorContext,
        group: LayerGroup,
        hparams: dict[str, Any],
    ) -> OperatorResult:
        started = time.perf_counter()
        epochs = int(hparams.get("epochs", 1))
        lr = float(hparams["lr"])

        # Seeded per (operator, group) so label resampling in RL is reproducible
        # yet different across groups rather than repeating the same draw.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(context.seed, self.name, group.name))

        result = OperatorResult(
            operator=self.name, group=group.name, hparams=dict(hparams)
        )

        with restrict_to_group(model, context.registry, group) as group_parameters:
            if not group_parameters:
                result.wall_time = time.perf_counter() - started
                result.extra["skipped"] = "group has no parameters"
                return result

            before = parameter_snapshot(group_parameters)
            self._group_parameters = group_parameters
            self.prepare(model, context, group, hparams, group_parameters)
            optimizer = torch.optim.SGD(group_parameters, lr=lr)

            total_loss = 0.0
            steps = 0
            for _ in range(epochs):
                if self.joint_sources:
                    steps, total_loss, result.diverged = self._run_joint_epoch(
                        model, context, generator, optimizer, steps, total_loss
                    )
                    if result.diverged:
                        break
                    continue

                for source in self.data_sources:
                    loader = context.loader_for(source)
                    for batch_index, (images, labels) in enumerate(loader):
                        if (
                            context.batch_cap is not None
                            and batch_index >= context.batch_cap
                        ):
                            break

                        images = images.to(context.device, non_blocking=True)
                        labels = labels.to(context.device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        loss = self.compute_loss(
                            model, images, labels, context, generator, source
                        )

                        if not torch.isfinite(loss):
                            # Gradient ascent at a high intensity can genuinely
                            # diverge. That is a legitimate (very poor)
                            # chromosome, not a crash: stop this operator, flag
                            # it, and let the SEC score the damage.
                            result.diverged = True
                            break

                        loss.backward()
                        optimizer.step()

                        total_loss += float(loss.item())
                        steps += 1

                    if result.diverged:
                        break

                if result.diverged:
                    break

            result.steps = steps
            result.mean_loss = total_loss / steps if steps else float("nan")
            result.parameter_delta_norm = delta_norm(before, group_parameters)
            self.cleanup()

        result.wall_time = time.perf_counter() - started
        return result

    def _run_joint_epoch(
        self,
        model: nn.Module,
        context: OperatorContext,
        generator: torch.Generator,
        optimizer: torch.optim.Optimizer,
        steps: int,
        total_loss: float,
    ) -> tuple[int, float, bool]:
        """One epoch of joint-loss steps over several loaders simultaneously.

        The first joint source drives the epoch length and the rest are cycled,
        so a small forget set is reused against a larger retain set rather than
        truncating it -- the reference's ``zip(r_loader, cycle(f_loader))``.
        """
        primary, *others = self.joint_sources
        iterators = [context.loader_for(primary)] + [
            itertools.cycle(context.loader_for(name)) for name in others
        ]

        for batch_index, combined in enumerate(zip(*iterators)):
            if context.batch_cap is not None and batch_index >= context.batch_cap:
                break

            batches = {
                name: (
                    images.to(context.device, non_blocking=True),
                    labels.to(context.device, non_blocking=True),
                )
                for name, (images, labels) in zip(self.joint_sources, combined)
            }

            optimizer.zero_grad(set_to_none=True)
            loss = self.compute_joint_loss(model, batches, context, generator)

            if not torch.isfinite(loss):
                return steps, total_loss, True

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        return steps, total_loss, False
