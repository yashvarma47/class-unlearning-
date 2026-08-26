"""The layer-group registry -- the bridge between a chromosome and a model.

A chromosome addresses *layer groups*, not individual parameters. Group ``i``
corresponds to gene position ``i`` in every component of
``x = (b, g, s, d_g, d_s)``, so the number of groups **is** the chromosome
length ``L`` (currently 6 for both ResNet-18 and VGG19).

This module resolves the group definitions in ``configs/model/*.yaml`` into
concrete parameters of an instantiated model, and enforces two invariants that
the correctness of the whole search depends on:

**Disjointness**
    No parameter may belong to two groups, otherwise two genes would fight over
    the same weights and the decoded strategy would be order-dependent in a way
    the chromosome does not express.

**Coverage**
    Every trainable parameter should belong to some group, otherwise part of the
    network is silently unreachable by the search.

It also provides the freezing primitive the decoder relies on: when ``b_i = 0``
the group is frozen, and an operator must be unable to modify it even though it
computes a global loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch.nn as nn


@dataclass(frozen=True)
class LayerGroup:
    """One addressable unit of the model.

    Attributes
    ----------
    index:
        Position in the chromosome (gene index).
    name:
        Human-readable group name, e.g. ``"layer3"``.
    module_names:
        Dotted module paths belonging to this group, as given in the config.
    parameter_names:
        Fully-qualified names of every parameter inside those modules.
    """

    index: int
    name: str
    module_names: tuple[str, ...]
    parameter_names: tuple[str, ...] = field(repr=False)

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)


class LayerGroupRegistry:
    """Ordered layer groups of one instantiated model."""

    def __init__(self, model: nn.Module, groups: list[LayerGroup]) -> None:
        self.model = model
        self.groups = groups
        self._by_name = {group.name: group for group in groups}

    def __len__(self) -> int:
        """``L`` -- the chromosome length."""
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, key: int | str) -> LayerGroup:
        if isinstance(key, str):
            return self._by_name[key]
        return self.groups[key]

    @property
    def names(self) -> list[str]:
        return [group.name for group in self.groups]

    def parameters_of(self, key: int | str) -> list[nn.Parameter]:
        """Return the live ``nn.Parameter`` objects belonging to a group."""
        group = self[key]
        lookup = dict(self.model.named_parameters())
        return [lookup[name] for name in group.parameter_names]

    def modules_of(self, key: int | str) -> list[nn.Module]:
        """Return the live modules belonging to a group."""
        group = self[key]
        lookup = dict(self.model.named_modules())
        return [lookup[name] for name in group.module_names]

    def set_trainable(self, active: Iterable[bool]) -> dict[str, bool]:
        """Apply the chromosome's ``b`` mask by toggling ``requires_grad``.

        Parameters
        ----------
        active:
            One boolean per group, in chromosome order. ``True`` unfreezes the
            group, ``False`` freezes it.

        Returns
        -------
        dict
            Group name -> whether it is now trainable, for logging.

        Notes
        -----
        Freezing via ``requires_grad`` stops *gradient-based* operators from
        changing a group. Operators that write parameters directly (Fisher
        noise, weight noise, a WoodFisher update) bypass autograd entirely and
        must consult the mask themselves -- ``medus.operators.base`` passes the
        active group explicitly for exactly this reason.
        """
        active = list(active)
        if len(active) != len(self.groups):
            raise ValueError(
                f"expected {len(self.groups)} activation flags (L), got {len(active)}"
            )

        lookup = dict(self.model.named_parameters())
        status: dict[str, bool] = {}
        for group, is_active in zip(self.groups, active):
            for name in group.parameter_names:
                lookup[name].requires_grad_(bool(is_active))
            status[group.name] = bool(is_active)
        return status

    def freeze_all(self) -> None:
        self.set_trainable([False] * len(self.groups))

    def unfreeze_all(self) -> None:
        self.set_trainable([True] * len(self.groups))

    def summary(self) -> list[dict[str, Any]]:
        """JSON-serialisable description, logged with experiment results."""
        return [
            {
                "index": group.index,
                "name": group.name,
                "modules": list(group.module_names),
                "n_parameter_tensors": group.n_parameters,
            }
            for group in self.groups
        ]


def build_registry(
    model: nn.Module,
    model_cfg: dict[str, Any],
    require_full_coverage: bool = True,
) -> LayerGroupRegistry:
    """Build the registry for ``model`` from its config's ``layer_groups``.

    Parameters
    ----------
    require_full_coverage:
        If ``True`` (default), raise when a trainable parameter belongs to no
        group. Set ``False`` only for deliberately partial groupings.

    Raises
    ------
    KeyError
        If a config lists a module the model does not have -- this is the check
        that catches hand-written module indices (e.g. VGG19's ``features.N``)
        drifting out of sync with the architecture.
    ValueError
        If groups overlap, or if coverage is incomplete and required.
    """
    module_lookup = dict(model.named_modules())
    all_parameters = dict(model.named_parameters())

    groups: list[LayerGroup] = []
    seen: dict[str, str] = {}  # parameter name -> owning group

    for index, spec in enumerate(model_cfg["layer_groups"]):
        name = spec["name"]
        module_names = tuple(spec["modules"])

        parameter_names: list[str] = []
        for module_name in module_names:
            if module_name not in module_lookup:
                raise KeyError(
                    f"layer group '{name}' lists module '{module_name}', which "
                    f"does not exist in {type(model).__name__}. Available "
                    f"top-level modules: "
                    f"{[n for n, _ in model.named_children()]}"
                )
            prefix = f"{module_name}."
            for parameter_name in all_parameters:
                if parameter_name == module_name or parameter_name.startswith(prefix):
                    parameter_names.append(parameter_name)

        for parameter_name in parameter_names:
            if parameter_name in seen:
                raise ValueError(
                    f"parameter '{parameter_name}' belongs to both layer group "
                    f"'{seen[parameter_name]}' and '{name}'; groups must be disjoint"
                )
            seen[parameter_name] = name

        groups.append(
            LayerGroup(
                index=index,
                name=name,
                module_names=module_names,
                parameter_names=tuple(parameter_names),
            )
        )

    if require_full_coverage:
        uncovered = sorted(set(all_parameters) - set(seen))
        if uncovered:
            raise ValueError(
                f"{len(uncovered)} parameters belong to no layer group and are "
                f"therefore unreachable by the search: {uncovered[:5]}"
                + (" ..." if len(uncovered) > 5 else "")
            )

    return LayerGroupRegistry(model, groups)
