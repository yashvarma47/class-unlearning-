"""YAML configuration loading with `_base_` composition.

Configs are split by concern (``configs/data``, ``configs/model``,
``configs/operators``, ``configs/evolution``) so that an experiment config can
compose them:

.. code-block:: yaml

    _base_:
      - data/cifar10.yaml
      - model/resnet18.yaml
    seed: 42

Later entries in ``_base_`` override earlier ones; the including file overrides
all of its bases. Merging is recursive for dictionaries and replace-wins for
lists and scalars.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repository root = three parents up from src/medus/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data"


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` relative to the project root when it is not absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested dictionaries are merged key-by-key; every other type is replaced
    wholesale so that, for example, overriding a list of layer groups does not
    leave stale entries behind.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving any ``_base_`` includes.

    Parameters
    ----------
    path:
        Path to the YAML file, absolute or relative to the project root. Bare
        names such as ``"data/cifar10.yaml"`` are also resolved against
        ``configs/``.

    Raises
    ------
    FileNotFoundError
        If the config (or one of its bases) does not exist.
    ValueError
        If the ``_base_`` includes form a cycle.
    """
    config_path = _locate(path)
    _seen = set() if _seen is None else _seen
    if config_path in _seen:
        raise ValueError(f"Circular _base_ include detected at {config_path}")
    _seen = _seen | {config_path}

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} must contain a mapping at the top level")

    bases = raw.pop("_base_", [])
    if isinstance(bases, (str, Path)):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases:
        merged = deep_merge(merged, load_config(base, _seen))

    return deep_merge(merged, raw)


def _locate(path: str | Path) -> Path:
    """Find a config file, trying the project root then ``configs/``."""
    candidate = Path(path)
    for option in (candidate, PROJECT_ROOT / candidate, CONFIG_ROOT / candidate):
        if option.is_file():
            return option.resolve()
    raise FileNotFoundError(
        f"Config not found: {path!r} (searched {candidate}, {PROJECT_ROOT / candidate}, "
        f"{CONFIG_ROOT / candidate})"
    )
