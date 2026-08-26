"""Checkpoint save/load with provenance metadata.

Every checkpoint records how it was produced -- seed, config, torch version,
epoch, metrics -- because the original model is the fixed starting point of
every SEC evaluation. If a checkpoint is silently replaced by one trained with a
different seed or split, every objective value already computed becomes
meaningless, and without metadata that is undetectable after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from medus_class.utils.config import resolve_path

#: Bumped if the on-disk layout ever changes incompatibly.
CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class CheckpointMetadata:
    """Provenance recorded inside every checkpoint."""

    model_name: str
    dataset: str
    seed: int
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    split_file: str | None = None
    #: Recipe needed to reproduce the run: train_size, epochs, optimiser and
    #: scheduler settings, batch size. Optional and defaults to ``None`` so
    #: checkpoints written before this field existed still load unchanged.
    training_config: dict[str, Any] | None = None
    notes: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    # str() is required: torch.__version__ is a TorchVersion instance, which the
    # safe unpickler used by weights_only=True refuses to load.
    torch_version: str = field(default_factory=lambda: str(torch.__version__))
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "seed": self.seed,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "split_file": self.split_file,
            "training_config": self.training_config,
            "notes": self.notes,
            "created_at": self.created_at,
            "torch_version": self.torch_version,
            "format_version": self.format_version,
        }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> Path:
    """Save model (and optionally optimiser/scheduler) state with metadata.

    State dicts are moved to CPU so a checkpoint written on the GPU can be
    loaded on a CPU-only machine. A human-readable ``.json`` sidecar is written
    next to the checkpoint so provenance can be inspected without torch.
    """
    checkpoint_path = resolve_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "metadata": metadata.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(metadata.to_dict(), indent=2), encoding="utf-8"
    )
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights into ``model`` and return the checkpoint metadata.

    Parameters
    ----------
    strict:
        Passed to ``load_state_dict``. Keep ``True``: a silently ignored
        mismatch means the model is not the one that was trained.

    Returns
    -------
    dict
        The stored metadata dictionary.
    """
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # weights_only=True refuses to unpickle arbitrary objects; our payload is
    # tensors and plain dicts, so this is both safe and sufficient.
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    if "model_state_dict" not in payload:
        raise ValueError(
            f"{checkpoint_path} is not a MED-US checkpoint "
            f"(keys: {sorted(payload)[:5]})"
        )

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload.get("metadata", {})


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read checkpoint provenance without loading any weights."""
    checkpoint_path = resolve_path(path)
    sidecar = checkpoint_path.with_suffix(".json")
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return payload.get("metadata", {})
