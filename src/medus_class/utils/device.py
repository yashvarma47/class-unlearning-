"""Device selection and environment reporting.

The target machine is a single NVIDIA GTX 1650 (Turing, sm_75, 4 GB VRAM).
That 4 GB budget is the binding constraint on the whole project: batch sizes,
whether VGG19 fits, and whether Tiny ImageNet is feasible all follow from it.
``describe_environment()`` captures the facts that must be reported in the
dissertation's reproducibility section.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device and its capabilities."""

    device: str
    cuda_available: bool
    gpu_name: str | None = None
    total_memory_gb: float | None = None
    compute_capability: str | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


def get_device(prefer: str = "cuda", index: int = 0) -> DeviceInfo:
    """Resolve the compute device, falling back to CPU when CUDA is absent.

    Parameters
    ----------
    prefer:
        ``"cuda"`` to use the GPU when available, ``"cpu"`` to force CPU.
    index:
        CUDA device index (always 0 on this machine).
    """
    if torch is None:  # pragma: no cover
        return DeviceInfo(device="cpu", cuda_available=False)

    cuda_available = torch.cuda.is_available()
    if prefer != "cuda" or not cuda_available:
        return DeviceInfo(device="cpu", cuda_available=cuda_available)

    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        device=f"cuda:{index}",
        cuda_available=True,
        gpu_name=props.name,
        total_memory_gb=round(props.total_memory / (1024**3), 2),
        compute_capability=f"{props.major}.{props.minor}",
    )


def describe_environment() -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of the runtime environment.

    Saved alongside every experiment so results can be traced back to the exact
    software/hardware stack that produced them.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if torch is None:  # pragma: no cover
        info["torch"] = None
        return info

    import torchvision  # imported here so the module stays importable without it

    info.update(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_compiled_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": asdict(get_device()),
        }
    )
    return info


def empty_cache() -> None:
    """Release cached CUDA blocks.

    Called between SEC evaluations: each chromosome deep-copies the original
    model, so on a 4 GB card the allocator can fragment across a population.
    """
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_summary_gb(index: int = 0) -> dict[str, float]:
    """Current CUDA allocation in GB -- used to sanity-check batch-size limits."""
    if torch is None or not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(index) / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(index) / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(index) / 1024**3, 3),
    }
