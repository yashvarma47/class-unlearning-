"""Reproducible seeding for MED-US.

Every stochastic component of the pipeline (forget-set sampling, weight
initialisation, data-loader shuffling, dropout, operator noise) must be
reproducible: a chromosome evaluated twice with the same seed has to return
identical objective values, otherwise the NSGA-II selection pressure is
measuring noise rather than strategy quality.

Determinism policy
------------------
``seed_everything(seed)`` seeds ``random``, ``numpy`` and ``torch`` (CPU + CUDA).

``deterministic=True`` additionally forces cuDNN into deterministic mode and
turns on ``torch.use_deterministic_algorithms``. This is the default for SEC
evaluations. It costs throughput and makes a few CUDA kernels raise instead of
falling back to a non-deterministic implementation, so full model *training*
(Phase 3) may be run with ``deterministic=False`` for speed -- the trained
checkpoint is then fixed on disk and everything downstream stays reproducible.

Note on ``torch.use_deterministic_algorithms``: some CUDA kernels additionally
require the environment variable ``CUBLAS_WORKSPACE_CONFIG`` to be set *before*
CUDA is initialised, so this module sets it at import time.
"""

from __future__ import annotations

import os

# Must be set before the first CUDA context is created, hence at import time.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from dataclasses import dataclass

import numpy as np

try:  # torch is optional at import time so that `import medus` works pre-install
    import torch
except ImportError:  # pragma: no cover - exercised only in a broken environment
    torch = None  # type: ignore[assignment]


DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedState:
    """Record of what was seeded, for logging into result JSON files."""

    seed: int
    deterministic: bool
    cudnn_benchmark: bool


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> SeedState:
    """Seed all RNGs used by MED-US.

    Parameters
    ----------
    seed:
        Master seed. Sub-components derive their own seeds from this one.
    deterministic:
        If ``True``, force deterministic cuDNN/torch algorithms. Use ``True``
        for SEC evaluations, ``False`` for long training runs where throughput
        matters more.

    Returns
    -------
    SeedState
        The applied configuration, suitable for serialising into results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return SeedState(seed=seed, deterministic=deterministic, cudnn_benchmark=False)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a handful of ops (e.g. some pooling backward kernels) have
        # no deterministic implementation; we warn rather than hard-crash a run.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    return SeedState(
        seed=seed,
        deterministic=deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )


def seed_worker(worker_id: int) -> None:
    """Per-worker seeding hook for ``DataLoader(worker_init_fn=seed_worker)``.

    Without this, each dataloader worker process inherits an unseeded numpy /
    random state and augmentation becomes non-reproducible across runs.
    """
    if torch is None:  # pragma: no cover
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED):
    """Return a seeded ``torch.Generator`` for ``DataLoader(generator=...)``.

    Passing an explicit generator is what makes shuffling order reproducible
    independently of any global RNG consumed elsewhere in the same process.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for make_generator()")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def derive_seed(master_seed: int, *tags: object) -> int:
    """Derive a stable sub-seed from a master seed and arbitrary tags.

    Used to give each (split, chromosome, repeat) its own reproducible stream
    without them all sharing -- and therefore correlating with -- one RNG.

    >>> derive_seed(42, "forget_split", 10) == derive_seed(42, "forget_split", 10)
    True
    """
    # Python's builtin hash() is salted per process (PYTHONHASHSEED), so use a
    # stable arithmetic mix instead.
    value = master_seed & 0xFFFFFFFF
    for tag in tags:
        for byte in str(tag).encode("utf-8"):
            value = (value * 31 + byte) & 0xFFFFFFFF
    return value
