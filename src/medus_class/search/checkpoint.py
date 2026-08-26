"""Search checkpointing, so a long NSGA-II run survives interruption.

A 50 x 100 search is ~9.6 hours. This machine has already lost one multi-hour
job (the retrained-reference training died at epoch 180 of 200), so a run of
that length without a resume path is a gamble rather than an experiment.

What a resumable checkpoint has to hold
---------------------------------------
Saving the population and its objectives is not enough. Three further pieces
matter, and omitting any one of them makes a resumed run *different* from an
uninterrupted one rather than a continuation of it:

``rng_state``
    The NumPy generator state. Crossover and mutation draw from it, so without
    it a resumed run produces different offspring from the same parents and
    ``seed: 42`` stops meaning anything.
``records``
    Every per-individual evaluation so far. Without them the final
    ``evaluation_history.csv`` contains only the post-resume portion, which
    would silently corrupt the runtime statistics and the selectivity analysis.
``cache``
    The objective cache, keyed by canonical decoded strategy. Losing it is not
    incorrect -- entries would simply be recomputed -- but at ~16% hit rate that
    is an hour of wasted GPU on a full run.

The config fingerprint is stored alongside so a resume can refuse to continue a
run under settings that would change what the objectives mean.

Writes are atomic (temp file then replace) and the previous checkpoint is kept,
because a process killed *during* a write would otherwise leave a truncated file
and destroy the whole run it was meant to protect.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medus_class.search import Chromosome, ChromosomeBounds

#: Bumped if the on-disk layout changes incompatibly.
CHECKPOINT_VERSION = 1

#: Config keys that change what an objective value MEANS. A resume whose config
#: differs on any of these would mix two definitions into one Pareto front.
FINGERPRINT_KEYS = (
    "objective_mode",
    "forget_loss_cap",
    "reference_checkpoint",
    "population_size",
    "generations",
    "seed",
    "crossover_probability",
    "mutation_probability",
    "p_active",
    "normalise_objectives",
)


class ResumeMismatch(RuntimeError):
    """Raised when a checkpoint cannot safely continue under the given config."""


def build_fingerprint(search_cfg: dict[str, Any], sec_cfg: dict[str, Any]) -> dict[str, Any]:
    """The settings a resume must match, flattened into one comparable dict."""
    merged = {**sec_cfg, **search_cfg}
    return {key: merged.get(key) for key in FINGERPRINT_KEYS}


@dataclass
class SearchState:
    """Everything needed to continue a search exactly where it stopped."""

    version: int
    generation: int              #: last COMPLETED generation
    population: list[list[int]]  #: flat gene vectors
    objectives: list[tuple[float, float, float]]
    rng_state: dict[str, Any]
    history: list[dict[str, Any]]
    records: list[dict[str, Any]]
    cache: dict[str, tuple[float, float, float]]
    cached_status: dict[str, str]
    counters: dict[str, int]
    fingerprint: dict[str, Any]
    results_dir: str
    elapsed_seconds: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generation": self.generation,
            "population": self.population,
            "objectives": self.objectives,
            "rng_state": self.rng_state,
            "history": self.history,
            "records": self.records,
            "cache": self.cache,
            "cached_status": self.cached_status,
            "counters": self.counters,
            "fingerprint": self.fingerprint,
            "results_dir": self.results_dir,
            "elapsed_seconds": self.elapsed_seconds,
        }


def save_state(path: str | Path, state: SearchState) -> Path:
    """Write a checkpoint atomically, keeping the previous one as a fallback.

    Pickle rather than JSON: the NumPy generator state contains integer arrays
    that JSON cannot round-trip without a lossy conversion, and this file is
    written by and read by this project only.

    A ``.json`` sidecar carries the human-readable summary so a run's progress
    can be inspected without unpickling anything.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        previous = path.with_suffix(path.suffix + ".prev")
        try:
            os.replace(path, previous)
        except OSError:
            pass  # a missing fallback is not worth failing the save for

    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            pickle.dump(state.to_payload(), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": state.version,
                "generation": state.generation,
                "population_size": len(state.population),
                "evaluations_recorded": len(state.records),
                "cache_entries": len(state.cache),
                "counters": state.counters,
                "fingerprint": state.fingerprint,
                "elapsed_seconds": round(state.elapsed_seconds, 1),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def load_state(path: str | Path) -> SearchState:
    """Read a checkpoint, falling back to the previous one if it is unreadable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no search checkpoint at {path}")

    for candidate in (path, path.with_suffix(path.suffix + ".prev")):
        if not candidate.is_file():
            continue
        try:
            payload = pickle.loads(candidate.read_bytes())
        except Exception as exc:  # truncated by a kill mid-write
            print(f"  WARNING: {candidate.name} is unreadable ({exc}); trying fallback")
            continue
        if candidate != path:
            print(f"  NOTE: resumed from the fallback checkpoint {candidate.name}")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ResumeMismatch(
                f"checkpoint version {payload.get('version')} != "
                f"{CHECKPOINT_VERSION}; it was written by a different layout"
            )
        return SearchState(**payload)

    raise ResumeMismatch(f"{path} and its fallback are both unreadable")


def assert_compatible(state: SearchState, fingerprint: dict[str, Any]) -> None:
    """Refuse to resume under settings that change what the objectives mean.

    Continuing a ``loss_kl`` run under ``forget_acc`` settings, or with a
    different forget-loss cap, would put two incompatible objective definitions
    into one Pareto front -- and nothing downstream would flag it.
    """
    differences = {
        key: (state.fingerprint.get(key), fingerprint.get(key))
        for key in FINGERPRINT_KEYS
        if state.fingerprint.get(key) != fingerprint.get(key)
    }
    if differences:
        lines = "\n".join(
            f"    {key}: checkpoint={was!r}  requested={now!r}"
            for key, (was, now) in differences.items()
        )
        raise ResumeMismatch(
            "refusing to resume: the configuration differs from the "
            "checkpoint's.\n" + lines + "\n"
            "Resuming would mix two objective definitions in one front. Use the "
            "matching config, or start a fresh run."
        )


def restore_population(
    state: SearchState, bounds: ChromosomeBounds
) -> list[Chromosome]:
    """Rebuild chromosomes from their flat gene vectors."""
    return [Chromosome.from_vector(np.array(v), bounds) for v in state.population]
