"""Which CIFAR-10 class carries the most forget-specific structure?

Background
----------
``analyse_forget_specificity.py`` asked whether the *instance-level* forget set
leaves any trace in the network's channels. It does not: 0.55% of channels
exceeded the noise floor where pure noise gives 1.00%. D_f and D_r were the same
ten classes, so the network had no reason to treat them differently, and no
weight-editing operator could be selective.

The class-unlearning contrast tests the other side of that claim. If the ceiling
exists because D_f and D_r were statistically identical, then a forget set that
is a whole class -- genuinely different data -- should show real structure.

This script measures that for **all ten classes** before any class is chosen, so
the Plan A run is aimed at the class with the most structure rather than at
whichever one comes first alphabetically. It needs no reference model and no
training: forward passes through the existing original checkpoint only.

Method
------
For each candidate forget class ``c``, three sets of the SAME size ``N``:

    F     N training samples of class c                       -- the forget set
    R1    N training samples drawn from the other nine        -- retain data
    R2    N more, disjoint from R1                            -- the null control

Per channel, the normalised contrast

.. math::

    c_j = \\frac{a^{F}_j - a^{R}_j}{a^{F}_j + a^{R}_j},
    \\qquad a^{R} = (a^{R_1} + a^{R_2}) / 2

is compared against the null contrast between R1 and R2. Both halves are retain
data, so anything the null finds is sampling noise by construction. That is the
yardstick, and it is why all three sets must be the same size: a larger sample
is quieter and would look artificially specific.

Signal-to-noise per layer group is ``sd(real) / sd(null)``. A group at 1.0 says
the class is indistinguishable from more retain data; well above 1.0 says the
class has its own channels.

Cost
----
All groups' modules are hooked in ONE forward pass per set, not one pass per
group, which is what makes ten classes affordable.

Run::

    .venv\\Scripts\\python.exe experiments/analyse_class_structure.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data import CIFAR10_CLASS_NAMES, load_cifar10, make_loader  # noqa: E402
from medus_class.models import build_model, build_registry, load_checkpoint  # noqa: E402
from medus_class.operators.selection import (  # noqa: E402
    _input_activation_norms,
    _target_modules,
)
from medus_class.utils.config import load_config, resolve_path  # noqa: E402
from medus_class.utils.device import get_device  # noqa: E402

#: Groups with fewer channels than this cannot support the statistic and are
#: excluded from the verdict. ResNet-18's stem reads three RGB inputs, so "1 of
#: 3 channels above the noise floor" reads as 33% while meaning nothing.
#: Identical to the threshold in analyse_forget_specificity.py.
MIN_CHANNELS = 32


def contrast(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """(a - b) / (a + b), per channel, guarded against dead channels."""
    total = (a + b).clamp(min=1e-12)
    return ((a - b) / total).cpu().numpy()


def stratified_retain_indices(
    labels: np.ndarray, forget_class: int, total: int, rng: np.random.RandomState
) -> np.ndarray:
    """Draw ``total`` retain indices spread evenly over the other nine classes.

    A uniform draw from the pooled retain set would be *approximately* balanced
    and would probably do; drawing per class makes it exact, so a class that
    happens to be over-sampled cannot masquerade as structure.
    """
    others = [c for c in range(10) if c != forget_class]
    per_class = int(np.ceil(total / len(others)))

    chunks = []
    for c in others:
        pool = np.flatnonzero(labels == c)
        take = min(per_class, pool.size)
        chunks.append(rng.choice(pool, take, replace=False))

    drawn = np.concatenate(chunks)
    rng.shuffle(drawn)
    if drawn.size < total:
        raise ValueError(f"retain pool exhausted: wanted {total}, got {drawn.size}")
    return drawn[:total]


def activation_norms_over(
    model, modules, dataset, indices, data_cfg, device, seed: int
) -> dict[str, torch.Tensor]:
    """RMS input activations for every module, over the given sample indices."""
    loader = make_loader(
        Subset(dataset, indices.tolist()),
        int(data_cfg["batch_size"]["eval"]),
        False,
        data_cfg,
        seed,
    )
    return _input_activation_norms(model, modules, loader, device)


def analyse_class(
    model, group_modules, dataset, labels, forget_class, n_samples, data_cfg,
    device, seed,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-group structure statistics for one candidate forget class."""
    rng = np.random.RandomState(seed + forget_class)

    forget_idx = np.flatnonzero(labels == forget_class)
    if forget_idx.size < n_samples:
        raise ValueError(
            f"class {forget_class} has {forget_idx.size} samples, "
            f"fewer than the requested {n_samples}"
        )
    forget_idx = rng.choice(forget_idx, n_samples, replace=False)

    retain_idx = stratified_retain_indices(labels, forget_class, 2 * n_samples, rng)
    retain_a, retain_b = retain_idx[:n_samples], retain_idx[n_samples:]

    # Every module in every group, hooked in one pass per set.
    all_modules = [m for mods in group_modules.values() for m in mods]

    a_forget = activation_norms_over(
        model, all_modules, dataset, forget_idx, data_cfg, device, seed)
    a_first = activation_norms_over(
        model, all_modules, dataset, retain_a, data_cfg, device, seed)
    a_second = activation_norms_over(
        model, all_modules, dataset, retain_b, data_cfg, device, seed)

    group_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []

    for group_name, modules in group_modules.items():
        real_all: list[float] = []
        null_all: list[float] = []

        for name, _module in modules:
            if name not in a_forget or name not in a_first or name not in a_second:
                continue
            a_f, a_1, a_2 = a_forget[name], a_first[name], a_second[name]
            a_r = (a_1 + a_2) / 2.0

            real = contrast(a_f, a_r)
            null = contrast(a_1, a_2)
            real_all.extend(real.tolist())
            null_all.extend(null.tolist())

            for channel, (r, n) in enumerate(zip(real, null)):
                channel_rows.append({
                    "forget_class": forget_class,
                    "class_name": CIFAR10_CLASS_NAMES[forget_class],
                    "group": group_name,
                    "module": name,
                    "channel": channel,
                    "rms_forget": round(float(a_f[channel]), 6),
                    "rms_retain": round(float(a_r[channel]), 6),
                    "contrast_forget_vs_retain": round(float(r), 6),
                    "contrast_null_retain_vs_retain": round(float(n), 6),
                })

        if not real_all:
            continue

        real_np = np.array(real_all)
        null_np = np.array(null_all)
        threshold = float(np.percentile(np.abs(null_np), 99))
        beyond = int((np.abs(real_np) > threshold).sum())

        group_rows.append({
            "forget_class": forget_class,
            "class_name": CIFAR10_CLASS_NAMES[forget_class],
            "group": group_name,
            "channels": int(real_np.size),
            "real_sd": float(real_np.std()),
            "null_sd": float(null_np.std()),
            "snr": float(real_np.std() / max(null_np.std(), 1e-12)),
            "real_max_abs": float(np.abs(real_np).max()),
            "null_p99": threshold,
            "channels_beyond_noise": beyond,
            "pct_beyond_noise": 100.0 * beyond / real_np.size,
        })

    return group_rows, channel_rows


def summarise_class(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one class's per-group rows into the headline numbers."""
    large = [g for g in group_rows if g["channels"] >= MIN_CHANNELS]
    if not large:
        raise ValueError("no layer group large enough to summarise")

    snrs = np.array([g["snr"] for g in large])
    total_channels = sum(g["channels"] for g in large)
    total_beyond = sum(g["channels_beyond_noise"] for g in large)

    strongest = sorted(large, key=lambda g: g["snr"], reverse=True)

    return {
        "forget_class": group_rows[0]["forget_class"],
        "class_name": group_rows[0]["class_name"],
        "median_snr": float(np.median(snrs)),
        "max_snr": float(snrs.max()),
        "mean_snr": float(snrs.mean()),
        "channels": total_channels,
        "channels_beyond_noise": total_beyond,
        "pct_beyond_noise": 100.0 * total_beyond / total_channels,
        "strongest_groups": [
            {"group": g["group"], "snr": round(g["snr"], 3),
             "pct_beyond_noise": round(g["pct_beyond_noise"], 2)}
            for g in strongest[:3]
        ],
        "per_group_snr": {g["group"]: round(g["snr"], 3) for g in large},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="data/cifar10_class.yaml")
    parser.add_argument("--model-config", default="model/resnet18.yaml")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/cifar10_resnet18_seed42_best.pt",
        help="The ORIGINAL model. No reference model is needed.",
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Samples per set. All three sets use this many, so the noise "
             "floor is identical across them and across classes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", default="all",
                        help="'all' or a comma-separated list of labels.")
    parser.add_argument("--out", default="results/analysis/class_structure")
    parser.add_argument("--no-channel-csv", action="store_true",
                        help="Skip the per-channel dump.")
    args = parser.parse_args()

    data_cfg = dict(load_config(args.data_config)["data"])
    model_cfg = load_config(args.model_config)["model"]
    device = get_device().device

    # Thirty short-lived loaders (three sets x ten classes) each spawning
    # Windows worker processes costs far more than the forward passes it feeds.
    # CIFAR-10 is already resident, so single-process loading is strictly
    # faster here. This is a measurement-harness choice and touches nothing the
    # search or the SEC uses.
    data_cfg["num_workers"] = 0
    data_cfg["persistent_workers"] = False

    print("=" * 100)
    print("WHICH CIFAR-10 CLASS CARRIES THE MOST FORGET-SPECIFIC STRUCTURE?")
    print("=" * 100)

    bundle = load_cifar10(data_cfg)
    labels = np.asarray(bundle.train_clean.targets, dtype=np.int64)

    model = build_model(model_cfg, num_classes=int(data_cfg["num_classes"]))
    metadata = load_checkpoint(args.checkpoint, model)
    model.to(device).eval()

    registry = build_registry(model, model_cfg)
    group_modules = {
        group.name: _target_modules(model, group) for group in registry.groups
    }
    group_modules = {name: mods for name, mods in group_modules.items() if mods}

    print(f"  checkpoint   {args.checkpoint}")
    print(f"  test acc     {metadata.get('test_accuracy', 'n/a')}")
    print(f"  device       {device}")
    print(f"  groups       {', '.join(group_modules)}")
    print(f"  samples/set  {args.samples}  (forget, retain half A, retain half B)")

    if args.classes == "all":
        candidates = list(range(10))
    else:
        candidates = [int(c) for c in args.classes.split(",")]

    all_group_rows: list[dict[str, Any]] = []
    all_channel_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    print("\n" + "-" * 100)
    print("PER-CLASS RESULTS")
    print("-" * 100)
    print(f"  {'class':<12}{'median SNR':>12}{'max SNR':>10}"
          f"{'above noise':>14}{'strongest groups':>30}")

    started = time.time()
    for forget_class in candidates:
        group_rows, channel_rows = analyse_class(
            model, group_modules, bundle.train_clean, labels, forget_class,
            args.samples, data_cfg, device, args.seed,
        )
        summary = summarise_class(group_rows)

        all_group_rows.extend(group_rows)
        if not args.no_channel_csv:
            all_channel_rows.extend(channel_rows)
        summaries.append(summary)

        strongest = "  ".join(
            f"{g['group']} {g['snr']:.2f}" for g in summary["strongest_groups"][:2]
        )
        print(f"  {summary['class_name']:<12}{summary['median_snr']:>12.3f}"
              f"{summary['max_snr']:>10.3f}"
              f"{summary['pct_beyond_noise']:>13.2f}%{strongest:>30}")

    elapsed = time.time() - started

    ranked = sorted(summaries, key=lambda s: s["median_snr"], reverse=True)

    print("\n" + "-" * 100)
    print("RANKED BY STRUCTURE  (median SNR across layer groups)")
    print("-" * 100)
    print(f"  {'#':<4}{'class':<12}{'median SNR':>12}{'max SNR':>10}"
          f"{'above noise':>14}{'best group':>18}")
    for rank, s in enumerate(ranked, start=1):
        best = s["strongest_groups"][0]
        print(f"  {rank:<4}{s['class_name']:<12}{s['median_snr']:>12.3f}"
              f"{s['max_snr']:>10.3f}{s['pct_beyond_noise']:>13.2f}%"
              f"{best['group']:>12} {best['snr']:.2f}")

    print("\n  Reading this table:")
    print("    median SNR  -- sd(real contrast) / sd(null contrast), median over")
    print("                   layer groups. 1.0 means the class is")
    print("                   indistinguishable from more retain data.")
    print("    above noise -- channels exceeding the null's 99th percentile,")
    print("                   pooled over groups. Pure noise gives about 1%.")
    print(f"    groups with < {MIN_CHANNELS} channels are excluded from both.")

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "per_class_groups.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_group_rows[0]))
        writer.writeheader()
        writer.writerows(all_group_rows)

    if all_channel_rows:
        with (out / "channel_contrast_all_classes.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_channel_rows[0]))
            writer.writeheader()
            writer.writerows(all_channel_rows)

    payload = {
        "checkpoint": args.checkpoint,
        "samples_per_set": args.samples,
        "seed": args.seed,
        "min_channels_for_verdict": MIN_CHANNELS,
        "elapsed_seconds": round(elapsed, 1),
        "ranked": ranked,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    winner = ranked[0]
    print("\n" + "=" * 100)
    print("RECOMMENDED FORGET CLASS")
    print("=" * 100)
    print(f"  {winner['class_name']} (label {winner['forget_class']})")
    print(f"    median SNR    {winner['median_snr']:.3f}")
    print(f"    max SNR       {winner['max_snr']:.3f}")
    print(f"    above noise   {winner['pct_beyond_noise']:.2f}%  "
          f"({winner['channels_beyond_noise']} of {winner['channels']} channels)")
    print("    strongest     " + ", ".join(
        f"{g['group']} (SNR {g['snr']:.2f}, {g['pct_beyond_noise']:.1f}% above noise)"
        for g in winner["strongest_groups"]))

    print("\n  For comparison, the instance-level forget set gave 0.55% above")
    print("  noise, against the 1.00% pure noise produces. Anything at or below")
    print("  1% here would mean class unlearning has no more structure to")
    print("  exploit than instance unlearning did, and the explanation on")
    print("  record is wrong.")

    print(f"\n  elapsed {elapsed:.1f}s")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
