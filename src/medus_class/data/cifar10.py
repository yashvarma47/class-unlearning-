"""CIFAR-10 loading and the class-unlearning loader bundle.

Adapted from the predecessor project's ``datasets.py``. The dataset handling,
transform split (augmented for training, clean for evaluation) and reproducible
loader construction are unchanged -- they were verified there and none of it is
specific to how ``D_f`` is chosen.

What is different is the bundle. The instance-level project had one ``test_eval``
loader covering the whole test set, because ``D_f`` and the test set were the
same ten classes and "utility" was a single number. Under class unlearning the
test set has two halves that mean opposite things:

* ``D_r_test`` -- utility. Should stay as high as the reference's.
* ``D_f_test`` -- the forgetting measure, on data the model never trained on
  either way. This is the headline result and it has no instance-level analogue.

Collapsing them into one loader would average the two together and hide both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from medus_class.data.class_split import ClassSplit
from medus_class.utils.config import resolve_path
from medus_class.utils.seeding import make_generator, seed_worker


@dataclass
class CIFAR10Bundle:
    """The base datasets, in both augmented and clean form.

    ``train_augmented`` and ``train_clean`` wrap the *same* 50 000 images and
    differ only in transform, so an index means the same sample in both.
    """

    train_augmented: CIFAR10
    train_clean: CIFAR10
    test: CIFAR10

    @property
    def train_size(self) -> int:
        return len(self.train_augmented)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @property
    def train_labels(self) -> np.ndarray:
        return np.asarray(self.train_clean.targets, dtype=np.int64)

    @property
    def test_labels(self) -> np.ndarray:
        return np.asarray(self.test.targets, dtype=np.int64)


def build_transforms(data_cfg: dict[str, Any]) -> tuple[Any, Any]:
    """``(train_transform, eval_transform)`` from a data config."""
    normalize = transforms.Normalize(
        mean=data_cfg["normalize"]["mean"], std=data_cfg["normalize"]["std"]
    )
    augmentation = data_cfg.get("augmentation", {})

    train_steps: list[Any] = []
    if augmentation.get("random_crop"):
        train_steps.append(
            transforms.RandomCrop(
                int(augmentation["random_crop"]),
                padding=int(augmentation.get("random_crop_padding", 4)),
            )
        )
    if augmentation.get("random_horizontal_flip"):
        train_steps.append(transforms.RandomHorizontalFlip())
    train_steps += [transforms.ToTensor(), normalize]

    return (
        transforms.Compose(train_steps),
        transforms.Compose([transforms.ToTensor(), normalize]),
    )


def load_cifar10(data_cfg: dict[str, Any]) -> CIFAR10Bundle:
    """Download (if needed) and construct the CIFAR-10 datasets."""
    root = resolve_path(data_cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    download = bool(data_cfg.get("download", True))

    train_transform, eval_transform = build_transforms(data_cfg)
    return CIFAR10Bundle(
        train_augmented=CIFAR10(
            root=str(root), train=True, download=download, transform=train_transform
        ),
        train_clean=CIFAR10(
            root=str(root), train=True, download=False, transform=eval_transform
        ),
        test=CIFAR10(
            root=str(root), train=False, download=download, transform=eval_transform
        ),
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    data_cfg: dict[str, Any],
    seed: int,
) -> DataLoader:
    """Build a reproducible ``DataLoader``."""
    num_workers = int(data_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False))
        and num_workers > 0,
        drop_last=False,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )


@dataclass
class ClassLoaders:
    """Every loader a class-unlearning evaluation needs.

    ==================  ==================================================
    ``forget_train``    ``D_f`` augmented + shuffled -- operators
    ``retain_train``    ``D_r`` augmented + shuffled -- operators
    ``forget_eval``     ``D_f`` clean + ordered -- objective ``f1``
    ``retain_eval``     ``D_r`` clean + ordered -- objective ``f2``
    ``forget_test``     ``D_f_test`` -- THE forgetting result
    ``retain_test``     ``D_r_test`` -- utility
    ==================  ==================================================
    """

    forget_train: DataLoader
    retain_train: DataLoader
    forget_eval: DataLoader
    retain_eval: DataLoader
    forget_test: DataLoader
    retain_test: DataLoader
    forget_class: int

    def sizes(self) -> dict[str, int]:
        return {
            "forget_train": len(self.forget_train.dataset),
            "retain_train": len(self.retain_train.dataset),
            "forget_eval": len(self.forget_eval.dataset),
            "retain_eval": len(self.retain_eval.dataset),
            "forget_test": len(self.forget_test.dataset),
            "retain_test": len(self.retain_test.dataset),
        }


def _fixed_subset(
    indices: np.ndarray, size: int | None, rng: np.random.RandomState
) -> np.ndarray:
    """A sorted seeded subset, or all of ``indices`` when not restricted."""
    if size is None or size >= indices.size:
        return indices
    return np.sort(rng.choice(indices, size, replace=False))


def build_class_loaders(
    bundle: CIFAR10Bundle,
    split: ClassSplit,
    data_cfg: dict[str, Any],
    seed: int = 42,
    batch_size_key: str = "train",
    forget_subset_size: int | None = None,
    retain_subset_size: int | None = None,
) -> ClassLoaders:
    """Assemble the loaders for one class split.

    Parameters
    ----------
    forget_subset_size, retain_subset_size:
        Screening-stage cost levers. Each subset is drawn once from a seeded
        RandomState and reused for every chromosome, so objectives stay
        comparable across a population. ``None`` means the full set.

        Note that ``forget_subset_size`` is not a pure measurement knob: it sizes
        both ``forget_train`` (what the operators consume) and ``forget_eval``
        (where ``f1`` is measured), so a screening run executes a slightly
        different strategy as well as measuring it more coarsely. That is
        inherent to cheap-stage screening, and it is why every reported number is
        re-measured at full fidelity.

        The **test** loaders are never subset. ``D_f_test`` holds 1 000 images and
        is the headline result; sampling it to save a few seconds would add noise
        to the one number the experiment exists to produce.
    """
    batch_sizes = data_cfg["batch_size"]
    train_bs = int(batch_sizes[batch_size_key])
    eval_bs = int(batch_sizes["eval"])

    rng = np.random.RandomState(seed)
    forget_idx = _fixed_subset(split.forget_train, forget_subset_size, rng)
    retain_eval_idx = _fixed_subset(split.retain_train, retain_subset_size, rng)

    return ClassLoaders(
        forget_train=make_loader(
            Subset(bundle.train_augmented, forget_idx.tolist()),
            train_bs, True, data_cfg, seed,
        ),
        retain_train=make_loader(
            Subset(bundle.train_augmented, split.retain_train.tolist()),
            train_bs, True, data_cfg, seed,
        ),
        forget_eval=make_loader(
            Subset(bundle.train_clean, forget_idx.tolist()),
            eval_bs, False, data_cfg, seed,
        ),
        retain_eval=make_loader(
            Subset(bundle.train_clean, retain_eval_idx.tolist()),
            eval_bs, False, data_cfg, seed,
        ),
        forget_test=make_loader(
            Subset(bundle.test, split.forget_test.tolist()),
            eval_bs, False, data_cfg, seed,
        ),
        retain_test=make_loader(
            Subset(bundle.test, split.retain_test.tolist()),
            eval_bs, False, data_cfg, seed,
        ),
        forget_class=split.forget_class,
    )
