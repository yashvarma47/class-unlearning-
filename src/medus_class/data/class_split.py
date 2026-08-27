"""Class-level forget/retain splits. This project supports no other kind.

.. math::

    D_f = \\{(x, y) \\in D : y = c\\}, \\qquad D_r = D \\setminus D_f

for a single forget class ``c``. The test set is partitioned the same way, into
``D_f_test`` and ``D_r_test``.

Why random-instance splitting is absent by design
-------------------------------------------------
The predecessor project searched instance-level unlearning -- ``D_f`` a random
subset of training indices -- across 10 534 evaluated strategies, three operator
families, five selectors and four objective formulations. Best selectivity
measured 1.158 against ~932 for retraining. A channel-level measurement
explained why: only 0.55% of channels responded differently to ``D_f`` than to
``D_r``, where pure sampling noise alone produces 1.00%. ``D_f`` and ``D_r`` were
the same ten classes, so the network used the same weights for both and no
weight edit could be selective.

That mode is therefore not carried over, not even as an option. Supporting it
would invite exactly the comparison that has already been settled, and would let
a config typo turn a class-unlearning run into an instance-unlearning one
silently -- the two produce identically shaped 5 000-index splits on CIFAR-10 and
are indistinguishable from the files alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medus_class.utils.config import resolve_path

#: CIFAR-10 label order as shipped by torchvision. Index == integer label.
CIFAR10_CLASS_NAMES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

#: The project's forget class. Chosen by measurement, not convention -- see
#: ``experiments/analyse_class_structure.py`` and docs/class_structure.md.
DEFAULT_FORGET_CLASS = 6  # frog


@dataclass(frozen=True)
class ClassSplit:
    """A forget/retain partition of both the training set and the test set.

    Attributes
    ----------
    forget_class:
        The CIFAR-10 label being unlearned.
    forget_train / retain_train:
        Sorted training indices. Their union is the whole training set.
    forget_test / retain_test:
        Sorted test indices. Their union is the whole test set.
    """

    forget_class: int
    forget_train: np.ndarray
    retain_train: np.ndarray
    forget_test: np.ndarray
    retain_test: np.ndarray

    def __post_init__(self) -> None:
        self.validate()

    @property
    def class_name(self) -> str:
        return CIFAR10_CLASS_NAMES[self.forget_class]

    def validate(self) -> None:
        """Assert both partitions are genuine, disjoint and complete.

        Cheap, and it runs on every construction. A split that silently overlaps
        would put forget samples in the retain set and make every downstream
        number meaningless in a way no later check would catch.
        """
        for name, forget, retain in (
            ("train", self.forget_train, self.retain_train),
            ("test", self.forget_test, self.retain_test),
        ):
            if forget.size == 0:
                raise ValueError(f"{name}: forget set is empty")
            if retain.size == 0:
                raise ValueError(f"{name}: retain set is empty")
            if np.intersect1d(forget, retain).size:
                raise ValueError(f"{name}: D_f and D_r overlap")
            if forget.size != np.unique(forget).size:
                raise ValueError(f"{name}: duplicate forget indices")
            if retain.size != np.unique(retain).size:
                raise ValueError(f"{name}: duplicate retain indices")

    def summary(self) -> dict[str, object]:
        return {
            "forget_class": int(self.forget_class),
            "forget_class_name": self.class_name,
            "n_forget_train": int(self.forget_train.size),
            "n_retain_train": int(self.retain_train.size),
            "n_forget_test": int(self.forget_test.size),
            "n_retain_test": int(self.retain_test.size),
        }


def build_class_split(
    train_labels: np.ndarray | list[int],
    test_labels: np.ndarray | list[int],
    forget_class: int = DEFAULT_FORGET_CLASS,
) -> ClassSplit:
    """Partition train and test by ``forget_class``.

    There is no seed and no randomness: the label fully determines the split, so
    it is reproducible by construction rather than by bookkeeping.
    """
    if not 0 <= forget_class < len(CIFAR10_CLASS_NAMES):
        raise ValueError(
            f"forget_class must be a CIFAR-10 label 0-9, got {forget_class}"
        )

    train = np.asarray(train_labels, dtype=np.int64)
    test = np.asarray(test_labels, dtype=np.int64)

    forget_train = np.sort(np.flatnonzero(train == forget_class))
    forget_test = np.sort(np.flatnonzero(test == forget_class))

    if forget_train.size == 0:
        raise ValueError(f"class {forget_class} has no training samples")
    if forget_test.size == 0:
        raise ValueError(f"class {forget_class} has no test samples")

    return ClassSplit(
        forget_class=int(forget_class),
        forget_train=forget_train,
        retain_train=np.setdiff1d(np.arange(train.size), forget_train),
        forget_test=forget_test,
        retain_test=np.setdiff1d(np.arange(test.size), forget_test),
    )


def save_class_split(split: ClassSplit, path: str | Path) -> Path:
    """Write the split to JSON, indices included.

    Everything is stored rather than only the forget indices. The predecessor
    project stored one side and derived the other, which is compact but means a
    reader cannot verify the partition without reconstructing it.
    """
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **split.summary(),
        "forget_train": split.forget_train.tolist(),
        "retain_train": split.retain_train.tolist(),
        "forget_test": split.forget_test.tolist(),
        "retain_test": split.retain_test.tolist(),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def load_class_split(path: str | Path) -> ClassSplit:
    """Load a split written by :func:`save_class_split`."""
    source = resolve_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"class split not found: {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    return ClassSplit(
        forget_class=int(payload["forget_class"]),
        forget_train=np.asarray(payload["forget_train"], dtype=np.int64),
        retain_train=np.asarray(payload["retain_train"], dtype=np.int64),
        forget_test=np.asarray(payload["forget_test"], dtype=np.int64),
        retain_test=np.asarray(payload["retain_test"], dtype=np.int64),
    )


def get_or_create_class_split(
    train_labels: np.ndarray | list[int],
    test_labels: np.ndarray | list[int],
    forget_class: int,
    path: str | Path,
) -> tuple[ClassSplit, bool]:
    """Load the split at ``path``, building and saving it if absent.

    A stored split for a *different* class raises rather than being reused: the
    run would unlearn one class while every report named another.

    Returns
    -------
    tuple
        ``(split, created)``.
    """
    target = resolve_path(path)
    if target.is_file():
        split = load_class_split(target)
        if split.forget_class != int(forget_class):
            raise ValueError(
                f"existing split {target} is for class {split.forget_class} "
                f"({split.class_name}) but class {forget_class} "
                f"({CIFAR10_CLASS_NAMES[forget_class]}) was requested. "
                f"Delete the file or fix the config."
            )
        # Independent of what the file claims: check the indices really are
        # that class. A hand-edited file is the failure this catches.
        train = np.asarray(train_labels, dtype=np.int64)
        actual = np.unique(train[split.forget_train])
        if actual.size != 1 or int(actual[0]) != int(forget_class):
            raise ValueError(
                f"{target} does not hold exactly class {forget_class}: its "
                f"forget indices cover labels {actual.tolist()}"
            )
        return split, False

    split = build_class_split(train_labels, test_labels, forget_class)
    save_class_split(split, target)
    return split, True
