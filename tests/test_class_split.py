"""The class split must be exact. Everything downstream assumes it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medus_class.data.class_split import (  # noqa: E402
    CIFAR10_CLASS_NAMES,
    ClassSplit,
    build_class_split,
    get_or_create_class_split,
    load_class_split,
    save_class_split,
)


@pytest.fixture
def balanced_labels():
    """10 classes x 20 train, 10 classes x 4 test -- CIFAR-10's shape in miniature."""
    train = np.repeat(np.arange(10), 20)
    test = np.repeat(np.arange(10), 4)
    return train, test


def test_forget_set_is_exactly_one_class(balanced_labels):
    train, test = balanced_labels
    split = build_class_split(train, test, forget_class=6)

    assert set(train[split.forget_train].tolist()) == {6}
    assert set(test[split.forget_test].tolist()) == {6}


def test_retain_set_excludes_the_forget_class(balanced_labels):
    train, test = balanced_labels
    split = build_class_split(train, test, forget_class=6)

    assert 6 not in set(train[split.retain_train].tolist())
    assert 6 not in set(test[split.retain_test].tolist())


def test_partitions_are_complete_and_disjoint(balanced_labels):
    train, test = balanced_labels
    split = build_class_split(train, test, forget_class=6)

    assert split.forget_train.size + split.retain_train.size == train.size
    assert split.forget_test.size + split.retain_test.size == test.size
    assert np.intersect1d(split.forget_train, split.retain_train).size == 0
    assert np.intersect1d(split.forget_test, split.retain_test).size == 0


def test_split_is_deterministic(balanced_labels):
    """No seed, no sampling -- two builds must be identical."""
    train, test = balanced_labels
    a = build_class_split(train, test, forget_class=3)
    b = build_class_split(train, test, forget_class=3)

    assert np.array_equal(a.forget_train, b.forget_train)
    assert np.array_equal(a.retain_test, b.retain_test)


@pytest.mark.parametrize("bad_class", [-1, 10, 99])
def test_rejects_invalid_class(balanced_labels, bad_class):
    train, test = balanced_labels
    with pytest.raises(ValueError, match="CIFAR-10 label"):
        build_class_split(train, test, forget_class=bad_class)


def test_rejects_absent_class(balanced_labels):
    train, test = balanced_labels
    train = train[train != 6]  # class 6 no longer present in train
    with pytest.raises(ValueError, match="no training samples"):
        build_class_split(train, test, forget_class=6)


def test_validate_catches_an_overlapping_partition():
    """The invariant is enforced on construction, not merely documented."""
    with pytest.raises(ValueError, match="overlap"):
        ClassSplit(
            forget_class=0,
            forget_train=np.array([0, 1, 2]),
            retain_train=np.array([2, 3, 4]),   # 2 is in both
            forget_test=np.array([0]),
            retain_test=np.array([1]),
        )


def test_roundtrip_through_disk(balanced_labels, tmp_path):
    train, test = balanced_labels
    split = build_class_split(train, test, forget_class=6)
    path = tmp_path / "split.json"
    save_class_split(split, path)

    loaded = load_class_split(path)
    assert loaded.forget_class == 6
    assert np.array_equal(loaded.forget_train, split.forget_train)
    assert np.array_equal(loaded.retain_test, split.retain_test)


def test_reuses_an_existing_split(balanced_labels, tmp_path):
    train, test = balanced_labels
    path = tmp_path / "split.json"

    first, created_first = get_or_create_class_split(train, test, 6, path)
    second, created_second = get_or_create_class_split(train, test, 6, path)

    assert created_first is True
    assert created_second is False
    assert np.array_equal(first.forget_train, second.forget_train)


def test_refuses_a_split_built_for_another_class(balanced_labels, tmp_path):
    """A run must not unlearn one class while reporting another."""
    train, test = balanced_labels
    path = tmp_path / "split.json"
    get_or_create_class_split(train, test, 6, path)

    with pytest.raises(ValueError, match="is for class 6"):
        get_or_create_class_split(train, test, 3, path)


def test_detects_a_tampered_split_file(balanced_labels, tmp_path):
    """The stored indices are checked against the LABELS, not just the header.

    The file is rewritten to hold a valid, complete, disjoint partition -- of the
    wrong class. That passes every structural check, so only the label check can
    catch it, which is the point of having one.
    """
    train, test = balanced_labels
    path = tmp_path / "split.json"
    get_or_create_class_split(train, test, 6, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    wrong = np.flatnonzero(train == 0)
    payload["forget_train"] = wrong.tolist()
    payload["retain_train"] = np.setdiff1d(np.arange(train.size), wrong).tolist()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not hold exactly class"):
        get_or_create_class_split(train, test, 6, path)


def test_structural_damage_is_caught_before_the_label_check(balanced_labels, tmp_path):
    """A file that is not a partition at all fails on construction."""
    train, test = balanced_labels
    path = tmp_path / "split.json"
    get_or_create_class_split(train, test, 6, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["forget_train"] = [0, 1, 2]      # now overlapping retain_train
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        get_or_create_class_split(train, test, 6, path)


def test_class_name_matches_the_label():
    assert CIFAR10_CLASS_NAMES[6] == "frog"
    assert CIFAR10_CLASS_NAMES[0] == "airplane"
