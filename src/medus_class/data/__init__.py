"""CIFAR-10 data and class-level splits.

This project supports ONE split mode: ``D_f`` is a whole CIFAR-10 class. See
``class_split.py`` for why random-instance splitting is deliberately absent.
"""

from medus_class.data.cifar10 import (
    CIFAR10Bundle,
    ClassLoaders,
    build_class_loaders,
    build_transforms,
    load_cifar10,
    make_loader,
)
from medus_class.data.class_split import (
    CIFAR10_CLASS_NAMES,
    DEFAULT_FORGET_CLASS,
    ClassSplit,
    build_class_split,
    get_or_create_class_split,
    load_class_split,
    save_class_split,
)

__all__ = [
    "CIFAR10Bundle",
    "ClassLoaders",
    "build_class_loaders",
    "build_transforms",
    "load_cifar10",
    "make_loader",
    "CIFAR10_CLASS_NAMES",
    "DEFAULT_FORGET_CLASS",
    "ClassSplit",
    "build_class_split",
    "get_or_create_class_split",
    "load_class_split",
    "save_class_split",
]
