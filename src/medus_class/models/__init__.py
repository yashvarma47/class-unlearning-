"""ResNet-18 and the layer-group registry.

Only the CIFAR ResNet-18 is carried over. The predecessor project also defined a
VGG19 variant that was built but never trained; it is left behind rather than
copied forward untested.

Layer groups (L = 6): stem, layer1, layer2, layer3, layer4, fc. The chromosome
addresses these blocks, not individual PyTorch parameters.
"""

from typing import Any

import torch.nn as nn

from medus_class.models.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    read_metadata,
    save_checkpoint,
)
from medus_class.models.layer_groups import LayerGroup, LayerGroupRegistry, build_registry
from medus_class.models.resnet18 import ResNetCIFAR, resnet18_cifar

MODEL_REGISTRY = {"resnet18": resnet18_cifar}


def build_model(model_cfg: dict[str, Any], num_classes: int) -> nn.Module:
    """Instantiate the model named by a model config."""
    name = model_cfg["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


__all__ = [
    "MODEL_REGISTRY", "build_model", "count_parameters",
    "CheckpointMetadata", "load_checkpoint", "read_metadata", "save_checkpoint",
    "LayerGroup", "LayerGroupRegistry", "build_registry",
    "ResNetCIFAR", "resnet18_cifar",
]
