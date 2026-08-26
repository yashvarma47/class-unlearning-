"""ResNet-18, CIFAR variant.

The torchvision ResNet-18 is built for 224x224 ImageNet inputs: a 7x7 stride-2
stem followed by a 3x3 stride-2 max-pool downsamples by 4x before the first
residual stage. On 32x32 CIFAR images that discards almost all spatial
information. The standard CIFAR variant -- used by AMUN and by the SalUn /
Unlearn-Sparse line this project draws its operators from -- replaces the stem
with a single 3x3 stride-1 convolution and removes the max-pool.

Module names (``conv1``, ``bn1``, ``layer1`` ... ``layer4``, ``fc``) are kept
identical to torchvision's, because the layer-group registry and hence the
chromosome address parameters by these names.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard two-convolution residual block (ResNet-18/34)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Projection shortcut, needed when the block changes resolution or width.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    """ResNet for 32x32 inputs.

    Parameters
    ----------
    block:
        Residual block type.
    num_blocks:
        Blocks per stage, e.g. ``[2, 2, 2, 2]`` for ResNet-18.
    num_classes:
        Number of output classes.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: list[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        # CIFAR stem: 3x3, stride 1, no max-pool -- resolution stays 32x32.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 32x32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16x16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8x8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialise_weights()

    def _make_layer(
        self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialise_weights(self) -> None:
        """He initialisation for convolutions; standard affine init for BN."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet18_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-18 for CIFAR-sized inputs (~11.2M parameters)."""
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
