"""MobileNet model factory for Sudoku digit recognition."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def build_model(num_classes: int = 10) -> nn.Module:
    """Build a grayscale MobileNetV3 classifier with a custom output head."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    model = mobilenet_v3_small(weights=None)

    first_conv = model.features[0][0]
    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError("Unexpected MobileNet stem layout")

    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )

    classifier_head = model.classifier[-1]
    if not isinstance(classifier_head, nn.Linear):
        raise TypeError("Unexpected MobileNet classifier layout")

    model.classifier[-1] = nn.Linear(classifier_head.in_features, num_classes)
    return model
