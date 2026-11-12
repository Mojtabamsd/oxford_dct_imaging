"""CNN models and helpers for training them."""

from __future__ import annotations

import random

import numpy as np


def get_device(name: str = "auto"):
    """'auto' picks a CUDA GPU, then Apple MPS, then CPU. Or pass 'cuda', 'cuda:1', 'mps', 'cpu'."""
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


ARCHITECTURES = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """A torchvision classifier with its last layer resized to ``num_classes``.

    ``pretrained`` loads ImageNet weights (downloaded once, ~45-100 MB).
    """
    import torch.nn as nn
    from torchvision import models

    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {ARCHITECTURES}")
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, arch)(weights=weights)

    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith(("efficientnet", "convnext")):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
