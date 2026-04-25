from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50_binary(pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=pretrained)
    # Binary head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model
