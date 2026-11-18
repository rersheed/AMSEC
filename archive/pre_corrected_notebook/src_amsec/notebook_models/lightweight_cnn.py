"""PyTorch port of notebook Lightweight Conv1D stack (cell 4)."""
from __future__ import annotations

from torch import nn

from amsec.models import LightweightCNN


def build_lightweight(input_dim: int, num_classes: int = 2) -> nn.Module:
    del num_classes  # binary head is fixed in LightweightCNN
    return LightweightCNN(input_dim)
