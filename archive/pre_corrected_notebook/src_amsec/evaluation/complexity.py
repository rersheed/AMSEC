from __future__ import annotations

from torch import nn

from amsec.models import keras_style_parameter_count
from amsec.notebook_models.keras_builders import notebook_flops


def parameter_count(module: nn.Module) -> int:
    return keras_style_parameter_count(module)


__all__ = ["parameter_count", "notebook_flops"]
