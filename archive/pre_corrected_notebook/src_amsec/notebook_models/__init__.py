"""Notebook-faithful model definitions.

Keras builders require TensorFlow (optional). Trainable local runs use PyTorch
ports that match the recorded layer widths from cells 2/4/22.
"""

from amsec.notebook_models.ensemble_stacking import EnsembleStackingTorch
from amsec.notebook_models.keras_builders import keras_available, notebook_flops

__all__ = ["EnsembleStackingTorch", "keras_available", "notebook_flops"]
