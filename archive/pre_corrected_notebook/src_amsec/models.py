"""PyTorch reimplementation of the cell-4 Keras base learners.

The Kaggle notebook built these models in TensorFlow 2.20 and saved only
HDF5 files under ``/kaggle/working/models``. Those files are not in the local
package, and TensorFlow does not publish a Python 3.14 wheel. This module
matches the recorded layer stack so a local retrain can be served honestly.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def batch_norm(features: int) -> nn.BatchNorm1d:
    # Keras BatchNorm momentum 0.99 is PyTorch momentum 0.01. Epsilon matches Keras.
    return nn.BatchNorm1d(features, eps=1e-3, momentum=0.01)


class Conv1dSame(nn.Module):
    """Conv1d with TensorFlow ``padding='same'`` for stride 1.

    Odd kernels are padded symmetrically. Even kernels put the extra column
    on the right, which is the TensorFlow Conv ``same`` rule.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        left = total // 2
        right = total - left
        return self.conv(F.pad(x, (left, right)))


def _conv_block(in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        Conv1dSame(in_channels, out_channels, kernel_size),
        nn.ReLU(),
        batch_norm(out_channels),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Dropout(dropout),
    )


class _ConvBackbone(nn.Module):
    def __init__(self, input_dim: int, channels: list[tuple[int, int, float]], dense: list[tuple[int, float]]):
        super().__init__()
        self.input_dim = input_dim
        self.input_norm = batch_norm(input_dim)
        blocks = []
        in_channels = 1
        last_channels = 1
        for out_channels, kernel, drop in channels:
            if drop > 0 and kernel:
                blocks.append(_conv_block(in_channels, out_channels, kernel, drop))
            else:
                blocks.append(
                    nn.Sequential(
                        Conv1dSame(in_channels, out_channels, kernel),
                        nn.ReLU(),
                        batch_norm(out_channels),
                    )
                )
            in_channels = out_channels
            last_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.pool_channels = last_channels * 2
        layers: list[nn.Module] = [nn.Dropout(0.2 if dense[0][0] == 128 else 0.15)]
        in_features = self.pool_channels
        for index, (units, drop) in enumerate(dense):
            layers.append(nn.Linear(in_features, units))
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="relu")
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU())
            layers.append(batch_norm(units))
            layers.append(nn.Dropout(drop))
            in_features = units
        self.head = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) -> (batch, 1, features)
        x = self.input_norm(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        return self.classifier(self.head(pooled))


class MobileNetV4Style(nn.Module):
    """Notebook ``MobileNetV4Model``: three Conv1D stages then two dense layers."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = _ConvBackbone(
            input_dim,
            channels=[(48, 3, 0.12), (64, 3, 0.12), (96, 3, 0.0)],
            dense=[(128, 0.25), (64, 0.2)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightCNN(nn.Module):
    """Notebook ``LightweightModel``: two Conv1D stages then two dense layers."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = _ConvBackbone(
            input_dim,
            channels=[(32, 2, 0.1), (48, 2, 0.1)],
            dense=[(64, 0.2), (32, 0.15)],
        )
        # The notebook uses Dropout(0.15) after the pooled vector, not 0.2.
        # _ConvBackbone selects 0.15 when the first dense width is not 128.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def keras_style_parameter_count(module: nn.Module) -> int:
    """Count weights the way Keras ``count_params`` does.

    Running mean and variance are included. ``num_batches_tracked`` is not.
    """
    total = sum(parameter.numel() for parameter in module.parameters())
    for name, buffer in module.named_buffers():
        if name.endswith("num_batches_tracked"):
            continue
        total += buffer.numel()
    return int(total)
