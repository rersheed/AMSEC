"""PyTorch ports of notebook MobileNetV4-style Conv1D (+ attention)."""
from __future__ import annotations

import torch
from torch import nn

from amsec.models import MobileNetV4Style, batch_norm


class MobileNetV4AttentionTorch(nn.Module):
    """Cell-2 attention variant: MultiHeadAttention on 64-d dense features."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        base = MobileNetV4Style(input_dim)
        self.input_norm = base.net.input_norm
        self.conv = base.net.conv
        self.head = base.net.head  # ends at 64-d features after dropout
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.post = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            batch_norm(64),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x).unsqueeze(1)
        h = self.conv(h)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        feats = self.head(h)
        attn_out, _ = self.attn(feats.unsqueeze(1), feats.unsqueeze(1), feats.unsqueeze(1))
        attn_out = self.norm(attn_out.squeeze(1))
        return self.post(torch.cat([feats, attn_out], dim=1))


def build_mobilenet(input_dim: int, use_attention: bool = False) -> nn.Module:
    if use_attention:
        return MobileNetV4AttentionTorch(input_dim)
    return MobileNetV4Style(input_dim)
