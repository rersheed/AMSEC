"""CNN–BiLSTM dual-input model from notebook cell 22 (PyTorch)."""
from __future__ import annotations

import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128, num_layers: int = 2, dropout_rate: float = 0.72):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv1 = nn.Sequential(nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(
            128,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        x = self.conv2(self.conv1(x)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.dropout(self.relu(self.fc_out(context)))


class DualInputModel(nn.Module):
    def __init__(self, static_dim: int, vocab_size: int, hidden_dim: int = 128, dropout_rate: float = 0.72):
        super().__init__()
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.seq_branch = CNNBiLSTM(vocab_size, hidden_dim, num_layers=2, dropout_rate=dropout_rate)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
        )

    def forward(self, static_features: torch.Tensor, seq_features: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.static_branch(static_features), self.seq_branch(seq_features)], dim=1))
