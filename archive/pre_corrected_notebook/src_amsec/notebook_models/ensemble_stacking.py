"""Ensemble stacking trainer matching cell-4 protocol (PyTorch bases + LR meta)."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from amsec.models import keras_style_parameter_count
from amsec.notebook_models.keras_builders import notebook_flops
from amsec.notebook_models.lightweight_cnn import build_lightweight
from amsec.notebook_models.mobilenetv4 import build_mobilenet


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    total = counts.sum()
    weights = total / (len(counts) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _train_torch_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    progress: Callable | None = None,
    tag: str = "model",
) -> dict:
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_train).to(device))
    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_state = None
    best_val = float("inf")
    stale = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                v_loss += float(loss.item()) * len(yb)
                v_correct += int((logits.argmax(1) == yb).sum().item())
                v_n += len(yb)
        history["loss"].append(total_loss / max(n, 1))
        history["accuracy"].append(correct / max(n, 1))
        history["val_loss"].append(v_loss / max(v_n, 1))
        history["val_accuracy"].append(v_correct / max(v_n, 1))
        if progress:
            progress({"phase": "train", "tag": tag, "epoch": epoch + 1, "epochs": epochs, "val_loss": history["val_loss"][-1]})
        if history["val_loss"][-1] < best_val - 1e-6:
            best_val = history["val_loss"][-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "training_seconds": time.time() - t0}


@torch.no_grad()
def _predict_proba(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        outs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(outs)


class EnsembleStackingTorch:
    """Cell-4 stacking: MobileNetV4 + Lightweight + LogisticRegression meta-learner."""

    def __init__(self, input_dim: int, num_classes: int = 2):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mobilenet = build_mobilenet(input_dim, use_attention=False)
        self.lightweight = build_lightweight(input_dim)
        self.meta_learner: LogisticRegression | None = None
        self.history: dict = {}
        self.params = keras_style_parameter_count(self.mobilenet) + keras_style_parameter_count(self.lightweight)
        self.flops = notebook_flops(self.params)
        self.training_time = 0.0
        self.backend = "pytorch_notebook_port"

    def train(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs: int = 30,
        batch_size: int = 128,
        progress: Callable | None = None,
        # notebook reproduction fits meta on train predictions (leaky);
        # leakage-controlled can pass oof-style matrices later
        meta_on_train: bool = True,
    ) -> dict:
        t0 = time.time()
        mobile_pack = _train_torch_classifier(
            self.mobilenet,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=1.2e-3,
            weight_decay=5e-5,
            patience=12,
            progress=progress,
            tag="mobilenetv4",
        )
        self.mobilenet = mobile_pack["model"]
        light_pack = _train_torch_classifier(
            self.lightweight,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=1.5e-3,
            weight_decay=1e-4,
            patience=10,
            progress=progress,
            tag="lightweight",
        )
        self.lightweight = light_pack["model"]

        if meta_on_train:
            p_m = _predict_proba(self.mobilenet, x_train)
            p_l = _predict_proba(self.lightweight, x_train)
            x_meta = np.hstack([p_m, p_l])
            y_meta = y_train
        else:
            p_m = _predict_proba(self.mobilenet, x_val)
            p_l = _predict_proba(self.lightweight, x_val)
            x_meta = np.hstack([p_m, p_l])
            y_meta = y_val

        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
        self.meta_learner.fit(x_meta, y_meta)
        self.history = {
            "mobile_history": mobile_pack["history"],
            "light_history": light_pack["history"],
        }
        self.training_time = time.time() - t0
        self.params = keras_style_parameter_count(self.mobilenet) + keras_style_parameter_count(self.lightweight)
        self.flops = notebook_flops(self.params)
        return self.history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.meta_learner is not None
        p_m = _predict_proba(self.mobilenet, x)
        p_l = _predict_proba(self.lightweight, x)
        x_meta = np.hstack([p_m, p_l])
        return self.meta_learner.predict_proba(x_meta)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
