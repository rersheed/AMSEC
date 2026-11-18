"""Train a leakage-corrected local AMSEC ensemble and write the inference bundle.

This is not a replay of the Kaggle 30-epoch run. Deviations are written into
the model manifest. Historical notebook metrics are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.models import LightweightCNN, MobileNetV4Style, keras_style_parameter_count
from amsec.selection import GeneticFeatureSelector

DATA_PATH = ROOT / "others files that can help" / "NF-UQ-NIDS-Dataset.csv"
ARTIFACTS = ROOT / "artifacts"

MAX_ROWS = 150_000
GA_SAMPLE = 8_000
GA_POPULATION = 12
GA_GENERATIONS = 5
GA_TREES = 25
EPOCHS = 8
BATCH_SIZE = 128
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_nids(path: Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    preview = pd.read_csv(path, nrows=5)
    label_col = next((c for c in preview.columns if c.lower() in {"label", "attack", "class"}), None)
    if label_col is None:
        raise SystemExit(f"No label column in {path.name}. Columns: {list(preview.columns)}")
    frame = pd.read_csv(path, nrows=MAX_ROWS, low_memory=False)
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    keep = labels.notna()
    frame = frame.loc[keep].drop(columns=[label_col])
    y = (labels.loc[keep].to_numpy() > 0).astype(np.int64)
    numeric = frame.select_dtypes(include=[np.number]).copy()
    missing = int(numeric.isna().sum().sum())
    numeric = numeric.fillna(0.0)
    print(
        f"Loaded {path.name}: rows={len(numeric):,} numeric_features={numeric.shape[1]} "
        f"label={label_col} training_cells_filled_with_zero={missing}",
        flush=True,
    )
    print(f"  normal={(y == 0).sum():,} attack={(y == 1).sum():,}", flush=True)
    return numeric, y, label_col


def choose_features(train: pd.DataFrame, y_train: np.ndarray) -> tuple[list[str], dict]:
    columns = np.array(train.columns)
    values = train.to_numpy(dtype=np.float64)
    variance = VarianceThreshold(threshold=1e-5)
    filtered = variance.fit_transform(values)
    kept = columns[variance.get_support()]
    print(f"Variance threshold kept {len(kept)} of {len(columns)} numeric columns", flush=True)
    sample_n = min(GA_SAMPLE, filtered.shape[0])
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(filtered.shape[0], sample_n, replace=False)
    scaler = RobustScaler()
    sample = scaler.fit_transform(filtered[sample_idx])
    k_mi = min(100, sample.shape[1])
    mi = SelectKBest(mutual_info_classif, k=k_mi)
    reduced = mi.fit_transform(sample, y_train[sample_idx])
    mi_index = np.flatnonzero(mi.get_support())
    print(f"Mutual information retained {reduced.shape[1]} features for the genetic search", flush=True)
    selector = GeneticFeatureSelector(
        reduced,
        y_train[sample_idx],
        population_size=GA_POPULATION,
        generations=GA_GENERATIONS,
        n_estimators=GA_TREES,
        random_state=SEED,
    )
    relative, history = selector.run()
    chosen_pos = mi_index[relative]
    names = [str(kept[i]) for i in chosen_pos]
    scores = pd.Series(mi.scores_, index=[str(c) for c in kept]).sort_values(ascending=False)
    if len(names) < 5:
        for name in scores.index:
            if name not in names:
                names.append(name)
            if len(names) >= 5:
                break
    report = {
        "variance_threshold": 1e-5,
        "features_after_variance_threshold": int(len(kept)),
        "mutual_information_k": int(k_mi),
        "ga_sample_rows": int(sample_n),
        "ga_population": GA_POPULATION,
        "ga_generations": GA_GENERATIONS,
        "ga_trees": GA_TREES,
        "ga_fitness_history": history,
        "selected_features": names,
        "top_mutual_information": [
            {"feature": str(idx), "score": float(val)} for idx, val in scores.head(15).items()
        ],
    }
    print(f"Selected features: {names}", flush=True)
    return names, report


def fit_preprocess(train: pd.DataFrame, other: list[pd.DataFrame]):
    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train.to_numpy(dtype=np.float64))
    power = PowerTransformer(method="yeo-johnson", standardize=True)
    transformed = [np.nan_to_num(power.fit_transform(scaled_train), nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)]
    for frame in other:
        scaled = scaler.transform(frame.to_numpy(dtype=np.float64))
        transformed.append(np.nan_to_num(power.transform(scaled), nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32))
    return scaler, power, transformed


def train_network(module, x_train, y_train, x_val, y_val, lr: float, weight_decay: float, patience_lr: int) -> list[dict]:
    counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float32)
    weights = torch.tensor(len(y_train) / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    best_loss = float("inf")
    wait = 0
    history = []
    for epoch in range(EPOCHS):
        module.train()
        total = 0.0
        seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(module(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(yb)
            seen += len(yb)
        module.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(module(x_val_t), y_val_t).item())
        history.append({"epoch": epoch + 1, "train_loss": total / max(seen, 1), "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        print(f"    epoch {epoch + 1}/{EPOCHS} train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience_lr:
                optimizer.param_groups[0]["lr"] = max(optimizer.param_groups[0]["lr"] * 0.5, 1e-7)
                wait = 0
        if epoch + 1 >= 4 and val_loss > best_loss and (epoch + 1 - int(np.argmin([h["val_loss"] for h in history]))) >= 4:
            print("    early stop", flush=True)
            break
    module.load_state_dict(best_state)
    module.eval()
    return history


def softmax_numpy(module, matrix: np.ndarray) -> np.ndarray:
    module.eval()
    with torch.inference_mode():
        logits = module(torch.from_numpy(matrix))
        return torch.softmax(logits, dim=1).cpu().numpy()


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    attack = probabilities[:, 1]
    predicted = (attack >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, attack)),
        "threshold": 0.5,
        "confusion_matrix": {
            "labels": ["normal", "attack"],
            "matrix": matrix.astype(int).tolist(),
        },
        "support": {"normal": int((y_true == 0).sum()), "attack": int((y_true == 1).sum())},
    }


def main() -> None:
    started = time.time()
    set_seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    features, y, label_col = load_nids(DATA_PATH)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, y, test_size=0.30, random_state=SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    names, selection = choose_features(x_train, y_train)
    scaler, power, (z_train, z_val, z_test) = fit_preprocess(
        x_train.loc[:, names], [x_val.loc[:, names], x_test.loc[:, names]]
    )
    print("Fitting SMOTE-ENN on the training fold only", flush=True)
    resampler = SMOTEENN(random_state=SEED)
    z_bal, y_bal = resampler.fit_resample(z_train, y_train)
    print(f"  resampled train rows={len(y_bal):,} normal={(y_bal == 0).sum():,} attack={(y_bal == 1).sum():,}", flush=True)
    input_dim = z_bal.shape[1]
    mobile = MobileNetV4Style(input_dim)
    light = LightweightCNN(input_dim)
    print(f"MobileNet-style parameter count (Keras-comparable)={keras_style_parameter_count(mobile):,}", flush=True)
    print(f"Lightweight parameter count (Keras-comparable)={keras_style_parameter_count(light):,}", flush=True)
    print("Training MobileNet-style base learner", flush=True)
    mobile_history = train_network(mobile, z_bal, y_bal, z_val, y_val, lr=1.2e-3, weight_decay=5e-5, patience_lr=3)
    print("Training lightweight CNN base learner", flush=True)
    light_history = train_network(light, z_bal, y_bal, z_val, y_val, lr=1.5e-3, weight_decay=1e-4, patience_lr=2)
    meta_x = np.hstack([softmax_numpy(mobile, z_bal), softmax_numpy(light, z_bal)])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, class_weight="balanced")
    meta.fit(meta_x, y_bal)
    if list(meta.classes_) != [0, 1]:
        raise SystemExit(f"Unexpected meta class order: {list(meta.classes_)}")

    def predict(matrix: np.ndarray) -> np.ndarray:
        return meta.predict_proba(np.hstack([softmax_numpy(mobile, matrix), softmax_numpy(light, matrix)]))

    test_proba = predict(z_test)
    metrics = binary_metrics(y_test, test_proba)
    print("Local test metrics:", json.dumps({k: metrics[k] for k in ("accuracy", "f1", "balanced_accuracy", "roc_auc")}), flush=True)
    rng = np.random.RandomState(SEED)
    take = rng.choice(len(x_test), size=min(32, len(x_test)), replace=False)
    parity_raw = x_test.iloc[take].loc[:, names].reset_index(drop=True)
    parity_proba = predict(z_test[take])
    parity_raw.to_csv(ARTIFACTS / "parity_batch.csv", index=False)
    np.save(ARTIFACTS / "parity_proba.npy", parity_proba)

    mobile_path = ARTIFACTS / "mobilenetv4.pt"
    light_path = ARTIFACTS / "lightweight.pt"
    pre_path = ARTIFACTS / "preprocessor.joblib"
    meta_path = ARTIFACTS / "meta_learner.joblib"
    torch.save(mobile.state_dict(), mobile_path)
    torch.save(light.state_dict(), light_path)
    joblib.dump({"scaler": scaler, "power": power, "feature_names": names}, pre_path)
    joblib.dump(meta, meta_path)
    evaluation = {
        "protocol": "leakage_corrected_local_retrain",
        "rows_loaded": int(len(features)),
        "label_column": label_col,
        "split": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)), "resampled_train": int(len(y_bal))},
        "test_metrics": metrics,
        "mobile_history": mobile_history,
        "lightweight_history": light_history,
        "meta_coefficients": {
            "classes": [int(c) for c in meta.classes_],
            "coef": meta.coef_.tolist(),
            "intercept": meta.intercept_.tolist(),
            "feature_order": ["mobilenet_p_normal", "mobilenet_p_attack", "lightweight_p_normal", "lightweight_p_attack"],
        },
        "selection": selection,
        "elapsed_seconds": time.time() - started,
    }
    (ARTIFACTS / "local_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    manifest = {
        "experiment_id": "local_retrain_cell4_stack_smoteenn",
        "status": "local_retrain_not_kaggle_weights",
        "source_notebook": "Final phd-Experiment-android-malware.ipynb",
        "source_cell": 4,
        "claim": "These weights were trained in this workspace. They are not the missing Kaggle HDF5 files and are not claimed to reproduce the notebook's 0.97789 accuracy.",
        "feature_names": names,
        "feature_types": ["float"] * len(names),
        "label_mapping": {"0": "normal", "1": "attack"},
        "decision_threshold": 0.5,
        "meta_feature_order": ["mobilenet_p_normal", "mobilenet_p_attack", "lightweight_p_normal", "lightweight_p_attack"],
        "preprocessor_file": pre_path.name,
        "meta_learner_file": meta_path.name,
        "mobilenet_file": mobile_path.name,
        "lightweight_file": light_path.name,
        "model_files": [mobile_path.name, light_path.name, meta_path.name],
        "preprocessor_files": [pre_path.name],
        "hashes": {
            "preprocessor": sha256(pre_path),
            "meta_learner": sha256(meta_path),
            "mobilenet": sha256(mobile_path),
            "lightweight": sha256(light_path),
            "dataset_prefix": sha256(DATA_PATH),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "architecture_parameter_counts_keras_style": {
            "mobilenetv4_style": keras_style_parameter_count(mobile),
            "lightweight_cnn": keras_style_parameter_count(light),
        },
        "training": {
            "max_rows": MAX_ROWS,
            "epochs_cap": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "resampler": "SMOTEENN",
            "meta_learner": "LogisticRegression",
            "split_before_fit": True,
            "deviations_from_notebook": [
                "TensorFlow 2.20 weights were not in the package and TensorFlow has no Python 3.14 wheel, so the same layer stack was trained in PyTorch.",
                "Preprocessing, feature selection and SMOTE-ENN were fit on the training fold only. The notebook fit them before the split.",
                "The meta-learner still receives the four base-class probabilities in notebook order, but it is fit on resampled training predictions.",
                f"Genetic search used population {GA_POPULATION}, generations {GA_GENERATIONS}, forests of {GA_TREES} trees and {GA_SAMPLE} rows. The notebook used 40, 30, 100 and 20000.",
                f"Training stopped at or before {EPOCHS} epochs. The notebook trained for up to 30.",
                "The notebook's kernel L2 penalties were not added on top of AdamW weight decay.",
                "The five Kaggle feature names were never printed, so this feature set is a new selection.",
            ],
        },
        "historical_reference_not_this_model": {
            "notebook_best_config": "SMOTE-ENN_GA_Ensemble_Stacking",
            "accuracy": 0.97789,
            "f1": 0.97789,
            "balanced_accuracy": 0.97775,
        },
        "local_test_metrics": metrics,
    }
    (ARTIFACTS / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote bundle in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
