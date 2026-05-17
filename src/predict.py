from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train_cnn import CHANNELS, GaitCNN, TORCH_IMPORT_ERROR, dataframe_to_tensor, torch


def load_model(model_path: str | Path = "models/gaitrec_baseline.joblib") -> dict:
    path = Path(model_path)
    if path.suffix == ".pt":
        if torch is None:
            raise RuntimeError(
                "This is a CNN model, but PyTorch is not installed. "
                "Install it with `pip install -r requirements-cnn.txt` or run inference in Colab."
            ) from TORCH_IMPORT_ERROR
        checkpoint = torch.load(path, map_location="cpu")
        model = GaitCNN(len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return {"model_type": "cnn", "checkpoint": checkpoint, "model": model}

    artifact = joblib.load(path)
    artifact["model_type"] = artifact.get("model_type", "tabular")
    return artifact


def predict_tabular(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    features = artifact["feature_columns"]
    missing = [col for col in features if col not in df.columns]
    if missing:
        present_features = [col for col in features if col in df.columns]
        raise ValueError(
            "Input is missing model feature columns. "
            f"Found {len(present_features)} of {len(features)} expected features. "
            f"First missing columns: {missing[:8]} / total missing={len(missing)}. "
            "Upload data/processed/gaitrec_features.parquet or a table created by src.gait_data.build_dataset()."
        )

    pipeline = artifact["pipeline"]
    encoder = artifact["label_encoder"]
    preds = pipeline.predict(df[features])
    labels = encoder.inverse_transform(preds)
    result = df.copy()
    result["prediction"] = labels

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df[features])
        for i, class_name in enumerate(encoder.classes_):
            result[f"prob_{class_name}"] = proba[:, i]
        result["confidence"] = proba.max(axis=1)
    return result


def predict_cnn(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for CNN inference. "
            "Install it with `pip install -r requirements-cnn.txt` or use the sklearn `.joblib` model."
        ) from TORCH_IMPORT_ERROR

    checkpoint = artifact["checkpoint"]
    missing_channels = [channel for channel in CHANNELS if not any(col.startswith(channel) for col in df.columns)]
    if missing_channels:
        raise ValueError(
            "Input is missing CNN sequence channels. "
            f"Missing channel groups: {missing_channels}. "
            "Upload a table created by src.gait_data.build_dataset()."
        )

    X = dataframe_to_tensor(df)
    n, c, t = X.shape
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    X = ((X.transpose(0, 2, 1).reshape(-1, c) - mean) / scale).reshape(n, t, c).transpose(0, 2, 1)

    model = artifact["model"]
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    class_names = checkpoint["classes"]
    pred_idx = proba.argmax(axis=1)

    result = df.copy()
    result["prediction"] = [class_names[i] for i in pred_idx]
    for i, class_name in enumerate(class_names):
        result[f"prob_{class_name}"] = proba[:, i]
    result["confidence"] = proba.max(axis=1)
    return result


def predict_trials(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    if artifact.get("model_type") == "cnn":
        return predict_cnn(df, artifact)
    return predict_tabular(df, artifact)


def extract_trial_signal(row: pd.Series, channel_prefix: str) -> pd.DataFrame:
    cols = sorted([col for col in row.index if col.startswith(channel_prefix)])
    return pd.DataFrame({"stance_percent": range(len(cols)), "value": [row[col] for col in cols]})


def extract_group_mean_signal(df: pd.DataFrame, channel_prefix: str, group_col: str) -> pd.DataFrame:
    cols = sorted([col for col in df.columns if col.startswith(channel_prefix)])
    if not cols or group_col not in df:
        return pd.DataFrame(columns=["stance_percent", "value", group_col])
    rows = []
    for group_name, group in df.groupby(group_col):
        values = group[cols].mean(axis=0)
        rows.extend(
            {"stance_percent": i, "value": value, group_col: str(group_name)}
            for i, value in enumerate(values.to_numpy())
        )
    return pd.DataFrame(rows)
