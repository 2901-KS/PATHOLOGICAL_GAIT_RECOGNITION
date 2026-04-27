from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: str | Path = "models/gaitrec_baseline.joblib") -> dict:
    return joblib.load(model_path)


def predict_trials(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    features = artifact["feature_columns"]
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            "Input is missing model feature columns. "
            f"First missing columns: {missing[:8]} / total missing={len(missing)}"
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


def extract_trial_signal(row: pd.Series, channel_prefix: str) -> pd.DataFrame:
    cols = sorted([col for col in row.index if col.startswith(channel_prefix)])
    return pd.DataFrame({"stance_percent": range(len(cols)), "value": [row[col] for col in cols]})
