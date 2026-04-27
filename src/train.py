from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.gait_data import build_dataset, download_gaitrec, feature_columns


def load_or_build_dataset(data_root: Path, rebuild: bool) -> Path:
    path = data_root / "processed" / "gaitrec_features.parquet"
    if rebuild or not path.exists():
        build_dataset(data_root)
    return path


def train_baseline(
    data_root: str | Path = "data",
    model_dir: str | Path = "models",
    target: str = "target_binary",
    model_type: str = "histgb",
    rebuild: bool = False,
) -> dict:
    data_root = Path(data_root)
    model_dir = Path(model_dir)
    dataset_path = load_or_build_dataset(data_root, rebuild)
    df = pd.read_parquet(dataset_path)
    feats = feature_columns(df)
    df = df.dropna(subset=[target])
    X = df[feats]
    y_raw = df[target].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    if "split" in df.columns and df["split"].str.contains("TEST", na=False).any():
        test_mask = df["split"].str.contains("TEST", na=False)
        train_idx = df.index[~test_mask]
        test_idx = df.index[test_mask]
    elif "SUBJECT_ID" in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_pos, test_pos = next(splitter.split(X, y, groups=df["SUBJECT_ID"]))
        train_idx = df.index[train_pos]
        test_idx = df.index[test_pos]
    else:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)

    if model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=350,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        steps = [("imputer", SimpleImputer()), ("classifier", classifier)]
    else:
        classifier = HistGradientBoostingClassifier(random_state=42, learning_rate=0.06, max_iter=220)
        steps = [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("classifier", classifier)]

    pipeline = Pipeline(steps)
    pipeline.fit(X.loc[train_idx], y[df.index.get_indexer(train_idx)])
    pred = pipeline.predict(X.loc[test_idx])
    y_test = y[df.index.get_indexer(test_idx)]

    report = classification_report(y_test, pred, target_names=encoder.classes_, output_dict=True)
    cm = confusion_matrix(y_test, pred).tolist()

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "label_encoder": encoder,
        "feature_columns": feats,
        "target": target,
    }
    model_path = model_dir / "gaitrec_baseline.joblib"
    metrics_path = model_dir / "gaitrec_metrics.json"
    joblib.dump(artifact, model_path)
    metrics = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "target": target,
        "classes": encoder.classes_.tolist(),
        "classification_report": report,
        "confusion_matrix": cm,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--target", default="target_binary", choices=["target_binary", "target"])
    parser.add_argument("--model-type", default="histgb", choices=["histgb", "rf"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_gaitrec(args.data_root)
    metrics = train_baseline(
        data_root=args.data_root,
        model_dir=args.model_dir,
        target=args.target,
        model_type=args.model_type,
        rebuild=args.rebuild,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
