from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.gait_data import build_dataset, download_gaitrec, feature_columns


CHANNELS = ["F_V_left", "F_V_right", "F_AP_left", "F_AP_right", "F_ML_left", "F_ML_right"]


if torch is not None:

    class GaitCNN(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(len(CHANNELS), 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, n_classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.net(x))
else:

    class GaitCNN:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for GaitCNN. "
                "Install it with `pip install -r requirements-cnn.txt` or run in Colab with GPU enabled."
            ) from TORCH_IMPORT_ERROR


def dataframe_to_tensor(df: pd.DataFrame) -> np.ndarray:
    tensors = []
    for channel in CHANNELS:
        cols = sorted([col for col in df.columns if col.startswith(channel)])
        if not cols:
            raise ValueError(f"Missing channel columns for {channel}")
        tensors.append(df[cols].to_numpy(dtype=np.float32))
    return np.stack(tensors, axis=1)


def train_cnn(
    data_root: str | Path = "data",
    model_dir: str | Path = "models",
    target: str = "target_binary",
    rebuild: bool = False,
    max_trials: int | None = 5000,
    epochs: int = 12,
    batch_size: int = 256,
) -> dict:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the optional CNN trainer. "
            "Run this in Colab with GPU enabled or install locally with "
            "`pip install -r requirements-cnn.txt`."
        ) from TORCH_IMPORT_ERROR

    data_root = Path(data_root)
    model_dir = Path(model_dir)
    dataset_path = data_root / "processed" / "gaitrec_features.parquet"
    if rebuild or not dataset_path.exists():
        build_dataset(data_root)

    df = pd.read_parquet(dataset_path).dropna(subset=[target])
    if max_trials and len(df) > max_trials:
        df = (
            df.groupby(target, group_keys=False)
            .apply(lambda x: x.sample(max(1, int(max_trials * len(x) / len(df))), random_state=42))
            .sample(frac=1, random_state=42)
            .head(max_trials)
        )

    X = dataframe_to_tensor(df)
    scaler = StandardScaler()
    n, c, t = X.shape
    X = scaler.fit_transform(X.transpose(0, 2, 1).reshape(-1, c)).reshape(n, t, c).transpose(0, 2, 1)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target].astype(str))

    if "SUBJECT_ID" in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["SUBJECT_ID"]))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx], dtype=torch.long))
    test_x = torch.tensor(X[test_idx]).to(device)
    test_y = torch.tensor(y[test_idx], dtype=torch.long).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = GaitCNN(len(encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(test_x).argmax(dim=1).cpu().numpy()
    y_true = test_y.cpu().numpy()

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "gaitrec_cnn.pt"
    metrics_path = model_dir / "gaitrec_cnn_metrics.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": encoder.classes_.tolist(),
            "channels": CHANNELS,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "target": target,
        },
        model_path,
    )
    metrics = {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "classes": encoder.classes_.tolist(),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "classification_report": classification_report(y_true, pred, target_names=encoder.classes_, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
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
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-trials", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.download:
        download_gaitrec(args.data_root)
    print(
        json.dumps(
            train_cnn(
                data_root=args.data_root,
                model_dir=args.model_dir,
                target=args.target,
                rebuild=args.rebuild,
                max_trials=args.max_trials,
                epochs=args.epochs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
