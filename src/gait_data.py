from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


FIGSHARE_ARTICLE_ID = "12162300"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

SIGNAL_FILES = [
    "GRF_F_V_PRO_left.csv",
    "GRF_F_V_PRO_right.csv",
    "GRF_F_AP_PRO_left.csv",
    "GRF_F_AP_PRO_right.csv",
    "GRF_F_ML_PRO_left.csv",
    "GRF_F_ML_PRO_right.csv",
]
METADATA_FILE = "GRF_metadata.csv"
METADATA_ALIASES = {"GRF_metadata.csv", "GRF-metadata.csv"}


@dataclass(frozen=True)
class GaitRecPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw" / "gaitrec"

    @property
    def processed(self) -> Path:
        return self.root / "processed"


def fetch_figshare_file_index() -> list[dict]:
    response = requests.get(FIGSHARE_API_URL, timeout=30)
    response.raise_for_status()
    article = response.json()
    return article.get("files", [])


def download_gaitrec(data_root: str | Path = "data", files: Iterable[str] | None = None) -> None:
    """Download selected GaitRec CSV files from Figshare.

    The full article is large, so the default fetches only processed GRF files needed by
    this project plus metadata.
    """
    paths = GaitRecPaths(Path(data_root))
    paths.raw.mkdir(parents=True, exist_ok=True)
    wanted = set(files or [METADATA_FILE, *SIGNAL_FILES])
    index = fetch_figshare_file_index()
    (paths.raw / "figshare_files.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    by_name = {item["name"]: item for item in index}

    def resolve_file(name: str) -> dict:
        if name in by_name:
            return by_name[name]
        if name in METADATA_ALIASES:
            for alias in METADATA_ALIASES:
                if alias in by_name:
                    return by_name[alias]
        normalized = name.lower().replace("-", "_")
        for hosted_name, item in by_name.items():
            if hosted_name.lower().replace("-", "_") == normalized:
                return item
        raise FileNotFoundError(f"Figshare article is missing expected file: {name}")

    for name in sorted(wanted):
        item = resolve_file(name)
        target = paths.raw / name
        if target.exists() and target.stat().st_size == item.get("size"):
            continue
        url = item["download_url"]
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with target.open("wb") as handle, tqdm(
                total=total, unit="B", unit_scale=True, desc=name
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        bar.update(len(chunk))


def _read_signal(path: Path, channel_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_cols = ["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]
    value_cols = [col for col in df.columns if col not in id_cols]
    renamed = {col: f"{channel_name}_{i:03d}" for i, col in enumerate(value_cols)}
    return df.rename(columns=renamed)


def _find_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {col.lower().replace("-", "_"): col for col in columns}
    for candidate in candidates:
        key = candidate.lower().replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None


def _derive_label(row: pd.Series) -> str:
    label = str(row.get("CLASS", row.get("CLASS_LABEL", row.get("LABEL", "")))).strip()
    if label.upper() == "HC":
        return "HC"
    if "_" in label:
        return label.split("_", 1)[0]
    if label and label[0].upper() in {"H", "K", "A", "C"}:
        return label[0].upper()
    return label or "UNKNOWN"


def build_dataset(data_root: str | Path = "data", output_name: str = "gaitrec_features.parquet") -> Path:
    """Merge processed GaitRec force channels into one ML-ready table."""
    paths = GaitRecPaths(Path(data_root))
    frames: list[pd.DataFrame] = []
    for filename in SIGNAL_FILES:
        channel = filename.replace("GRF_", "").replace("_PRO", "").replace(".csv", "")
        frames.append(_read_signal(paths.raw / filename, channel))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"], how="inner")

    metadata_path = paths.raw / METADATA_FILE
    if not metadata_path.exists():
        candidates = list(paths.raw.glob("*metadata*.csv")) + list(paths.raw.glob("*Metadata*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Missing metadata file under {paths.raw}")
        metadata_path = candidates[0]

    metadata = pd.read_csv(metadata_path)
    split_col = _find_col(metadata.columns, ["DATASET", "SET", "SPLIT", "TRAIN_TEST_SPLIT"])
    label_col = _find_col(metadata.columns, ["CLASS", "CLASS_LABEL", "LABEL"])

    merged = merged.merge(metadata, on=["SUBJECT_ID", "SESSION_ID"], how="left")
    if label_col and label_col != "CLASS":
        merged["CLASS"] = merged[label_col]
    merged["target"] = merged.apply(_derive_label, axis=1)
    merged["target_binary"] = np.where(merged["target"].eq("HC"), "healthy", "pathological")
    if split_col:
        merged["split"] = merged[split_col].astype(str).str.upper()
    else:
        merged["split"] = "UNSPECIFIED"

    paths.processed.mkdir(parents=True, exist_ok=True)
    out = paths.processed / output_name
    merged.to_parquet(out, index=False)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if any(col.startswith(prefix) for prefix in ("F_V", "F_AP", "F_ML"))]
