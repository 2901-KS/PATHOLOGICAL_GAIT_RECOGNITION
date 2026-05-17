from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.gait_data import build_dataset, download_gaitrec, feature_columns
from src.predict import extract_group_mean_signal, extract_trial_signal, load_model, predict_trials
from src.train import train_baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "gaitrec_baseline.joblib"
DEFAULT_CNN_MODEL = ROOT / "models" / "gaitrec_cnn.pt"
DEFAULT_DATASET = ROOT / "data" / "processed" / "gaitrec_features.parquet"

st.set_page_config(page_title="Pathological Gait Monitor", layout="wide")


@st.cache_resource
def cached_model(path: str):
    return load_model(path)


def read_table(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    return pd.read_csv(uploaded_file)


def show_metrics(metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "accuracy" in metrics:
        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    if "macro_f1" in metrics:
        st.sidebar.metric("Macro F1", f"{metrics['macro_f1']:.3f}")


def metrics_for_model(model_path: Path) -> Path:
    if model_path.suffix == ".pt":
        return model_path.with_name("gaitrec_cnn_metrics.json")
    return model_path.with_name("gaitrec_metrics.json")


def load_metrics_table(model_dir: Path) -> pd.DataFrame:
    rows = []
    for name, path in [
        ("HistGB / RF baseline", model_dir / "gaitrec_metrics.json"),
        ("1D CNN", model_dir / "gaitrec_cnn_metrics.json"),
    ]:
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "train_trials": metrics.get("n_train"),
                "test_trials": metrics.get("n_test"),
                "device": metrics.get("device", "cpu/sklearn"),
            }
        )
    return pd.DataFrame(rows)


def load_metrics_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion_matrix(metrics: dict, title: str) -> None:
    matrix = metrics.get("confusion_matrix")
    classes = metrics.get("classes")
    if not matrix or not classes:
        return
    cm_df = pd.DataFrame(matrix, index=classes, columns=classes)
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def channel_pair_signal(row: pd.Series, base_channel: str) -> pd.DataFrame:
    frames = []
    for side in ["left", "right"]:
        prefix = f"{base_channel}_{side}"
        cols = sorted([col for col in row.index if col.startswith(prefix)])
        frames.extend(
            {"stance_percent": i, "value": row[col], "side": side}
            for i, col in enumerate(cols)
        )
    return pd.DataFrame(frames)


def main() -> None:
    st.title("Pathological Gait & Activity Recognition")
    st.caption(
        "Real-data MVP using GaitRec processed walking trials: healthy vs pathological gait, "
        "with class-level extension."
    )

    with st.sidebar:
        st.header("Model")
        model_choice = st.selectbox(
            "Model type",
            ["Classical ML (.joblib)", "1D CNN (.pt)", "Custom path"],
        )
        if model_choice.startswith("Classical"):
            default_path = DEFAULT_MODEL
        elif model_choice.startswith("1D CNN"):
            default_path = DEFAULT_CNN_MODEL
        else:
            available_models = [path for path in [DEFAULT_MODEL, DEFAULT_CNN_MODEL] if path.exists()]
            default_path = available_models[0] if available_models else DEFAULT_MODEL
        model_path = st.text_input("Model artifact", str(default_path))
        st.radio("Dashboard task", ["Binary anomaly", "Impairment class"], index=0)
        st.markdown(
            "Train first in Colab or locally:\n\n"
            "`python -m src.train --download --rebuild --target target_binary --max-trials 5000`"
        )
        with st.expander("One-click local bootstrap"):
            max_trials = st.number_input("Training sample size", min_value=500, max_value=50000, value=5000, step=500)
            if st.button("Download, build, train baseline"):
                with st.spinner("Downloading/building GaitRec and training baseline. This can take a while."):
                    download_gaitrec(ROOT / "data")
                    build_dataset(ROOT / "data")
                    train_baseline(ROOT / "data", ROOT / "models", max_trials=int(max_trials), rebuild=False)
                cached_model.clear()
                st.success("Baseline model trained.")

    artifact = None
    model_file = Path(model_path)
    if model_file.exists():
        try:
            artifact = cached_model(str(model_file))
            st.sidebar.success(f"{artifact.get('model_type', 'tabular').upper()} model loaded")
            show_metrics(metrics_for_model(model_file))
        except Exception as exc:
            st.sidebar.error(f"Could not load model: {exc}")
    else:
        st.sidebar.warning("Model artifact not found yet")

    uploaded = st.file_uploader(
        "Upload a real GaitRec-compatible CSV/parquet table, or use the processed local dataset",
        type=["csv", "parquet"],
        help="Use data/processed/gaitrec_features.parquet or a CSV with the same feature columns.",
    )

    data = None
    if uploaded is not None:
        try:
            data = read_table(uploaded)
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
            return
    elif DEFAULT_DATASET.exists():
        data = pd.read_parquet(DEFAULT_DATASET)
        st.success(f"Loaded local processed dataset: {DEFAULT_DATASET}")

    if data is None:
        st.info(
            "Upload the processed GaitRec feature table after running the training pipeline. "
            "No synthetic examples are shown here because the project constraint forbids synthetic data."
        )
        return

    st.subheader("Input Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trials", f"{len(data):,}")
    c2.metric("Feature columns", f"{len(feature_columns(data)):,}")
    c3.metric("Subjects", f"{data['SUBJECT_ID'].nunique():,}" if "SUBJECT_ID" in data else "unknown")
    c4.metric("Sessions", f"{data['SESSION_ID'].nunique():,}" if "SESSION_ID" in data else "unknown")

    if "target_binary" in data:
        fig = px.histogram(data, x="target_binary", color="target_binary", title="Ground-truth distribution")
        st.plotly_chart(fig, use_container_width=True)

    comparison = load_metrics_table(ROOT / "models")
    if not comparison.empty:
        st.subheader("Model Comparison")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(comparison, use_container_width=True)
        with c2:
            metric_rows = comparison.melt(
                id_vars=["model"],
                value_vars=[col for col in ["accuracy", "macro_f1"] if col in comparison],
                var_name="metric",
                value_name="score",
            ).dropna()
            st.plotly_chart(px.bar(metric_rows, x="model", y="score", color="metric", barmode="group"), use_container_width=True)
        cm1, cm2 = st.columns(2)
        with cm1:
            plot_confusion_matrix(load_metrics_json(ROOT / "models" / "gaitrec_metrics.json"), "Baseline confusion matrix")
        with cm2:
            plot_confusion_matrix(load_metrics_json(ROOT / "models" / "gaitrec_cnn_metrics.json"), "CNN confusion matrix")

    if artifact is None:
        st.warning("Load a trained model artifact to run predictions.")
        st.dataframe(data.head(50), use_container_width=True)
        return

    try:
        pred = predict_trials(data, artifact)
    except ValueError as exc:
        st.error(str(exc))
        st.dataframe(data.head(50), use_container_width=True)
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Predictions")
    pred_counts = pred["prediction"].value_counts().reset_index()
    pred_counts.columns = ["prediction", "count"]
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(pred_counts, use_container_width=True)
    with right:
        st.plotly_chart(px.bar(pred_counts, x="prediction", y="count", color="prediction"), use_container_width=True)

    if "confidence" in pred:
        st.plotly_chart(
            px.histogram(pred, x="confidence", nbins=30, title="Prediction confidence"),
            use_container_width=True,
        )

    st.subheader("Gait Curve Comparison")
    channels = [
        prefix
        for prefix in ["F_V_left", "F_V_right", "F_AP_left", "F_AP_right", "F_ML_left", "F_ML_right"]
        if any(col.startswith(prefix) for col in pred.columns)
    ]
    if channels:
        comparison_channel = st.selectbox("Comparison channel", channels, key="comparison_channel")
        grouping_options = [col for col in ["target_binary", "target", "prediction"] if col in pred]
        group_col = st.selectbox("Compare by", grouping_options, index=len(grouping_options) - 1)
        mean_signal = extract_group_mean_signal(pred, comparison_channel, group_col)
        st.plotly_chart(
            px.line(
                mean_signal,
                x="stance_percent",
                y="value",
                color=group_col,
                title=f"Average {comparison_channel} curve",
            ),
            use_container_width=True,
        )

    st.subheader("Trial Inspector")
    row_number = st.number_input("Trial row", min_value=0, max_value=max(len(pred) - 1, 0), value=0, step=1)
    row = pred.iloc[int(row_number)]
    id_cols = [
        col
        for col in ["SUBJECT_ID", "SESSION_ID", "TRIAL_ID", "target", "target_binary", "prediction", "confidence"]
        if col in pred
    ]
    st.dataframe(row[id_cols].to_frame("value"), use_container_width=True)

    if channels:
        channel = st.selectbox("Signal channel", channels, key="trial_channel")
        signal = extract_trial_signal(row, channel)
        st.plotly_chart(
            px.line(signal, x="stance_percent", y="value", title=f"{channel} over normalized stance"),
            use_container_width=True,
        )
        base_options = [base for base in ["F_V", "F_AP", "F_ML"] if f"{base}_left" in channels and f"{base}_right" in channels]
        if base_options:
            base_channel = st.selectbox("Left/right comparison", base_options)
            pair_signal = channel_pair_signal(row, base_channel)
            st.plotly_chart(
                px.line(
                    pair_signal,
                    x="stance_percent",
                    y="value",
                    color="side",
                    title=f"{base_channel} left vs right",
                ),
                use_container_width=True,
            )

    st.download_button(
        "Download predictions",
        pred.to_csv(index=False).encode("utf-8"),
        file_name="gait_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
