from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.gait_data import feature_columns
from src.predict import extract_trial_signal, predict_trials


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "gaitrec_baseline.joblib"

st.set_page_config(page_title="Pathological Gait Monitor", layout="wide")


@st.cache_resource
def cached_model(path: str):
    return joblib.load(path)


def main() -> None:
    st.title("Pathological Gait & Activity Recognition")
    st.caption(
        "Real-data MVP using GaitRec processed walking trials: healthy vs pathological gait, "
        "with class-level extension."
    )

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Model artifact", str(DEFAULT_MODEL))
        st.radio("Dashboard task", ["Binary anomaly", "Impairment class"], index=0)
        st.markdown(
            "Train first in Colab or locally:\n\n"
            "`python -m src.train --download --rebuild --target target_binary`"
        )

    artifact = None
    model_file = Path(model_path)
    if model_file.exists():
        artifact = cached_model(str(model_file))
        st.sidebar.success("Model loaded")
    else:
        st.sidebar.warning("Model artifact not found yet")

    uploaded = st.file_uploader(
        "Upload a real GaitRec-compatible CSV/parquet table",
        type=["csv", "parquet"],
        help="Use data/processed/gaitrec_features.parquet or a CSV with the same feature columns.",
    )

    if uploaded is None:
        st.info(
            "Upload the processed GaitRec feature table after running the training pipeline. "
            "No synthetic examples are shown here because the project constraint forbids synthetic data."
        )
        return

    if uploaded.name.endswith(".parquet"):
        data = pd.read_parquet(uploaded)
    else:
        data = pd.read_csv(uploaded)

    st.subheader("Input Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trials", f"{len(data):,}")
    c2.metric("Feature columns", f"{len(feature_columns(data)):,}")
    c3.metric("Subjects", f"{data['SUBJECT_ID'].nunique():,}" if "SUBJECT_ID" in data else "unknown")
    c4.metric("Sessions", f"{data['SESSION_ID'].nunique():,}" if "SESSION_ID" in data else "unknown")

    if "target_binary" in data:
        fig = px.histogram(data, x="target_binary", color="target_binary", title="Ground-truth distribution")
        st.plotly_chart(fig, use_container_width=True)

    if artifact is None:
        st.warning("Load a trained model artifact to run predictions.")
        st.dataframe(data.head(50), use_container_width=True)
        return

    pred = predict_trials(data, artifact)

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

    st.subheader("Trial Inspector")
    row_number = st.number_input("Trial row", min_value=0, max_value=max(len(pred) - 1, 0), value=0, step=1)
    row = pred.iloc[int(row_number)]
    id_cols = [
        col
        for col in ["SUBJECT_ID", "SESSION_ID", "TRIAL_ID", "target", "target_binary", "prediction", "confidence"]
        if col in pred
    ]
    st.dataframe(row[id_cols].to_frame("value"), use_container_width=True)

    channels = [
        prefix
        for prefix in ["F_V_left", "F_V_right", "F_AP_left", "F_AP_right", "F_ML_left", "F_ML_right"]
        if any(col.startswith(prefix) for col in pred.columns)
    ]
    if channels:
        channel = st.selectbox("Signal channel", channels)
        signal = extract_trial_signal(row, channel)
        st.plotly_chart(
            px.line(signal, x="stance_percent", y="value", title=f"{channel} over normalized stance"),
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
