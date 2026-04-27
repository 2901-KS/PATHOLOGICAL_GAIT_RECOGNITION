# Pathological Gait & Activity Recognition

This is a real-data MVP for detecting abnormal walking patterns outside the notebook environment. It uses **GaitRec** rather than synthetic data or HugaDB.

## Dataset Choice

Primary dataset: **GaitRec, a large-scale ground reaction force dataset of healthy and impaired gait**.

- Published in Scientific Data, 2020.
- Contains 75,732 bilateral walking trials.
- Includes 211 healthy controls and 2,084 patients with hip, knee, ankle, and calcaneus impairments.
- Hosted on Figshare under DOI `10.6084/m9.figshare.12162300`.

This is not a post-stroke hemiplegic IMU dataset; it is a clinically annotated pathological gait dataset. The MVP therefore focuses on abnormal gait recognition from real walking trials, with an explicit path to swap in wearable IMU datasets later.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.train --download --rebuild --target target_binary
streamlit run app.py
```

For multi-class impairment recognition:

```bash
python -m src.train --download --rebuild --target target
```

## Colab GPU Path

1. Upload this folder to Google Drive or GitHub.
2. Open `notebooks/gaitrec_colab.ipynb` in Colab.
3. Select `Runtime > Change runtime type > GPU`.
4. Run the notebook to download GaitRec, build features, train, and export `models/gaitrec_baseline.joblib`.
5. Download the model artifact and run the dashboard locally:

```bash
streamlit run app.py
```

## Files

- `src/gait_data.py` downloads selected GaitRec files and builds the ML table.
- `src/train.py` trains a baseline classifier and writes metrics.
- `src/predict.py` loads a trained artifact and predicts uploaded trials.
- `app.py` is the Streamlit dashboard.
- `notebooks/gaitrec_colab.ipynb` is the Colab entry point.

## Model

The default baseline uses a tree-based classifier over six processed GRF channels:

- vertical force left/right
- anterior-posterior force left/right
- medio-lateral force left/right

Each signal is already normalized to 101 stance-percent points in GaitRec, so one trial becomes a fixed-length feature vector.

## Important Limitations

- This is decision support, not diagnosis.
- GaitRec is collected in a lab with force plates, not free-living phone or wearable sensors.
- Generalization to post-stroke hemiplegic gait requires an appropriate stroke or neurological-gait dataset and validation.
- No synthetic data is generated or used.
