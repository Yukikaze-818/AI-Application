"""Simple Streamlit app for probability prediction and what-if analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rain_prediction.config import DEFAULT_DEMO_DATA, DEFAULT_RAW_DATA, MODELS_DIR
from rain_prediction.data import load_weather_data


@st.cache_resource
def load_artifacts():
    """Load the trained model and summary metadata once."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    summary_path = MODELS_DIR / "training_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    return model, summary


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Load the best available dataset so we can initialize sensible defaults."""
    if DEFAULT_RAW_DATA.exists():
        return load_weather_data(DEFAULT_RAW_DATA)
    return load_weather_data(DEFAULT_DEMO_DATA)


def build_input_frame(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Collect user inputs and convert them into a one-row dataframe."""
    st.sidebar.header("What-if Inputs")

    default_row = reference_df.mode(dropna=False).iloc[0].to_dict()
    numeric_medians = reference_df.median(numeric_only=True)

    location = st.sidebar.selectbox("Location", sorted(reference_df["Location"].dropna().unique()))
    min_temp = st.sidebar.slider("MinTemp", -10.0, 35.0, float(numeric_medians.get("MinTemp", 12.0)))
    max_temp = st.sidebar.slider("MaxTemp", -5.0, 50.0, float(numeric_medians.get("MaxTemp", 23.0)))
    rainfall = st.sidebar.slider("Rainfall", 0.0, 100.0, float(numeric_medians.get("Rainfall", 0.8)))
    sunshine = st.sidebar.slider("Sunshine", 0.0, 14.0, float(numeric_medians.get("Sunshine", 7.0)))
    humidity_9am = st.sidebar.slider("Humidity9am", 0, 100, int(numeric_medians.get("Humidity9am", 65)))
    humidity_3pm = st.sidebar.slider("Humidity3pm", 0, 100, int(numeric_medians.get("Humidity3pm", 52)))
    pressure_9am = st.sidebar.slider("Pressure9am", 980.0, 1045.0, float(numeric_medians.get("Pressure9am", 1015.0)))
    pressure_3pm = st.sidebar.slider("Pressure3pm", 980.0, 1045.0, float(numeric_medians.get("Pressure3pm", 1013.0)))
    rain_today = st.sidebar.selectbox("RainToday", ["No", "Yes"], index=0)

    record = default_row
    record.update(
        {
            "Date": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "Location": location,
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall,
            "Evaporation": float(numeric_medians.get("Evaporation", 5.0)),
            "Sunshine": sunshine,
            "WindGustDir": default_row.get("WindGustDir", "N"),
            "WindGustSpeed": float(numeric_medians.get("WindGustSpeed", 40.0)),
            "WindDir9am": default_row.get("WindDir9am", "N"),
            "WindDir3pm": default_row.get("WindDir3pm", "NE"),
            "WindSpeed9am": float(numeric_medians.get("WindSpeed9am", 14.0)),
            "WindSpeed3pm": float(numeric_medians.get("WindSpeed3pm", 18.0)),
            "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm,
            "Cloud9am": float(numeric_medians.get("Cloud9am", 4.0)),
            "Cloud3pm": float(numeric_medians.get("Cloud3pm", 4.0)),
            "Temp9am": float(numeric_medians.get("Temp9am", 16.0)),
            "Temp3pm": float(numeric_medians.get("Temp3pm", 21.0)),
            "RainToday": rain_today,
            "RainTomorrow": "No",
        }
    )
    return pd.DataFrame([record])


def main() -> None:
    """Render the user interface."""
    st.set_page_config(page_title="Rain Prediction Challenge", layout="wide")
    st.title("Rain Prediction Challenge")
    st.write("Predict the probability of rain tomorrow and explore what-if scenarios.")

    model, summary = load_artifacts()
    reference_df = load_reference_data()
    input_df = build_input_frame(reference_df)

    probability = float(model.predict_proba(input_df.drop(columns=["RainTomorrow"]))[:, 1][0])
    st.metric("Predicted Probability of Rain Tomorrow", f"{probability:.2%}")
    st.write("Binary prediction:", "Yes" if probability >= 0.5 else "No")

    if summary:
        st.subheader("Model Summary")
        st.json(summary)

    st.subheader("Input Snapshot")
    st.dataframe(input_df)


if __name__ == "__main__":
    main()
