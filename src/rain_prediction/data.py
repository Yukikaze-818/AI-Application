"""Data loading and synthetic demo dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATE_COLUMN, TARGET_COLUMN


EXPECTED_COLUMNS = [
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "RainTomorrow",
]


def generate_demo_weather_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic-enough weather dataset for local testing."""
    rng = np.random.default_rng(random_state)

    locations = np.array(["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra"])
    wind_dirs = np.array(["N", "NNE", "NE", "E", "SE", "S", "SW", "W", "NW"])

    dates = pd.date_range("2014-01-01", periods=n_samples, freq="D")
    location = rng.choice(locations, size=n_samples, p=[0.22, 0.18, 0.18, 0.14, 0.12, 0.16])

    base_temp = rng.normal(18, 7, size=n_samples)
    seasonal = 8 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)

    min_temp = base_temp + seasonal - rng.uniform(4, 10, size=n_samples)
    max_temp = base_temp + seasonal + rng.uniform(4, 12, size=n_samples)
    rainfall = np.maximum(0, rng.gamma(shape=1.4, scale=3.0, size=n_samples) - 1.5)
    humidity_9am = np.clip(rng.normal(70, 18, size=n_samples) + rainfall * 2.5, 20, 100)
    humidity_3pm = np.clip(humidity_9am - rng.normal(15, 10, size=n_samples), 10, 100)
    pressure_9am = rng.normal(1015, 7, size=n_samples) - rainfall * 0.4
    pressure_3pm = pressure_9am + rng.normal(0, 3, size=n_samples)
    sunshine = np.clip(rng.normal(8, 3, size=n_samples) - rainfall * 0.35, 0, 14)
    evaporation = np.clip(rng.normal(5, 2, size=n_samples) + (max_temp - min_temp) * 0.05, 0, 15)
    wind_gust_speed = np.clip(rng.normal(38, 12, size=n_samples) + rainfall * 0.8, 5, 120)
    wind_speed_9am = np.clip(wind_gust_speed * rng.uniform(0.35, 0.75, size=n_samples), 0, 80)
    wind_speed_3pm = np.clip(wind_gust_speed * rng.uniform(0.4, 0.85, size=n_samples), 0, 90)
    cloud_9am = np.clip((humidity_9am / 12 + rng.normal(0, 1.5, size=n_samples)).round(), 0, 8)
    cloud_3pm = np.clip((humidity_3pm / 12 + rng.normal(0, 1.5, size=n_samples)).round(), 0, 8)
    temp_9am = min_temp + rng.uniform(1, 7, size=n_samples)
    temp_3pm = max_temp - rng.uniform(0, 5, size=n_samples)

    rain_today = np.where(rainfall > 1.0, "Yes", "No")

    logits = (
        9.5
        + 0.03 * humidity_3pm
        + 0.02 * humidity_9am
        + 0.08 * rainfall
        - 0.28 * sunshine
        - 0.012 * pressure_3pm
        + 0.012 * cloud_3pm
        + 0.01 * wind_gust_speed
        + 0.7 * (rain_today == "Yes").astype(float)
    )
    probability = 1 / (1 + np.exp(-logits))
    rain_tomorrow = np.where(rng.random(n_samples) < probability, "Yes", "No")

    data = pd.DataFrame(
        {
            "Date": dates,
            "Location": location,
            "MinTemp": min_temp.round(1),
            "MaxTemp": max_temp.round(1),
            "Rainfall": rainfall.round(1),
            "Evaporation": evaporation.round(1),
            "Sunshine": sunshine.round(1),
            "WindGustDir": rng.choice(wind_dirs, size=n_samples),
            "WindGustSpeed": wind_gust_speed.round(0),
            "WindDir9am": rng.choice(wind_dirs, size=n_samples),
            "WindDir3pm": rng.choice(wind_dirs, size=n_samples),
            "WindSpeed9am": wind_speed_9am.round(0),
            "WindSpeed3pm": wind_speed_3pm.round(0),
            "Humidity9am": humidity_9am.round(0),
            "Humidity3pm": humidity_3pm.round(0),
            "Pressure9am": pressure_9am.round(1),
            "Pressure3pm": pressure_3pm.round(1),
            "Cloud9am": cloud_9am,
            "Cloud3pm": cloud_3pm,
            "Temp9am": temp_9am.round(1),
            "Temp3pm": temp_3pm.round(1),
            "RainToday": rain_today,
            "RainTomorrow": rain_tomorrow,
        }
    )

    missing_rates = {
        "Evaporation": 0.20,
        "Sunshine": 0.15,
        "Cloud9am": 0.10,
        "Cloud3pm": 0.10,
        "Pressure9am": 0.05,
        "Pressure3pm": 0.05,
        "WindGustDir": 0.03,
        "WindDir9am": 0.03,
        "WindDir3pm": 0.03,
    }
    for column, rate in missing_rates.items():
        mask = rng.random(n_samples) < rate
        data.loc[mask, column] = np.nan

    return data[EXPECTED_COLUMNS]


def save_demo_dataset(output_path: Path, n_samples: int = 5000, random_state: int = 42) -> Path:
    """Generate and store a demo dataset on disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_demo_weather_data(n_samples=n_samples, random_state=random_state)
    df.to_csv(output_path, index=False)
    return output_path


def load_weather_data(path: Path | str) -> pd.DataFrame:
    """Load the dataset and perform lightweight schema validation."""
    df = pd.read_csv(path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize target values to binary integers."""
    df = df.copy()
    df = df[df[TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df[df[TARGET_COLUMN].isin([0, 1])].copy()


def train_valid_test_split(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train, validation, and test partitions."""
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=random_state,
        stratify=target,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test

