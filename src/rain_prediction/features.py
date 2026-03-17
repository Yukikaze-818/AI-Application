"""Feature engineering and preprocessing pipelines."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import DATE_COLUMN, TARGET_COLUMN


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived meteorological and calendar features."""
    engineered = df.copy()

    if DATE_COLUMN in engineered.columns:
        engineered[DATE_COLUMN] = pd.to_datetime(engineered[DATE_COLUMN], errors="coerce")
        engineered["Year"] = engineered[DATE_COLUMN].dt.year
        engineered["Month"] = engineered[DATE_COLUMN].dt.month
        engineered["Day"] = engineered[DATE_COLUMN].dt.day
        engineered["DayOfYear"] = engineered[DATE_COLUMN].dt.dayofyear
        engineered["MonthSin"] = np.sin(2 * np.pi * engineered["Month"] / 12.0)
        engineered["MonthCos"] = np.cos(2 * np.pi * engineered["Month"] / 12.0)
        engineered = engineered.drop(columns=[DATE_COLUMN])

    if {"MaxTemp", "MinTemp"}.issubset(engineered.columns):
        engineered["TempRange"] = engineered["MaxTemp"] - engineered["MinTemp"]
    if {"Pressure9am", "Pressure3pm"}.issubset(engineered.columns):
        engineered["PressureChange"] = engineered["Pressure3pm"] - engineered["Pressure9am"]
    if {"Humidity9am", "Humidity3pm"}.issubset(engineered.columns):
        engineered["HumidityDrop"] = engineered["Humidity9am"] - engineered["Humidity3pm"]
    if {"Temp3pm", "Temp9am"}.issubset(engineered.columns):
        engineered["TempChange"] = engineered["Temp3pm"] - engineered["Temp9am"]
    if {"WindGustSpeed", "WindSpeed3pm"}.issubset(engineered.columns):
        engineered["GustSpeedGap"] = engineered["WindGustSpeed"] - engineered["WindSpeed3pm"]
    if "Rainfall" in engineered.columns:
        engineered["LogRainfall"] = np.log1p(engineered["Rainfall"].clip(lower=0))

    return engineered


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns after feature engineering."""
    candidate = engineer_features(df.drop(columns=[TARGET_COLUMN], errors="ignore"))
    categorical_columns = candidate.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = candidate.select_dtypes(include=["number", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create the preprocessing transformer used by all models."""
    numeric_columns, categorical_columns = infer_feature_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor


def make_feature_transformer() -> FunctionTransformer:
    """Wrap the feature engineering logic so it can be inserted into pipelines."""
    return FunctionTransformer(engineer_features, validate=False)
