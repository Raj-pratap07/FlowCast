"""
FlowCast - Traffic Prediction Module

This module loads the trained model and
generates traffic volume predictions.
"""

from pathlib import Path

import joblib
import pandas as pd


# -------------------------
# Project path
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "traffic_random_forest.pkl"
)


# -------------------------
# Load model
# -------------------------

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# -------------------------
# Load forecasting data
# -------------------------

forecast_file = (
    PROCESSED_DATA_PATH /
    "traffic_forecasting.csv"
)

df = pd.read_csv(
    forecast_file
)


print(
    f"\nForecasting dataset loaded."
)

print(
    f"Rows : {len(df)}"
)


# -------------------------
# Select features
# -------------------------

features = [
    "latitude",
    "longitude",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "weather_condition",
    "temperature",
    "rainfall",
    "visibility",
    "public_holiday",
    "event_flag",
    "roadwork_flag",
    "is_rainy",
    "low_visibility",
    "traffic_lag_1",
    "traffic_lag_2",
    "traffic_lag_3",
    "traffic_lag_6",
    "traffic_lag_12",
    "traffic_lag_24",
    "traffic_lag_48",
    "traffic_rolling_mean_3",
    "traffic_rolling_mean_6",
    "traffic_rolling_mean_12",
    "traffic_rolling_std_6"
]


# -------------------------
# Prepare input
# -------------------------

X = df[features]


# -------------------------
# Generate predictions
# -------------------------

predictions = model.predict(
    X
)


# -------------------------
# Add predictions
# -------------------------

df["predicted_traffic_volume"] = (
    predictions
)


# -------------------------
# Display predictions
# -------------------------

print(
    "\nSample Predictions"
)

print(
    df[
        [
            "traffic_volume",
            "predicted_traffic_volume"
        ]
    ].head(10)
)


# -------------------------
# Save predictions
# -------------------------

output_file = (
    PROCESSED_DATA_PATH /
    "traffic_predictions.csv"
)


df.to_csv(
    output_file,
    index=False
)


print(
    "\nPredictions saved successfully."
)

print(
    output_file
)


print(
    "\nPrediction Completed."
)