"""
FlowCast - Model Training Module

This module trains and evaluates a traffic
forecasting model.
"""

import joblib

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -------------------------
# Project path
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
)


# -------------------------
# Load dataset
# -------------------------

forecast_file = (
    PROCESSED_DATA_PATH /
    "traffic_forecasting.csv"
)

df = pd.read_csv(
    forecast_file
)


print("\nForecasting dataset loaded.")

print(
    f"Rows : {len(df)}"
)

print(
    f"Columns : {len(df.columns)}"
)


# -------------------------
# Convert datetime
# -------------------------

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)


# -------------------------
# Sort chronologically
# -------------------------

df = df.sort_values(
    by="datetime"
).reset_index(
    drop=True
)


# -------------------------
# Target
# -------------------------

target = "traffic_volume"


# -------------------------
# Features
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
# Create X and y
# -------------------------

X = df[features]

y = df[target]


# -------------------------
# Chronological split
# -------------------------

split_index = int(
    len(df) * 0.8
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print(
    "\nDataset split completed."
)

print(
    f"Training rows : {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)


print(
    "\nTraining Random Forest..."
)


# -------------------------
# Model
# -------------------------

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)

# -------------------------
# Save model
# -------------------------

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "traffic_random_forest.pkl"
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel saved successfully:"
)

print(
    MODEL_PATH
)

print(
    "Model training completed."
)


# -------------------------
# Predictions
# -------------------------

predictions = model.predict(
    X_test
)


# -------------------------
# Evaluation
# -------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# -------------------------
# Results
# -------------------------

print(
    "\n" + "=" * 70
)

print(
    "MODEL EVALUATION"
)

print(
    "=" * 70
)

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R2   : {r2:.4f}"
)


print(
    "\nModel Training Completed."
)