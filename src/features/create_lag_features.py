"""
FlowCast - Lag Feature Module

This module creates time-based lag and rolling
features for traffic forecasting.
"""

from pathlib import Path

import pandas as pd


# -------------------------
# Project path
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
)


# -------------------------
# Load feature dataset
# -------------------------

feature_file = (
    PROCESSED_DATA_PATH /
    "traffic_features.csv"
)

df = pd.read_csv(
    feature_file
)


print("\nFeature dataset loaded.")

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


invalid_datetime = (
    df["datetime"].isna().sum()
)

print(
    f"\nInvalid datetime values : "
    f"{invalid_datetime}"
)


# -------------------------
# Remove missing target
# -------------------------

missing_target = (
    df["traffic_volume"]
    .isna()
    .sum()
)

print(
    f"Missing traffic volume : "
    f"{missing_target}"
)


df = df.dropna(
    subset=[
        "traffic_volume"
    ]
)


# -------------------------
# Sort chronologically
# -------------------------

df = df.sort_values(
    by=[
        "road_id",
        "datetime"
    ]
)


# -------------------------
# Create lag features
# -------------------------

df["traffic_lag_1"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(1)
)

df["traffic_lag_2"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(2)
)

df["traffic_lag_3"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(3)
)

df["traffic_lag_6"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(6)
)

df["traffic_lag_12"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(12)
)

df["traffic_lag_24"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(24)
)

df["traffic_lag_48"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .shift(48)
)


# -------------------------
# Rolling mean features
# -------------------------

df["traffic_rolling_mean_3"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .mean()
    )
)


df["traffic_rolling_mean_6"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(6)
        .mean()
    )
)


df["traffic_rolling_mean_12"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(12)
        .mean()
    )
)


# -------------------------
# Rolling standard deviation
# -------------------------

df["traffic_rolling_std_6"] = (
    df
    .groupby("road_id")["traffic_volume"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(6)
        .std()
    )
)


# -------------------------
# Remove rows without
# enough historical data
# -------------------------

lag_columns = [
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


before_rows = len(df)


df = df.dropna(
    subset=lag_columns
)


removed_rows = (
    before_rows - len(df)
)


print(
    f"\nRemoved {removed_rows} rows "
    "without sufficient historical data."
)


# -------------------------
# Save forecasting dataset
# -------------------------

output_file = (
    PROCESSED_DATA_PATH /
    "traffic_forecasting.csv"
)


df.to_csv(
    output_file,
    index=False
)


# -------------------------
# Final information
# -------------------------

print(
    "\nForecasting dataset saved successfully."
)

print(
    f"Rows : {len(df)}"
)

print(
    f"Columns : {len(df.columns)}"
)


print("\nLag features created:")

for column in lag_columns:
    print(
        f"- {column}"
    )


print(
    "\nLag Feature Engineering Completed."
)