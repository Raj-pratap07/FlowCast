"""
FlowCast - Feature Validation Module

This module validates the final feature dataset
before model training.
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


print("\nFeature Dataset Loaded")

print(
    f"Rows : {len(df)}"
)

print(
    f"Columns : {len(df.columns)}"
)


# -------------------------
# Missing values
# -------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_values = df.isna().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# -------------------------
# Duplicate rows
# -------------------------

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicates = df.duplicated().sum()

print(
    f"Duplicate rows : {duplicates}"
)


# -------------------------
# Data types
# -------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# -------------------------
# Feature list
# -------------------------

print("\n" + "=" * 70)
print("FEATURE LIST")
print("=" * 70)

for column in df.columns:
    print(column)


# -------------------------
# Target distribution
# -------------------------

print("\n" + "=" * 70)
print("TRAFFIC VOLUME SUMMARY")
print("=" * 70)

print(
    df["traffic_volume"].describe()
)


# -------------------------
# Congestion distribution
# -------------------------

print("\n" + "=" * 70)
print("CONGESTION DISTRIBUTION")
print("=" * 70)

print(
    df["congestion_level"]
    .value_counts()
)


print("\nFeature Validation Completed.")