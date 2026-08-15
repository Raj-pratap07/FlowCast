"""
FlowCast - Feature Engineering Module

This module prepares the traffic, weather and calendar
datasets for machine learning.
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
# Load datasets
# -------------------------

traffic_file = (
    PROCESSED_DATA_PATH /
    "traffic_sensor_log_clean.csv"
)

weather_file = (
    PROCESSED_DATA_PATH /
    "weather_observation_clean.csv"
)

calendar_file = (
    PROCESSED_DATA_PATH /
    "calendar_events_clean.csv"
)


traffic = pd.read_csv(
    traffic_file
)

weather = pd.read_csv(
    weather_file
)

calendar = pd.read_csv(
    calendar_file
)


print("\nDatasets loaded successfully.")

print(
    f"Traffic Rows : {len(traffic)}"
)

print(
    f"Weather Rows : {len(weather)}"
)

print(
    f"Calendar Rows : {len(calendar)}"
)


# -------------------------
# Create traffic datetime
# -------------------------

traffic["datetime"] = pd.to_datetime(
    traffic["date"] + " " + traffic["time"],
    errors="coerce"
)


invalid_datetime = (
    traffic["datetime"].isna().sum()
)

print("\nDatetime columns created.")

print(
    f"Invalid traffic datetime values : "
    f"{invalid_datetime}"
)


# -------------------------
# Create traffic time features
# -------------------------

traffic["hour"] = (
    traffic["datetime"].dt.hour
)

traffic["day_of_week"] = (
    traffic["datetime"].dt.dayofweek
)

traffic["day_name"] = (
    traffic["datetime"].dt.day_name()
)

traffic["month"] = (
    traffic["datetime"].dt.month
)

traffic["is_weekend"] = (
    traffic["day_of_week"] >= 5
).astype(int)


# -------------------------
# Peak hour feature
# -------------------------

traffic["is_peak_hour"] = (
    traffic["hour"].isin(
        [7, 8, 9, 17, 18, 19]
    )
).astype(int)


# -------------------------
# Prepare weather datetime
# -------------------------

weather["datetime"] = pd.to_datetime(
    weather["date"] + " " + weather["time"],
    dayfirst=True,
    errors="coerce"
)


weather["weather_hour"] = (
    weather["datetime"].dt.floor("h")
)

traffic["traffic_hour"] = (
    traffic["datetime"].dt.floor("h")
)


# -------------------------
# Remove duplicate weather records
# -------------------------

weather = weather.drop_duplicates(
    subset=[
        "station_id",
        "weather_hour"
    ]
)


# -------------------------
# Merge weather data
# -------------------------

traffic = traffic.merge(
    weather[
        [
            "station_id",
            "weather_hour",
            "weather_condition",
            "temperature",
            "rainfall",
            "visibility"
        ]
    ],
    left_on=[
        "weather_station_id",
        "traffic_hour"
    ],
    right_on=[
        "station_id",
        "weather_hour"
    ],
    how="left"
)


# -------------------------
# Remove weather helper columns
# -------------------------

traffic = traffic.drop(
    columns=[
        "station_id",
        "weather_hour",
        "traffic_hour"
    ],
    errors="ignore"
)


print("\nWeather data merged.")


# -------------------------
# Prepare calendar data
# -------------------------

calendar["calendar_date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)


calendar_features = calendar[
    [
        "calendar_date",
        "public_holiday",
        "holiday_name",
        "event_flag",
        "event_name",
        "roadwork_flag"
    ]
].copy()


# -------------------------
# Create traffic calendar date
# -------------------------

traffic["traffic_date"] = (
    traffic["datetime"].dt.normalize()
)


# -------------------------
# Merge calendar data
# -------------------------

traffic = traffic.merge(
    calendar_features,
    left_on="traffic_date",
    right_on="calendar_date",
    how="left"
)


print("\nCalendar data merged.")


# -------------------------
# Remove calendar helper columns
# -------------------------

traffic = traffic.drop(
    columns=[
        "traffic_date",
        "calendar_date"
    ],
    errors="ignore"
)


# -------------------------
# Weather condition cleaning
# -------------------------

traffic["weather_condition"] = (
    traffic["weather_condition"]
    .fillna("Clear")
)


weather_mapping = {
    "Clear": 0,
    "Cloudy": 1,
    "Overcast": 2,
    "Rain": 3,
    "Fog": 4
}


traffic["weather_condition"] = (
    traffic["weather_condition"]
    .map(weather_mapping)
)


# -------------------------
# Weather indicators
# -------------------------

traffic["is_rainy"] = (
    traffic["rainfall"] > 0
).astype(int)


traffic["low_visibility"] = (
    traffic["visibility"] < 5000
).astype(int)


# -------------------------
# Check missing weather data
# -------------------------

print("\nMissing Weather Values")

print(
    traffic[
        [
            "temperature",
            "rainfall",
            "visibility"
        ]
    ].isna().sum()
)


# -------------------------
# Fill missing weather data
# -------------------------

traffic["temperature"] = (
    traffic["temperature"]
    .fillna(
        traffic["temperature"].median()
    )
)

traffic["rainfall"] = (
    traffic["rainfall"]
    .fillna(0)
)

traffic["visibility"] = (
    traffic["visibility"]
    .fillna(
        traffic["visibility"].median()
    )
)


# -------------------------
# Handle missing target values
# -------------------------

missing_target = (
    traffic["traffic_volume"]
    .isna()
    .sum()
)

print(
    f"\nMissing traffic_volume values : "
    f"{missing_target}"
)


if missing_target > 0:

    traffic = traffic.dropna(
        subset=[
            "traffic_volume"
        ]
    )

    print(
        f"Removed {missing_target} rows "
        "with missing traffic_volume."
    )


# -------------------------
# Remove unnecessary columns
# -------------------------
traffic = traffic.drop(
    columns=[
        "date",
        "time",
        "day_name",
        "holiday_name",
        "event_name"
    ],
    errors="ignore"
)

# -------------------------
# Sort dataset
# -------------------------

traffic = traffic.sort_values(
    by=[
        "road_id",
        "hour"
    ]
)


# -------------------------
# Save feature dataset
# -------------------------

feature_file = (
    PROCESSED_DATA_PATH /
    "traffic_features.csv"
)


traffic.to_csv(
    feature_file,
    index=False
)


# -------------------------
# Final information
# -------------------------

print(
    "\nFeature dataset saved successfully."
)

print(
    f"Rows : {len(traffic)}"
)

print(
    f"Columns : {len(traffic.columns)}"
)

print(
    "\nFeature Engineering Completed."
)