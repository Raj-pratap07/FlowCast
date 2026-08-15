"""
FlowCast - Temporal EDA Module

This module analyzes traffic patterns based on
time, day of week, weekends and roads.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Project path
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "figures"
)

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------
# Load traffic dataset
# -------------------------

traffic_file = (
    PROCESSED_DATA_PATH /
    "traffic_sensor_log_clean.csv"
)

traffic = pd.read_csv(
    traffic_file
)

print(
    "Loaded traffic_sensor_log_clean.csv"
)

print(
    f"Rows : {len(traffic)}"
)

print(
    f"Columns : {len(traffic.columns)}"
)


# -------------------------
# Create datetime
# -------------------------

traffic["datetime"] = pd.to_datetime(
    traffic["date"] + " " + traffic["time"],
    errors="coerce"
)

print("\nDatetime conversion completed.")

print(
    f"Invalid datetime values : "
    f"{traffic['datetime'].isna().sum()}"
)


# -------------------------
# Create time features
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

traffic["is_weekend"] = (
    traffic["day_of_week"] >= 5
)


# -------------------------
# Hourly traffic analysis
# -------------------------

hourly_traffic = (
    traffic
    .groupby("hour")["traffic_volume"]
    .mean()
)


print("\nAverage Traffic Volume By Hour")

print(hourly_traffic)


plt.figure(figsize=(10, 6))

plt.plot(
    hourly_traffic.index,
    hourly_traffic.values,
    marker="o"
)

plt.title(
    "Average Traffic Volume by Hour"
)

plt.xlabel("Hour of Day")

plt.ylabel(
    "Average Traffic Volume"
)

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    REPORT_PATH /
    "traffic_volume_by_hour.png"
)

plt.show()


# -------------------------
# Day of week analysis
# -------------------------

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_traffic = (
    traffic
    .groupby("day_name")["traffic_volume"]
    .mean()
    .reindex(day_order)
)


print("\nAverage Traffic Volume By Day")

print(daily_traffic)


plt.figure(figsize=(10, 6))

plt.bar(
    daily_traffic.index,
    daily_traffic.values
)

plt.title(
    "Average Traffic Volume by Day of Week"
)

plt.xlabel("Day")

plt.ylabel(
    "Average Traffic Volume"
)

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    REPORT_PATH /
    "traffic_volume_by_day.png"
)

plt.show()


# -------------------------
# Weekday vs weekend
# -------------------------

weekend_traffic = (
    traffic
    .groupby("is_weekend")["traffic_volume"]
    .mean()
)


weekend_traffic.index = [
    "Weekday" if value is False else "Weekend"
    for value in weekend_traffic.index
]


print("\nWeekday vs Weekend Traffic")

print(weekend_traffic)


plt.figure(figsize=(7, 6))

plt.bar(
    weekend_traffic.index,
    weekend_traffic.values
)

plt.title(
    "Weekday vs Weekend Traffic Volume"
)

plt.xlabel("Day Type")

plt.ylabel(
    "Average Traffic Volume"
)

plt.tight_layout()

plt.savefig(
    REPORT_PATH /
    "weekday_vs_weekend.png"
)

plt.show()


# -------------------------
# Road-wise traffic
# -------------------------

road_traffic = (
    traffic
    .groupby("road_name")["traffic_volume"]
    .mean()
    .sort_values(
        ascending=False
    )
)


print("\nTop 10 Roads by Average Traffic Volume")

print(
    road_traffic.head(10)
)


plt.figure(figsize=(12, 7))

plt.barh(
    road_traffic.head(10).index[::-1],
    road_traffic.head(10).values[::-1]
)

plt.title(
    "Top 10 Roads by Average Traffic Volume"
)

plt.xlabel(
    "Average Traffic Volume"
)

plt.ylabel("Road")

plt.tight_layout()

plt.savefig(
    REPORT_PATH /
    "top_10_roads_traffic.png"
)

plt.show()


# -------------------------
# Peak traffic hours
# -------------------------

peak_hours = (
    hourly_traffic
    .sort_values(
        ascending=False
    )
    .head(5)
)


print("\nTop 5 Peak Traffic Hours")

print(peak_hours)


# -------------------------
# Low traffic hours
# -------------------------

low_hours = (
    hourly_traffic
    .sort_values()
    .head(5)
)


print("\nTop 5 Lowest Traffic Hours")

print(low_hours)


# -------------------------
# Save temporal dataset
# -------------------------

temporal_file = (
    PROCESSED_DATA_PATH /
    "traffic_temporal_analysis.csv"
)

traffic.to_csv(
    temporal_file,
    index=False
)

print(
    "\nTemporal analysis dataset saved successfully."
)

print(
    "\nTemporal EDA Completed."
)