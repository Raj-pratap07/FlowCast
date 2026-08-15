"""
FlowCast - Data cleaning module

This module cleans the raw dataset and saves
the clean dataset into processed folder.
"""

from pathlib import Path
import pandas as pd

from load_data import (
    load_dataset,
    TRAFFIC_FILES,
    WEATHER_FILE,
    CALENDAR_FILE,
)

# -------------------------
# Project path
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


def save_dataset(df, filename):
    """
    Save clean dataset into processed folder.
    """

    save_path = PROCESSED_DATA_PATH / filename

    df.to_csv(
        save_path,
        index=False
    )

    print(f"\n{filename} saved successfully")


def clean_traffic_data(df):
    """
    Clean the traffic dataset.
    """

    print("\nCleaning the traffic dataset.")

    # -------------------------
    # Remove duplicate rows
    # -------------------------

    duplicates = df.duplicated().sum()

    df = df.drop_duplicates()

    print(f"Removed {duplicates} duplicate rows.")

    # -------------------------
    # Handle negative traffic volume
    # -------------------------

    invalid = (
        df["traffic_volume"] < 0
    ).sum()

    df.loc[
        df["traffic_volume"] < 0,
        "traffic_volume"
    ] = pd.NA

    print(
        f"Replaced {invalid} negative traffic volume values with NA."
    )

    # -------------------------
    # Handle invalid occupancy
    # -------------------------

    invalid = (
        (df["occupancy"] < 0) |
        (df["occupancy"] > 100)
    ).sum()

    df.loc[
        (df["occupancy"] < 0) |
        (df["occupancy"] > 100),
        "occupancy"
    ] = pd.NA

    print(
        f"Replaced {invalid} invalid occupancy values with NA."
    )

    # -------------------------
    # Fill missing occupancy
    # -------------------------

    missing = df["occupancy"].isna().sum()

    df["occupancy"] = df["occupancy"].fillna(
        df["occupancy"].median()
    )

    print(
        f"Filled {missing} missing occupancy values."
    )

    # -------------------------
    # Handle invalid average speed
    # -------------------------

    invalid = (
        (df["avg_speed"] < 0) |
        (df["avg_speed"] > 200)
    ).sum()

    df.loc[
        (df["avg_speed"] < 0) |
        (df["avg_speed"] > 200),
        "avg_speed"
    ] = pd.NA

    print(
        f"Replaced {invalid} invalid avg_speed values with NA."
    )

    # -------------------------
    # Fill missing average speed
    # -------------------------

    missing = df["avg_speed"].isna().sum()

    df["avg_speed"] = df["avg_speed"].fillna(
        df["avg_speed"].median()
    )

    print(
        f"Filled {missing} missing avg_speed values."
    )

    # -------------------------
    # Fill missing occupancy
    # -------------------------

    missing = df["occupancy"].isna().sum()

    df["occupancy"] = df["occupancy"].fillna(
        df["occupancy"].median()
    )

    print(
        f"Filled {missing} missing occupancy values."
    )

    # -------------------------
    # Fill missing congestion level
    # -------------------------

    missing = df["congestion_level"].isna().sum()

    df["congestion_level"] = df["congestion_level"].fillna(
        df["congestion_level"].mode()[0]
    )

    print(
        f"Filled {missing} missing congestion_level values."
    )

    return df


def clean_weather_data(df):
    """
    Clean the weather dataset.
    """

    print("\nCleaning weather dataset")

    # -------------------------
    # Fill missing temperature
    # -------------------------

    missing = df["temperature"].isna().sum()

    df["temperature"] = df["temperature"].fillna(
        df["temperature"].median()
    )

    print(
        f"Filled {missing} missing temperature values."
    )

    # -------------------------
    # Fill missing visibility
    # -------------------------

    missing = df["visibility"].isna().sum()

    df["visibility"] = df["visibility"].fillna(
        df["visibility"].median()
    )

    print(
        f"Filled {missing} missing visibility values."
    )

    # -------------------------
    # Standardize weather condition
    # -------------------------

    df["weather_condition"] = (
        df["weather_condition"]
        .str.strip()
        .str.lower()
    )

    weather_mapping = {
        "clear": "Clear",
        "cloudy": "Cloudy",
        "overcast": "Overcast",
        "rain": "Rain",
        "rainy": "Rain",
        "fog": "Fog",
        "foggy": "Fog"
    }

    df["weather_condition"] = (
        df["weather_condition"]
        .replace(weather_mapping)
    )

    print("Weather conditions standardized.")

    return df


def clean_calendar_data(df):
    """
    Clean the calendar dataset.
    """

    print("\nCleaning Calendar Dataset")

    # -------------------------
    # Fill missing holiday names
    # -------------------------

    missing = df["holiday_name"].isna().sum()

    df["holiday_name"] = df["holiday_name"].fillna(
        "No Holiday"
    )

    print(
        f"Filled {missing} missing holiday_name values."
    )

    # -------------------------
    # Fill missing event names
    # -------------------------

    missing = df["event_name"].isna().sum()

    df["event_name"] = df["event_name"].fillna(
        "No Event"
    )

    print(
        f"Filled {missing} missing event_name values."
    )

    # -------------------------
    # Remove duplicate rows
    # -------------------------

    duplicates = df.duplicated().sum()

    df = df.drop_duplicates()

    print(
        f"Removed {duplicates} duplicate rows."
    )

    return df


if __name__ == "__main__":

    # -------------------------
    # Load datasets
    # -------------------------

    traffic = load_dataset(
        TRAFFIC_FILES,
        "Traffic Dataset"
    )

    weather = load_dataset(
        WEATHER_FILE,
        "Weather Dataset"
    )

    calendar = load_dataset(
        CALENDAR_FILE,
        "Calendar Dataset"
    )

    # -------------------------
    # Clean traffic dataset
    # -------------------------

    traffic = clean_traffic_data(traffic)

    save_dataset(
        traffic,
        "traffic_sensor_log_clean.csv"
    )

    # -------------------------
    # Clean weather dataset
    # -------------------------

    weather = clean_weather_data(weather)

    save_dataset(
        weather,
        "weather_observation_clean.csv"
    )

    # -------------------------
    # Clean calendar dataset
    # -------------------------

    calendar = clean_calendar_data(calendar)

    save_dataset(
        calendar,
        "calendar_events_clean.csv"
    )

    print("\nCleaning Completed.")