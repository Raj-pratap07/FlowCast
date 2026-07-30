"""
FlowCast - Data cleaning module 
this module cleans the raw dataset and saves
the clean dataset into processed folder
"""

from pathlib import Path
import pandas as pd 

from load_data import (
    load_dataset,
    TRAFFIC_FILES,
    WEATHER_FILE,
    CALENDAR_FILE,
)

#-------------------------
# Project path
#-------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

def save_dataset(df, filename):
    # save clean dataset into processed folder
    save_path = PROCESSED_DATA_PATH / filename
    df.to_csv(save_path, index=False)
    print(f"\n{filename} saved successfully")

def clean_traffic_data(df):
    #Clean the traffic dataset
    print(f"\nCleaning the traffic dataset.")

    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    print(f"Removed {duplicates} duplicate row")

    #remove negative traffic volume 
    invalid = (df["traffic_volume"] < 0).sum()

    df.loc[df["traffic_volume"] < 0, "traffic_volume"] = pd.NA

    print(f"Replaced {invalid} negative traffic volume values with NA.")

    #Fill missing Traffic Volume
    missing = df["traffic_volume"].isna().sum()

    df["traffic_volume"] = df["traffic_volume"].fillna(
        df["traffic_volume"].median()
    )

    print(f"Filled {missing} missing traffic volume values.")

    #Fill missing average speed
    missing = df["avg_speed"].isna().sum()

    df["avg_speed"] = df["avg_speed"].fillna(
        df["avg_speed"].median()
    )

    print(f"filled {missing} missing avg_speed values")

    #fill missing occupancy
    missing = df["occupancy"].isna().sum()

    df["occupancy"] = df["occupancy"].fillna(
        df["occupancy"].median()
    )

    print(f"filled {missing} missing occupancy values.")

    #Fill missing congestion level
    missing = df["congestion_level"].isna().sum()

    df["congestion_level"] = df["congestion_level"].fillna(
        df["congestion_level"].mode()[0]
    )

    print(f"filled {missing} missing congestion_level values")

    return df

def clean_weather_data(df):
    #Clean the weather dataset

    print(f"\nCleaning weather dataset")

    #Fill missing temperature
    missing = df["temperature"].isna().sum()

    df["temperature"] = df["temperature"].fillna(
        df["temperature"].median()
    )

    print(f"filled {missing} missing temperature values")

    #Fill missing visibility
    missing = df["visibility"].isna().sum()

    df["visibility"] = df["visibility"].fillna(
        df["visibility"].median()
    )

    print(f"filled {missing} visibility values")

    # Standarize weather condition 
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

    print("Weather condition standardized")


    return df

def clean_calendar_data(df):
    #Clean the calendar dataset

    print("\nCleaning Calendar Dataset")

    #Fill missing holiday names 
    missing = df["holiday_name"].isna().sum()

    df["holiday_name"] = df["holiday_name"].fillna(
        "No Holiday"
    )

    print(f"filled {missing} missing holiday_name values")

    # Fill missing event names 
    missing = df["event_name"].isna().sum()

    df["event_name"] = df["event_name"].fillna(
        "No Event"
    )

    print(f"filled {missing} missing event_name values")

    # Remove duplicate dates
    duplicates = df.duplicated().sum()

    df = df.drop_duplicates()

    print(f"removed {duplicates} duplicate rows")

    return df


if __name__ == "__main__":

    traffic = load_dataset(TRAFFIC_FILES, "Traffic Dataset")
    weather = load_dataset(WEATHER_FILE, "Weather Dataset")
    calendar = load_dataset(CALENDAR_FILE, "Calendar Dataset")

    traffic = clean_traffic_data(traffic)

    save_dataset(
        traffic,
        "traffic_sensor_log_clean.csv"
    )

    weather = clean_weather_data(weather)

    save_dataset(
    weather,
    "weather_observation_clean.csv"
    )

    calendar = clean_calendar_data(calendar)

    save_dataset(
        calendar,
        "calendar_events.csv"
    )

    print("\nCleaning Completed.")




