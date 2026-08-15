"""
FlowCast - this is the data validation module

This module validates the raw datasets before
cleaning or feature engineering.
"""

from load_data import (
    load_dataset,
    TRAFFIC_FILES,
    WEATHER_FILE,
    CALENDAR_FILE,
)
import pandas as pd


def validate_numeric_range(df, column, minimum=None, maximum=None):
    """
    Check whether the numeric values lie within the valid range.
    """

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    invalid = pd.Series(
        False,
        index=df.index
    )

    if minimum is not None:
        invalid = invalid | (values < minimum)

    if maximum is not None:
        invalid = invalid | (values > maximum)

    invalid_count = invalid.sum()

    print(f"\n{column}")

    print(f"Minimum : {values.min()}")
    print(f"Maximum : {values.max()}")

    if invalid_count == 0:
        print("PASS")
    else:
        print(
            f"FAIL -> {invalid_count} invalid values"
        )

def validate_categories(df, column):
    """
    Display all unique values in a categorical column.
    """

    print(f"\n{column}")

    categories = df[column].dropna().unique()

    print(categories)

    print(f"Total unique values : {len(categories)}")


if __name__ == "__main__":

    print("\nLoading datasets...")

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

    print("\n" + "=" * 70)
    print("NUMERIC VALIDATION - TRAFFIC DATASET")
    print("=" * 70)

    validate_numeric_range(
        traffic,
        "traffic_volume",
        minimum=0
    )

    validate_numeric_range(
        traffic,
        "vehicle_count",
        minimum=0
    )

    validate_numeric_range(
        traffic,
        "avg_speed",
        minimum=0,
        maximum=200
    )

    validate_numeric_range(
        traffic,
        "occupancy",
        minimum=0,
        maximum=100
    )

    validate_numeric_range(
        traffic,
        "travel_time",
        minimum=0
    )

    validate_numeric_range(
        traffic,
        "accident_count",
        minimum=0
    )

    validate_numeric_range(
        traffic,
        "signal_timing",
        minimum=0
    )

    validate_numeric_range(
        traffic,
        "road_capacity",
        minimum=0
    )

    print("\n" + "=" * 70)
    print("CATEGORY VALIDATION - TRAFFIC DATASET")
    print("=" * 70)

    validate_categories(
        traffic,
        "congestion_level"
    )

    print("\n" + "=" * 70)
    print("CATEGORY VALIDATION - WEATHER DATASET")
    print("=" * 70)

    validate_categories(
        weather,
        "weather_condition"
    )

    print("\nValidation Completed.")