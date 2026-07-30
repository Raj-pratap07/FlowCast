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


def validate_numeric_range(df, column, minimum=None, maximum=None):
    """
    Check whether the numeric values lie within the valid range.
    """

    invalid = df[column]

    if minimum is not None:
        invalid = invalid[invalid < minimum]

    if maximum is not None:
        invalid = invalid[invalid > maximum]

    print(f"\n{column}")

    if len(invalid) == 0:
        print("PASS")
    else:
        print(f"FAIL -> {len(invalid)} invalid values")


def validate_categories(df, column):
    """
    Display all unique values in a categorical column.
    """

    print(f"\n{column}")
    print(df[column].dropna().unique())


if __name__ == "__main__":

    print("\nLoading datasets...")

    traffic = load_dataset(TRAFFIC_FILES, "Traffic Dataset")
    weather = load_dataset(WEATHER_FILE, "Weather Dataset")
    calendar = load_dataset(CALENDAR_FILE, "Calendar Dataset")

    print("\n" + "=" * 70)
    print("NUMERIC VALIDATION - TRAFFIC DATASET")
    print("=" * 70)

    validate_numeric_range(traffic, "traffic_volume", minimum=0)
    validate_numeric_range(traffic, "vehicle_count", minimum=0)
    validate_numeric_range(traffic, "avg_speed", minimum=0, maximum=200)
    validate_numeric_range(traffic, "occupancy", minimum=0, maximum=100)
    validate_numeric_range(traffic, "travel_time", minimum=0)
    validate_numeric_range(traffic, "accident_count", minimum=0)
    validate_numeric_range(traffic, "signal_timing", minimum=0)
    validate_numeric_range(traffic, "road_capacity", minimum=0)

    print("\n" + "=" * 70)
    print("CATEGORY VALIDATION - TRAFFIC DATASET")
    print("=" * 70)

    validate_categories(traffic, "congestion_level")

    print("\n" + "=" * 70)
    print("CATEGORY VALIDATION - WEATHER DATASET")
    print("=" * 70)

    validate_categories(weather, "weather_condition")

    print("\nValidation Completed.")