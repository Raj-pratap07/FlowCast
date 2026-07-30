"""
FlowCast - Exploratory Data Analysis (EDA)
This module loads the cleaned datasets and
generates visualizations for analysis.
"""

from pathlib import Path
import pandas as pd

from plots import (
    plot_histogram,
    plot_bar_chart,
    plot_box_plot
)

# --------------------------------
# Project Paths
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


# --------------------------------
# Load Dataset
# --------------------------------

def load_processed_dataset(filename):

    file_path = PROCESSED_DATA_PATH / filename

    df = pd.read_csv(file_path)

    print(f"\nLoaded {filename}")
    print(f"Rows : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df

if __name__ == "__main__":

    traffic = load_processed_dataset(
        "traffic_sensor_log_clean.csv"
    )

    weather = load_processed_dataset(
        "weather_observation_clean.csv"
    )

    calendar = load_processed_dataset(
        "calendar_events.csv"
    )

    print("\nTRAFFIC DATA")

    print(traffic.head())

    print("\nWEATHER DATA")

    print(weather.head())

    print("\nCALENDAR DATA")

    print(calendar.head())

    print("\nGenerating Traffic Volume Histogram")

    plot_histogram(
        df=traffic,
        column="traffic_volume",
        title="Traffic Volume Distribution",
        xlabel="Traffic Volume"
    )


    print("\nGenerating Average Speed Histogram...")

    plot_histogram(
        df=traffic,
        column="avg_speed",
        title="Average Speed Distribution",
        xlabel="Average Speed (km/h)"
    )

    print("\nGenerating Congestion Level Chart...")

    plot_bar_chart(
        df=traffic,
        column="congestion_level",
        title="Congestion Level Distribution",
        xlabel="Congestion Level"
    )




