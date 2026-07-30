"""
FLOWCAST - DATA LOADING MODULE
PURPOSE: This script loads all the raw datasets used in the flowcast project 
Author: Kabir
"""
from pathlib import Path
import pandas as pd 

#------------------------------
# Project paths
#------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT/ "data" / "raw"

#------------------------------
# Dataset Path
#------------------------------

TRAFFIC_FILES = RAW_DATA_PATH / "traffic_sensor_log.csv"
WEATHER_FILE = RAW_DATA_PATH / "weather_observations.csv"
CALENDAR_FILE = RAW_DATA_PATH / "calendar_events.csv"

#----------------------------------
# Load dataset function
#--------------------------------
def load_dataset(file_path: Path, dataset_name: str):
    """
    Loads a CSV dataset safely.
    Parameters
    file_path : Path
        Path of the csv file.
    dataset_name : str
        Friendly name of dataset.
    Returns
    pandas.DataFrame
    """
    print(f"\nLoading {dataset_name}...")

    if not file_path.exists():
        raise FileNotFoundError(f"{dataset_name} not found at:\n{file_path}")

    df = pd.read_csv(file_path)

    print(f"{dataset_name} loaded successfully")
    print(f"Rows : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


#-----------------------
# Main
#---------------------- 

if __name__ == "__main__":
    traffic_df = load_dataset(TRAFFIC_FILES, "Traffic Dataset")
    weather_df = load_dataset(WEATHER_FILE, "Weather Dataset")
    calendar_df = load_dataset(CALENDAR_FILE, "Calendar Dataset")

    print("\nAll datasets loaded successfully.")  