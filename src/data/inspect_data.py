"""
FlowCast - data inspection model
this scripts load every datasets and prints
important informantion 
"""
from load_data import (
    load_dataset,
    TRAFFIC_FILES,
    WEATHER_FILE,
    CALENDAR_FILE,
)

def inspect_dataframe(df,name):
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nMemory Usage")
    print(df.memory_usage(deep=True))

    print("\nSummary Statistics")
    print(df.describe(include="all"))

if __name__ == "__main__":

    traffic = load_dataset(TRAFFIC_FILES, "Traffic Dataset")
    weather = load_dataset(WEATHER_FILE, "Weather Dataset")
    calendar = load_dataset(CALENDAR_FILE, "Calendar Dataset")

    inspect_dataframe(traffic, "Traffic Dataset")
    inspect_dataframe(weather, "Weather Dataset")
    inspect_dataframe(calendar, "Calendar Dataset")
    
