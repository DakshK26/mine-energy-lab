import pandas as pd
import numpy as np
import click as ui

def speedsAsArray(month: str):
    df = pd.read_csv(f"case_study/months_data/{month}.csv")
    speeds_kmh = df['Wind Spd (km/h)'].dropna().to_numpy() # Drop all empty entries and get wind speeds
    speeds_ms = speeds_kmh / 3.6
    return speeds_ms


if __name__ == "__main__":
    month = ui.prompt("Month: ")
    speeds_ms = speedsAsArray(month)
    # Find mean and std
    print(f"Mean wind speed: {np.mean(speeds_ms)}")
    print(f"Standard Deviation: {np.std(speeds_ms)}")

