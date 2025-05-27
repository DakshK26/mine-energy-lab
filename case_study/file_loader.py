import pandas as pd
import numpy as np

def speedsAsArray():
    df = pd.read_csv("case_study/jan.csv")
    speeds_kmh = df['Wind Spd (km/h)'].dropna().to_numpy() # Drop all empty entries and get wind speeds
    speeds_ms = speeds_kmh / 3.6
    return speeds_ms


if __name__ == "__main__":
    speeds_ms = speedsAsArray()
    # Find mean and std
    print(f"Mean wind speed: {np.mean(speeds_ms)}")
    print(f"Standard Deviation: {np.std(speeds_ms)}")

