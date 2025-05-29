import pandas as pd
import numpy as np
import click as ui

def speedsAsArray(month: str):
    df = pd.read_csv(f"case_study/weibull_modelling/months_data/{month}.csv")
    speeds_kmh = df['Wind Spd (km/h)'].dropna().to_numpy() # Drop all empty entries and get wind speeds
    speeds_ms = speeds_kmh / 3.6
    return speeds_ms

# Find most realistic max wind speed using 
def v_max_solver(k, c):
    return c * ((-np.log(9.99999999e-8))**(1.0/k))

if __name__ == "__main__":
    month = ui.prompt("Month: ")
    speeds_ms = speedsAsArray(month)
    # Find mean and std
    print(f"Mean wind speed: {np.mean(speeds_ms)}")
    print(f"Standard Deviation: {np.std(speeds_ms)}")

