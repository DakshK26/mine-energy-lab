import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1) Constants and file list
# ──────────────────────────────────────────────────────────────────────────────

RATED_SPEED = 9.75  # m/s

# CSV files live under "case_study/turbine_modelling/model_data/"
base_path = "case_study/turbine_modelling/model_data/"

file_names = [
    "2.3MW_nrel.csv",
    "4MW_nrel.csv",
    "5.5MW_nrel.csv",
    "7MW_nrel.csv"
]

rated_mw_list      = []
cp_at_rated_speed = []

# ──────────────────────────────────────────────────────────────────────────────
# 2) Read each CSV and extract Cp at 9.75 m/s, record rated power (MW)
# ──────────────────────────────────────────────────────────────────────────────

for fname in file_names:
    path = base_path + fname
    df   = pd.read_csv(path)
    
    # Find the row where Wind Speed == 9.75
    row = df[df["Wind Speed [m/s]"] == RATED_SPEED]
    if row.empty:
        raise ValueError(f"No row with wind speed {RATED_SPEED} m/s found in {fname}")
    
    cp_value = row.iloc[0]["Cp [-]"]
    
    # Extract rated power in MW from filename (e.g. "2.3MW" -> 2.3)
    rated_mw = float(fname.split("MW")[0])
    
    rated_mw_list.append(rated_mw)
    cp_at_rated_speed.append(cp_value)

# Convert to NumPy arrays
x = np.array(rated_mw_list)       # [2.3, 4.0, 5.5, 7.0]
y = np.array(cp_at_rated_speed)   # Cp at 9.75 m/s

print("Collected points: (Rated Power [MW], Cp at 9.75 m/s)")
for Pr, Cp in zip(x, y):
    print(f"  {Pr:4.1f} MW  →  Cp = {Cp:.5f}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Fit a 3rd-degree polynomial: Cp = a*P^3 + b*P^2 + c*P + d
# ──────────────────────────────────────────────────────────────────────────────

coeffs = np.polyfit(x, y, 3)  # [a, b, c, d]

print("3rd-degree polynomial coefficients (highest power first):")
for power, coef in zip(range(3, -1, -1), coeffs):
    print(f"  P^{power}: {coef:.6e}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Plot data points and fitted curve
# ──────────────────────────────────────────────────────────────────────────────

x_plot = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
y_plot = np.polyval(coeffs, x_plot)

plt.figure(figsize=(7,5))
plt.scatter(x, y, color='tab:blue', s=50, label="Cp @ 9.75 m/s")
plt.plot(x_plot, y_plot, color='tab:red', linewidth=2, label="Degree 3 fit")

for xi, yi in zip(x, y):
    plt.text(xi + 0.05, yi - 0.005, f"({xi:.1f}, {yi:.3f})", fontsize=9, va='top')

plt.xlabel("Rated Power (MW)")
plt.ylabel("Cp at 9.75 m/s")
plt.title("Cp vs. Rated Power (3rd-Degree Polynomial)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
