import pandas as pd
import numpy as np


file_info = [
    {"file": "case_study/turbine_modelling/cp_4MW.csv",   "P_r": 4.0, "V_rated": 10.0},
    {"file": "case_study/turbine_modelling/cp_5p5MW.csv", "P_r": 5.5, "V_rated": 10.0},
    {"file": "case_study/turbine_modelling/cp_7MW.csv",   "P_r": 7.0, "V_rated": 10.0},
]
V_cut_in = 3.0  # cut-in wind speed (m/s)

# r2 data
df_list = []
for info in file_info:
    df = pd.read_csv(info["file"])
    df = df.rename(columns={"Wind Speed [m/s]": "V", "Cp [-]": "Cp"})
    df["P_r"] = info["P_r"]
    df["V_rated"] = info["V_rated"]
    # filter for Region 2: cut-in ≤ V ≤ V_rated
    r2 = df[(df["V"] >= V_cut_in) & (df["V"] <= df["V_rated"])]
    df_list.append(r2[["V", "P_r", "Cp"]])

combined = pd.concat(df_list, ignore_index=True)

# matrix
V = combined["V"].values
Pr = combined["P_r"].values
Cp = combined["Cp"].values

# 
X = np.vstack([
    V**3,
    Pr * V**3,
    V**2,
    Pr * V**2,
    V,
    Pr * V,
    np.ones_like(V),
    Pr
]).T

# Solve least squares for coefficients [a3, b3, a2, b2, a1, b1, a0, b0]
coeffs, *_ = np.linalg.lstsq(X, Cp, rcond=None)
param_names = ["a3", "b3", "a2", "b2", "a1", "b1", "a0", "b0"]

# Print the coefficients
print("Bivariate polynomial coefficients for C_p(V, P_r):")
for name, val in zip(param_names, coeffs):
    print(f"  {name} = {val:.6e}")

# Print the final equation
equation = (
    "C_p(V, P_r) = "
    f"({coeffs[0]:.3e} + {coeffs[1]:.3e}*P_r)*V**3 + "
    f"({coeffs[2]:.3e} + {coeffs[3]:.3e}*P_r)*V**2 + "
    f"({coeffs[4]:.3e} + {coeffs[5]:.3e}*P_r)*V + "
    f"({coeffs[6]:.3e} + {coeffs[7]:.3e}*P_r)"
)
print("\nResulting model:")
print(equation)
