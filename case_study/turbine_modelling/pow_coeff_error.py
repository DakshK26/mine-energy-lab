import pandas as pd
import numpy as np


file_info = [
    {"file": "case_study/turbine_modelling/cp_4MW.csv",   "P_r": 4.0, "V_rated": 10.0},
    {"file": "case_study/turbine_modelling/cp_5p5MW.csv", "P_r": 5.5, "V_rated": 10.0},
    {"file": "case_study/turbine_modelling/cp_7MW.csv",   "P_r": 7.0, "V_rated": 10.0},
]
V_cut_in = 3.0

dfs = []
for info in file_info:
    df = pd.read_csv(info["file"])
    df = df.rename(columns={"Wind Speed [m/s]":"V", "Cp [-]":"Cp"})
    df["P_r"] = info["P_r"]
    # filter Region 2
    df = df[(df["V"]>=V_cut_in) & (df["V"]<=info["V_rated"])]
    dfs.append(df[["V","P_r","Cp"]])
data = pd.concat(dfs, ignore_index=True)


a3, b3, a2, b2, a1, b1, a0, b0 = (
    3.625317e-03, -2.774383e-05,
   -8.090100e-02,  5.059709e-04,
    5.880855e-01, -3.009707e-03,
   -9.961359e-01,  5.691422e-03
)


V  = data["V"].values
Pr = data["P_r"].values
Cp_true = data["Cp"].values

Cp_pred = (
    (a3 + b3*Pr)*V**3
  + (a2 + b2*Pr)*V**2
  + (a1 + b1*Pr)*V
  + (a0 + b0*Pr)
)

residuals = Cp_pred - Cp_true

# RMSE
rmse = np.sqrt(np.mean(residuals**2))
# R^2
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Cp_true - Cp_true.mean())**2)
r2 = 1 - ss_res/ss_tot

print(f"Overall RMSE: {rmse:.4f}")
print(f"Overall R^2:   {r2:.4f}")

# per-turbine stats:
for Pr_val in sorted(data["P_r"].unique()):
    mask = data["P_r"]==Pr_val
    res = residuals[mask]
    rmse_i = np.sqrt(np.mean(res**2))
    print(f"RMSE for {Pr_val} MW: {rmse_i:.4f}")
