import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander
import nrel_model_onshore as turbine   # for calc_power_mech

# ──────────────────────────────────────────────────────────────────────────────
# 1) Settings
# ──────────────────────────────────────────────────────────────────────────────
CUT_IN, V_RATED = 3.25, 9.75
DEGREE = 15   # degree of Chebyshev expansion

base_path = "case_study/turbine_model_R/model_data/"
file_names = [
    "2.3MW_nrel.csv",
    "4MW_nrel.csv",
    "5.5MW_nrel.csv",
    "7MW_nrel.csv"
]
radii = [56.5, 75, 87.5, 100]  # rotor radii for each file [m]

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load & filter Cp data
# ──────────────────────────────────────────────────────────────────────────────
per_turbine = []
v_list, Pr_list, cp_list = [], [], []

for fname in file_names:
    path = os.path.join(base_path, fname)
    df = pd.read_csv(path)

    v  = df["Wind Speed [m/s]"].to_numpy()
    cp = df["Cp [-]"].to_numpy()
    Pr = float(fname.split("MW")[0]) * 1000.0  # rated power [kW]

    mask = (v >= CUT_IN) & (v <= V_RATED)
    v_fit  = v[mask]
    cp_fit = cp[mask]
    Pr_fit = np.full_like(v_fit, Pr)

    per_turbine.append((fname.replace(".csv",""), v_fit, Pr_fit, cp_fit))
    v_list .append(v_fit)
    Pr_list.append(Pr_fit)
    cp_list.append(cp_fit)

v_all  = np.concatenate(v_list)
Pr_all = np.concatenate(Pr_list)
cp_all = np.concatenate(cp_list)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Scale wind speeds into [-1, 1] for Chebyshev
# ──────────────────────────────────────────────────────────────────────────────
def scale_to_cheb(v):
    return 2 * (v - CUT_IN) / (V_RATED - CUT_IN) - 1

x_all = scale_to_cheb(v_all)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Build design matrix using Chebyshev basis:
#    Cp(v, Pr) ≈ Σ_{i=0..DEGREE} [A_i + B_i * Pr] * T_i(x)
# ──────────────────────────────────────────────────────────────────────────────
T = chebvander(x_all, DEGREE)  # shape (n_points, DEGREE+1)
M = np.hstack([T, Pr_all[:, None] * T])
θ, *_ = np.linalg.lstsq(M, cp_all, rcond=None)
A = θ[:DEGREE+1]
B = θ[DEGREE+1:]

# ──────────────────────────────────────────────────────────────────────────────
# 5) Prediction function
# ──────────────────────────────────────────────────────────────────────────────
def cp_pred(v, Pr):
    x   = scale_to_cheb(v)
    T_v = chebvander(x, DEGREE)
    coefs = A[None, :] + (Pr[:, None] * B[None, :])
    return np.sum(T_v * coefs, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Compute & print RMSE + %RMSE per turbine (Cp)
# ──────────────────────────────────────────────────────────────────────────────
print("\nChebyshev-Cp RMSE & %RMSE (3.25–9.75 m/s) per turbine:")
rmse_list, pct_list = [], []
for label, v_fit, Pr_fit, cp_fit in per_turbine:
    cp_fit_pred = cp_pred(v_fit, Pr_fit)
    rmse  = np.sqrt(np.mean((cp_fit - cp_fit_pred)**2))
    pct   = rmse / np.mean(cp_fit) * 100
    rmse_list.append(rmse)
    pct_list.append(pct)
    print(f"  {label:8s} → RMSE = {rmse:.6f},  %RMSE = {pct:.2f}%")
print(f"\nMean Cp RMSE:  {np.mean(rmse_list):.6f}")
print(f"Mean Cp %RMSE: {np.mean(pct_list):.2f}%")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Plot measured vs. predicted Cp
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
colors = ["tab:blue","tab:orange","tab:green","tab:red"]
for (label, v_fit, _, cp_fit), c in zip(per_turbine, colors):
    plt.scatter(v_fit, cp_fit, s=10, alpha=0.4, color=c, label=f"{label} Measured")
v_plot = np.linspace(CUT_IN, V_RATED, 300)
for (label, _, Pr_fit, _), c in zip(per_turbine, colors):
    Pr = Pr_fit[0]
    plt.plot(v_plot, cp_pred(v_plot, np.full_like(v_plot,Pr)),
             linewidth=2, color=c, label=f"{label} Cheb")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Cp [-]")
plt.title("Measured vs. Chebyshev-Based Cp Model")
plt.legend(fontsize="small", ncol=2, loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 8) Compute RMSE and Plot each dataset’s raw points & its Chebyshev‐model power
# ──────────────────────────────────────────────────────────────────────────────

rmses_power = []
plt.figure(figsize=(10, 6))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for (label, v_full, _, _), R, c in zip(per_turbine, radii, colors):
    # full arrays for plotting
    df = pd.read_csv(os.path.join(base_path, f"{label}.csv"))
    v_all = df["Wind Speed [m/s]"].to_numpy()
    p_all = df["Power [kW]"].to_numpy()

    # scatter raw data points
    plt.scatter(v_all, p_all,
                s=10, color=c,
                alpha=0.4,
                label=f"{label} Measured")

    # compute predicted over dense grid
    v_plot = np.linspace(CUT_IN, V_RATED, 300)
    Cp_plot   = cp_pred(v_plot, np.full_like(v_plot, float(label.split("MW")[0]) * 1000))
    P_plot_W  = turbine.calc_power_mech(R, v_plot, Cp_plot)
    P_plot_kW = P_plot_W / 1e3

    # overlay the fitted curve
    plt.plot(v_plot, P_plot_kW,
             color=c,
             linewidth=2,
             label=f"{label} Cheb Model")

    # RMSE on filtered range
    mask = (v_all >= CUT_IN) & (v_all <= V_RATED)
    v_fit  = v_all[mask]
    p_fit  = p_all[mask]
    Cp_fit = cp_pred(v_fit, np.full_like(v_fit, float(label.split("MW")[0]) * 1000))
    P_fit  = turbine.calc_power_mech(R, v_fit, Cp_fit) / 1e3
    rmse_p = np.sqrt(np.mean((p_fit - P_fit)**2))
    rmses_power.append(rmse_p)
    print(f"  {label:8s} → RMSE = {rmse_p:.4f} kW")

mean_rmse_power = np.mean(rmses_power)
print(f"\nMean Power RMSE across all datasets: {mean_rmse_power:.4f} kW")

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power (kW)")
plt.title("Measured vs. Model Mechanical Power")
plt.legend(loc="upper left", fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 9) Save coefficients to the script's directory
# ──────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path   = os.path.join(script_dir, "cp_chebyshev_coeffs.json")
out = {**{f"A_{i}": float(A[i]) for i in range(DEGREE+1)},
       **{f"B_{i}": float(B[i]) for i in range(DEGREE+1)}}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
