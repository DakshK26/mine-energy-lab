import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander

# ──────────────────────────────────────────────────────────────────────────────
# 1) Settings
# ──────────────────────────────────────────────────────────────────────────────
CUT_IN, V_RATED = 3.25, 9.75
DEGREE = 15   # degree of Chebyshev expansion

# base_path = "case_study/turbine_modelling/model_data/"
base_path = "test_case_turbine/model_data/"
file_names = [
    "2.3MW_nrel.csv",
    "4MW_nrel.csv",
    "5.5MW_nrel.csv",
    "7MW_nrel.csv"
]

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

# Stack columns: [T,  Pr_all[:,None] * T]
M = np.hstack([T, Pr_all[:, None] * T])

# Solve for θ = [A_0...A_DEGREE,  B_0...B_DEGREE]
θ, *_ = np.linalg.lstsq(M, cp_all, rcond=None)
A = θ[:DEGREE+1]
B = θ[DEGREE+1:]

# ──────────────────────────────────────────────────────────────────────────────
# 5) Prediction function
# ──────────────────────────────────────────────────────────────────────────────
def cp_pred(v, Pr):
    x   = scale_to_cheb(v)
    T_v = chebvander(x, DEGREE)               # shape (n, DEGREE+1)
    coefs = A[None, :] + (Pr[:, None] * B[None, :])
    return np.sum(T_v * coefs, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Compute & print RMSE + %RMSE per turbine
# ──────────────────────────────────────────────────────────────────────────────
print("\nChebyshev-Cp RMSE & %RMSE (3.25–10 m/s) per turbine:")
rmse_list, pct_list = [], []

for label, v_fit, Pr_fit, cp_fit in per_turbine:
    cp_fit_pred = cp_pred(v_fit, Pr_fit)
    rmse  = np.sqrt(np.mean((cp_fit - cp_fit_pred)**2))
    pct   = rmse / np.mean(cp_fit) * 100

    rmse_list.append(rmse)
    pct_list.append(pct)
    print(f"  {label:8s} → RMSE = {rmse:.6f},  %RMSE = {pct:.2f}%")

print(f"\nMean RMSE:  {np.mean(rmse_list):.6f}")
print(f"Mean %RMSE: {np.mean(pct_list):.2f}%")

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
    plt.plot(v_plot, cp_pred(v_plot, np.full_like(v_plot, Pr)),
             linewidth=2, color=c, label=f"{label} Cheb")

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Cp [-]")
plt.title("Measured vs. Chebyshev-Based Cp Model")
plt.legend(fontsize="small", ncol=2, loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 8) Save coefficients to the script's directory
# ──────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path   = os.path.join(script_dir, "cp_chebyshev_coeffs.json")

out = {**{f"A_{i}": float(A[i]) for i in range(DEGREE+1)},
       **{f"B_{i}": float(B[i]) for i in range(DEGREE+1)}}

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
