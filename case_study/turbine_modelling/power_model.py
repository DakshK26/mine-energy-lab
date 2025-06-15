import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander
import nrel_model_onshore as turbine   

'''
Script to find model for power coefficient.
Uses Chebyshev polynomaials.
'''

# Define constants
CUT_IN, V_RATED = 3.25, 9.75
DEGREE = 15  

# Files to model
base_path = "case_study/turbine_modelling/model_data/"
file_names = [
    "2.3MW_nrel.csv",
    "4MW_nrel.csv",
    "5.5MW_nrel.csv",
    "7MW_nrel.csv"
]
radii = [56.5, 75, 87.5, 100]  # rotor radii for each file [m]

# Load data from files into code
per_turbine = []
v_list, Pr_list, cp_list = [], [], []

# Loop through each file
for fname in file_names:
    path = os.path.join(base_path, fname) # File path is base path + file name
    df = pd.read_csv(path) # Use pandas to read the csv

    v  = df["Wind Speed [m/s]"].to_numpy() # Get all v elements from row with given name and convert to a numpy array
    cp = df["Cp [-]"].to_numpy() # same for cp
    Pr = float(fname.split("MW")[0]) * 1000.0  # rated power [kW] (Splits string from the start of the file name to the split point)

    # Limit data from cut in to v rated
    mask = (v >= CUT_IN) & (v <= V_RATED)
    v_fit  = v[mask]
    cp_fit = cp[mask]
    Pr_fit = np.full_like(v_fit, Pr) # Create an array made of Pr the same size of v_fit to use with later functions

    # Take all data and compite into an array of tuples for per_turbine
    per_turbine.append((fname.replace(".csv",""), v_fit, Pr_fit, cp_fit))
    v_list.append(v_fit)
    Pr_list.append(Pr_fit)
    cp_list.append(cp_fit)

# Turn the 4 1d arrays of data into a single sulmative array
v_all  = np.concatenate(v_list)
Pr_all = np.concatenate(Pr_list)
cp_all = np.concatenate(cp_list)


# Scale all the speeds to be in the domain of [-1,1] since chebyshev polynomials only work there
def scale_to_cheb(v):
    return 2 * (v - CUT_IN) / (V_RATED - CUT_IN) - 1

x_all = scale_to_cheb(v_all)

# Build a design matrix
# T is a matrix made of 16 columns and x rows (speed entries), each entry corresponds to the value of the chebyshev polynomial at that point
# so T[j,n] = T_n(x_j)
T = chebvander(x_all, DEGREE)  # shape (n_points, DEGREE+1)
# Make an array that holds all of t in first column and P_r * T in second columb
M = np.hstack([T, Pr_all[:, None] * T])
# Use least squares approximation from numpy to solve for best coefficients
θ, *_ = np.linalg.lstsq(M, cp_all, rcond=None)
# Solved coefficients, A is with T and B is with P_r
A = θ[:DEGREE+1]
B = θ[DEGREE+1:]

# Returns a 1d array of the predicted C_p values for each input speed
def cp_pred(v, Pr):
    x   = scale_to_cheb(v) # Scale the input speeds
    T_v = chebvander(x, DEGREE) # Make chebyshev matrix for given input speeds
    coefs = A[None, :] + (Pr[:, None] * B[None, :]) # Use previously solved for coefficients and turn them into arrays
    return np.sum(T_v * coefs, axis=1) # Multiply the chebyshev matrix with the coefficients and sum all 16 columns into 1 to make a 1d array of cp values

# Compute & Print the RMSE for each turbine
print("\nChebyshev-Cp RMSE & %RMSE (3.25–9.75 m/s) per turbine:")
rmse_list, pct_list = [], [] # List of rmse and percent error
# Loop through turbine 
for label, v_fit, Pr_fit, cp_fit in per_turbine: 
    cp_fit_pred = cp_pred(v_fit, Pr_fit)
    rmse  = np.sqrt(np.mean((cp_fit - cp_fit_pred)**2)) # Find rmse
    pct   = rmse / np.mean(cp_fit) * 100 # Find percent error by diving rmse by mean C_p * 100
    # Add both to list
    rmse_list.append(rmse) 
    pct_list.append(pct)
    print(f"  {label:8s} → RMSE = {rmse:.6f},  %RMSE = {pct:.2f}%")
# Print errors
print(f"\nMean Cp RMSE:  {np.mean(rmse_list):.6f}")
print(f"Mean Cp %RMSE: {np.mean(pct_list):.2f}%")

# Print the C_p predicted vs measured
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

# Find power at each speed by using predicted C_p and formula for power and compare against data set
# Print on plot and show mean rmse + rmse for each
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

# Output coefficients to file 
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path   = os.path.join(script_dir, "cp_chebyshev_coeffs.json")
out = {**{f"A_{i}": float(A[i]) for i in range(DEGREE+1)},
       **{f"B_{i}": float(B[i]) for i in range(DEGREE+1)}}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
