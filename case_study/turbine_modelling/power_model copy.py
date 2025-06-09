import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
import json as file
import os
import R0_to_Pr as convert # So use this by doing convert.R0_of_Pr(P_r)

# For Parralelization
from multiprocessing.pool import ThreadPool

'''
Adjust model to account for radius aswell. Higher radius will generate increased power at lower wind speeds even for the same power rating

'''

# Constans and file processing

CUT_IN = 3.25   # m/s
V_RATED = 10   # m/s

base_path = "case_study/turbine_modelling/model_data/"

file_names = [
    "2.3MW_nrel.csv",
    "4MW_nrel.csv",
    "5.5MW_nrel.csv",
    "7MW_nrel.csv"
]

# Radius of every turbine (new)
radius = [
    56.5,
    75,
    87.5,
    100
]

# We'll store:
#   datasets_full:  tuples (label, v_full, p_full, rated_kw)
#   datasets_fit:   tuples (label, v_fit,  p_fit,  rated_kw)  for filtering/fitting

datasets_full = []
datasets_fit  = []

for fname in file_names:
    path = base_path + fname
    df = pd.read_csv(path)

    # Full arrays (for plotting)
    v_full = df["Wind Speed [m/s]"].to_numpy()
    p_full = df["Power [kW]"].to_numpy()
    # Extract rated power (e.g. "3.3MW" → 3.3 * 1000 = 3300 kW)
    rated_mw = float(fname.split("MW")[0])
    rated_kw = rated_mw * 1000.0

    label = fname.replace(".csv", "")
    datasets_full.append((label, v_full, p_full, rated_kw))

    # Filtered arrays: only 3.25 <= v <= 10.0
    mask = (v_full >= CUT_IN) & (v_full <= V_RATED)
    v_fit = v_full[mask]
    p_fit = p_full[mask]
    datasets_fit.append((label, v_fit, p_fit, rated_kw))

# Concatenate all filtered data
v_all      = np.concatenate([v for (_, v, _, _) in datasets_fit])
p_obs_all  = np.concatenate([p for (_, _, p, _) in datasets_fit])
rated_all  = np.concatenate([np.full_like(v, rated) for (_, v, _, rated) in datasets_fit])

# ──────────────────────────────────────────────────────────────────────────────
# 2) Define normalized 5PL:  L(v; a,b,c,d,g)
# ──────────────────────────────────────────────────────────────────────────────

def logistic5_norm(v, a, b, c, d, g):
    """
    Normalized 5PL:
      L(v) = d + (a - d) / [1 + (v / c)^b]^g

    Output is between 'a' (bottom fraction) and 'd' (top fraction).
    """
    return d + (a - d) / (1 + (v / c) ** b) ** g



# New function to account for radius. Will try adjusting c for radius.
def adj_c(c: float, R: float, R0: float, k:float, alpha: float):
    return c * k * (R/(R0))^alpha
    

# For each turbine: P_pred(v) = rated_kw * L(v; a,b,c,d,g)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Create a pymoo Problem: minimize Σ (P_obs - rated * L(v))^2
# ──────────────────────────────────────────────────────────────────────────────

class RatedPowerCurveProblem(ElementwiseProblem):
    def __init__(self, v, p_obs, rated, **kwargs):
        super().__init__(
            n_var=5,  # a, b, c, d, g
            n_obj=1,  # single‐objective: SSE
            xl=np.array([  0.0,  0.1,   0.1,   0.8,  0.5 ]),  # bounds for a,b,c,d,g
            xu=np.array([  0.2, 10.0,  25.0,   1.2, 50.0 ]),
            **kwargs
        )
        self.v     = v
        self.p_obs = p_obs
        self.rated = rated

    def _evaluate(self, x, out, *args, **kwargs):
        # Get parameters
        a, b, c, d, g = x
        
        # Calc f1
        L_pred = logistic5_norm(self.v, a, b, c, d, g)
        P_pred = self.rated * L_pred
        f1 = np.sum((self.p_obs - P_pred) ** 2)
        
        
        out["F"] = [f1]

# Parralelize
threads = 8
pool = ThreadPool(threads) # Creates a pool of worker threads
runner = StarmapParallelization(pool.starmap) # pool.starmap takes a function and a list of argument tuples and runs them parralel
# starmap parallelization takes the evaluate function and turns it into the input that pool.starmap expects

# Instantiate with concatenated filtered data
problem = RatedPowerCurveProblem(v_all, p_obs_all, rated_all, elementwise_runner = runner)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Configure Differential Evolution
# ──────────────────────────────────────────────────────────────────────────────

algorithm = DE(
    pop_size=50,
    sampling=FloatRandomSampling(),
    variant="DE/rand/1/bin",
    CR=0.7,
    F=0.5
)

# Termination Criteria
# Checks every 5 generations if there's been 0.1% improvement in the last 20, terminate if not
termination = RobustTermination(
    MultiObjectiveSpaceTermination(tol=0.01, n_skip=5),
    period=20
)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Run the optimization (v in [3.25, 10])
# ──────────────────────────────────────────────────────────────────────────────
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True
)
# Fetch best-fit parameters
a_opt, b_opt, c_opt, d_opt, g_opt = res.X
print("\nOptimized 5PL shape parameters (scaled by rated power):")
print(f"  a (bottom fraction) = {a_opt:.4f}")
print(f"  b (slope)          = {b_opt:.4f}")
print(f"  c (midpoint)       = {c_opt:.4f} m/s")
print(f"  d (top fraction)   = {d_opt:.4f}")
print(f"  g (asymmetry)      = {g_opt:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Compute RMSE (3.25–13) for each dataset and mean RMSE
# ──────────────────────────────────────────────────────────────────────────────

rmses = []
print("\nRMSE (3.25-10 m/s) for each file:")
for label, v_fit, p_fit, rated in datasets_fit:
    L_pred_i = logistic5_norm(v_fit, a_opt, b_opt, c_opt, d_opt, g_opt)
    P_pred_i = rated * L_pred_i
    mse_i  = np.mean((p_fit - P_pred_i) ** 2)
    rmse_i = np.sqrt(mse_i)
    rmses.append(rmse_i)
    print(f"  {label:8s}  →  RMSE = {rmse_i:.4f} kW")

mean_rmse = np.mean(rmses)
print(f"\nMean RMSE across all datasets: {mean_rmse:.4f} kW")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Plot each dataset’s raw points and its fitted curve P_hat(v)=rated * L(v)
# ──────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 6))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# (a) Scatter raw data points (all v) for each turbine
for (label, v_full, p_full, rated), c in zip(datasets_full, colors):
    rated_mw = rated / 1000.0
    plt.scatter(
        v_full, p_full,
        s=10,
        color=c,
        alpha=0.4,
        label=f"{rated_mw:.1f} MW Measured Data"
    )

# (b) Overlay the fitted curve for each turbine (only over [3.25, 13])
v_plot = np.linspace(CUT_IN, V_RATED, 300)
for (label, _, _, rated), c in zip(datasets_full, colors):
    rated_mw = rated / 1000.0
    L_plot = logistic5_norm(v_plot, a_opt, b_opt, c_opt, d_opt, g_opt)
    P_plot = rated * L_plot
    plt.plot(
        v_plot, P_plot,
        color=c,
        linewidth=2,
        label=f"{rated_mw:.1f} MW Fitted Model"
    )

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power (kW)")
plt.title("Power Curve Modelvs. Real Turbine Datasets")
plt.legend(loc="upper left", fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# Export data to json file


# 1) Collect the parameters into a dict
params = {
    "a": float(a_opt),
    "b": float(b_opt),
    "c": float(c_opt),
    "d": float(d_opt),
    "g": float(g_opt)
}

# File path
file_name = "power_parm.json"
file_dir = script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(file_dir, file_name)

# Export to file
with open(out_path, "w") as f:
    file.dump(params, f, indent=2)

print(f"Saved fitted parameters to {file_name}")

# Close threads
pool.close()

