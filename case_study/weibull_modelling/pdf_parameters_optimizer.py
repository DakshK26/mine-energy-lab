import numpy as np
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization # For problem definition
# For algorithm
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
# For termination
from pymoo.termination import get_termination
# For optimization
from pymoo.optimize import minimize
# For visualization
import matplotlib.pyplot as plt
# For decomp mcdm
from pymoo.decomposition.asf import ASF
# For diff eq and integration
import scipy
import scipy.integrate
from scipy.integrate import quad
# For Parralelization
from multiprocessing.pool import ThreadPool
import click as ui

import pdf_helper as helper

'''
Following general layout from: https://www.nature.com/articles/s41598-025-90583-2#Sec2 to find k and c 
Aim for RMSE around 0.015 or lower: https://www.sciencedirect.com/science/article/pii/S2352484721010969
'''

# RMSE Calculator
def RMSE(k, c, counts, edges) -> float:
    
    # Weibull CDF: lamba functions are small functions that take just one expression, the equation is the equation for CDF
    cdf = lambda v: 1 - np.exp(-(v/c)**k)
    
    # Since CDF is integral of PDF, the exact value for integral of PDF can be found by CDF(i)-CDF(i) where i is bin edge positions
    num = cdf(edges[1:]) - cdf(edges[:-1])
    width = edges[1:] - edges[:-1] # This gives the width of each bin
    pdf_avg = num / width # The pdf average will give amount of entries in each histogram entry
    
    # RMSE between your measured density and the exact average
    return np.sqrt(np.mean((counts - pdf_avg)**2))
    

# Problem definition 
class PDF_Parameter_Problem(ElementwiseProblem):
    
    # Constructor
    def __init__(self, speeds, hist_counts, bin_edges, **kwargs):
        # define as subclass of element wise problem
        super().__init__(
            n_var = 2, # k and c
            n_obj = 1, # Minimizing the root mean square error
            n_ieq_constr = 0,
            n_eq_constr = 0,
            xl = np.array([1,1]), # k and c must be > 0
            xu = np.array([10, 30]), # relatively high upper bounds to give room to work with
            **kwargs    
        )
        self.speeds = speeds
        self.counts = hist_counts
        self.edges = bin_edges

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a 1-d array of length 2 in this case
        k, c = x
        f = RMSE(k, c, self.counts, self.edges)
        
        out["F"] = [f]
        
#
# Parallelization
#
threads = 8
pool = ThreadPool(threads) # Creates a pool of worker threads
runner = StarmapParallelization(pool.starmap) # pool.starmap takes a function and a list of argument tuples and runs them parralel
# starmap parallelization takes the evaluate function and turns it into the input that pool.starmap expects

# Get speeds
month = ui.prompt("Month: ")
speeds = helper.speedsAsArray(month)
counts, edges = np.histogram(speeds, bins='auto', density=True) # Density histogram so frequencies sum to 1. Counts is number of entries in each bin, edges are positions of bin boundaries


# Elementwise runner is now set to the runner that will parallelize instead of the basic one
problem = PDF_Parameter_Problem(speeds, counts, edges, elementwise_runner=runner) # Create instance of object


#
# Algorithm Initialization
# Single objective algorithm
#
algorithm = DE(
    pop_size=50,
    variant="DE/rand/1/bin",
    CR=0.9,
    F=0.8
)

# number generations ran
n_gen = 2000

# Simple termination after 200 gens
termination = get_termination("n_gen", n_gen)

# Generate the results
res = minimize(
    problem, # Pass in problem, algorithm, and termination definitions
    algorithm,
    termination,
    seed = 1, # Fix a random seed (allows us to get same results running the code multiple times)
    save_history = False, # Store population at each generation
    verbose = True # Print progress to the console each generation
)

# Results
best_k, best_c = res.X # Returs 2-D array of the decision-variable vectors for final set
best_rmse = res.F[0]

print(f"k: {best_k}\nc: {best_c}\nRMSE: {best_rmse}")
print(f"Mean wind speed: {np.mean(speeds)}")
print(f"Standard Deviation: {np.std(speeds)}")
print(f"Max Speed: {helper.v_max_solver(best_k, best_c)}")
print(f"Generations Ran: {n_gen}")


# ──────────────────────────────────────────────────────────────────────────────
# 8) Plot histogram + fitted Weibull PDF
# ──────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))

# (a) Plot the empirical histogram (density=True so area = 1)
plt.hist(
    speeds,
    bins=edges,
    density=True,
    alpha=0.4,
    color='skyblue',
    edgecolor='gray',
    label='Empirical histogram'
)

# (b) Compute the continuous Weibull PDF on a fine grid
v_grid = np.linspace(edges[0], edges[-1], 200)
weib_pdf = (best_k / best_c) * (v_grid / best_c) ** (best_k - 1) * np.exp(- (v_grid / best_c) ** best_k)

# (c) Overlay the fitted Weibull curve
plt.plot(
    v_grid,
    weib_pdf,
    'r-',
    linewidth=2,
    label=f'Weibull PDF (k={best_k:.3f}, c={best_c:.3f})'
)

plt.xlabel("Wind speed (m/s)")
plt.ylabel("Probability density")
plt.title("Weibull fit vs. empirical histogram")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
