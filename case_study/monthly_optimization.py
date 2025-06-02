from turbine_modelling import nrel_onshore_model as turbine
import numpy as np
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization # For problem definition
import pandas as pd
# For algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
# For basic termination
from pymoo.termination import get_termination
# For robust termination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
# For optimization
from pymoo.optimize import minimize
# For visualization
import matplotlib.pyplot as plt
# For decomp mcdm
from pymoo.decomposition.asf import ASF
# For diff eq and integration
import scipy
import scipy.integrate
# For Parralelization
from multiprocessing.pool import ThreadPool
import click as ui

# Include Drivetrain efficiency end of this file
# Prompt for month first
month = ui.prompt("Month ", default="January")

# Load the CSV into a DataFrame
df = pd.read_csv("case_study/eureka_wind_stats.csv")

# Select the row matching that month
row = df.loc[df["Month"] == month]

# Get k and c if its there
if row.empty:
    raise ValueError(f"No data found for month '{month}'")
else:
    first = row.iloc[0]
    k = first["k"]
    c = first["c"]

sea_density = ui.prompt("Sea level density: ", default=1.225, type=float)


#
# Element wise problem definition
#
class TurbineOptimal(ElementwiseProblem):
    
    # Constructor (Make class object)
    def __init__(self, **kwargs):
        super().__init__( # Inherit from element wise problem class
            n_var = 2, # Number of vars: R, P_r
            n_obj = 2, # Number of objs: 2 - power and cost
            n_ieq_constr = 0, # Number of inequality constraints
            n_eq_constr = 0, # Number of equality constraints
            xl=np.array([0, 3000]), # Lower bound for both (R b/w 0 and 100) (P_r b/w 3000 and 7000 kW)
            xu=np.array([100, 7000]), # Upper bound for both
            **kwargs # Allows passing elementwise runner here
        )
    
    # Method for evaluation
    def _evaluate(self, x, out, *args, **kwargs):
        R, P_r = x
        
        # For Power:
        Hm = turbine.calc_height(R)
        f1 = turbine.mean_turbine_power(k, c, sea_density, R, Hm, P_r)

        # For cost:
        AEP = f1*8760/1000*0.95 # Take 95% NREL efficiency
        f2 = turbine.turbine_cost(P_r, R, Hm, AEP)
        
        # Output to dictionary
        out["F"] = [-f1, f2]
        
#
# Parallelization
# Python automatically runs on only one core, need to parallelize to run on more
#
threads = 8
pool = ThreadPool(threads) # Creates a pool of worker threads
runner = StarmapParallelization(pool.starmap) # pool.starmap takes a function and a list of argument tuples and runs them parralel
# starmap parallelization takes the evaluate function and turns it into the input that pool.starmap expects

# Elementwise runner is now set to the runner that will parallelize instead of the basic one
problem = TurbineOptimal(elementwise_runner=runner) # Create instance of object

#
# Create algorithm
#
algorithm = NSGA2(
    pop_size = 40, # Total # of solutions in each generation
    n_offsprings = 10, # New solutions generated per iteration
    sampling = FloatRandomSampling(), # Method to generate solutions
    crossover = SBX(prob=0.9, eta=15), # Simulated binary crossover, 90# of selected parents will exchange info, eta determines how spead out the parents are around their parents
    mutation=PM(eta=20), # How much mutation happens from last generation to this
    eliminate_duplicates = True # Removes duplicate solutions
)

#
# Termination
#
# Checks every 5 generations if there's been 1% improvement in the last 20, terminate if not
termination = RobustTermination(
    MultiObjectiveSpaceTermination(tol=0.01, n_skip=5),
    period=20
)

#
# Optimize
#

# Generate the results
res = minimize(
    problem, # Pass in problem, algorithm, and termination definitions
    algorithm,
    termination,
    seed = 1, # Fix a random seed (allows us to get same results running the code multiple times)
    save_history = True, # Store population at each generation
    verbose = True # Print progress to the console each generation
)

# Results
X = res.X # Returs 2-D array of the decision-variable vectors for final set
F = res.F # a 2-D array of corresponding object values
History = res.history # A list of intermediate populations and their stats

#
# MCDM
#

# Normalize the two objectives
# Ideal point is the best possible on each objective (smallest)
# Nadir point is worst value on each objective (largest)
ideal = F.min(axis=0)
nadir = F.max(axis=0)
# Normalized objectives
F_norm = (F - ideal) / (nadir - ideal)

# Define weights
weights = np.array([0.5, 0.5])

# Get best solution
decomp = ASF()
# Decomp works with ASF(f,w) = max(i, (fi-zi*)/wi)) + small augmentation term
i = decomp.do(F_norm, 1/weights).argmin()

# Get annual energy
P_mean = -F[i][0]
costs = F[i][1]
R = X[i][0]
P_r = X[i][1]
AEP = P_mean * 8760/1000

# Plot Power Generation over Month
# 12 difures - Show Weibull PDF 

print(f"For {month}")
print(f"Radius: {R}")
print(f"Rated Power: {P_r} kW")
print(f"Mean Turbine Power Output (Mechanical): {P_mean} kW")
print(f"Monthly Energy Production: {AEP/12} MWh/month")
print(f"Cost for Month: ${costs/12}")