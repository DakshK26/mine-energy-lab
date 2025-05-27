import numpy as np
from pymoo.core.problem import ElementwiseProblem # For problem definition
# For algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
# For robust termination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
# For optimization
from pymoo.optimize import minimize
# For visualization
import matplotlib.pyplot as plt
# For decomp mcdm
from pymoo.decomposition.asf import ASF

# For ui
import click as ui

import Wind_Turbine as wt

# Get values, all base cases based on research on HAWT
density_sea = ui.prompt("Sea level density: ", default=1.225, type=float) # ISA Standard Value
v_mean = ui.prompt("Mean wind speed: ", default=8, type=float) # Mean wind speed
c1 = ui.prompt("C1: ", default=0.5, type=float)
c2 = ui.prompt("C2: ", default=116, type=float)
c3 = ui.prompt("C3: ", default=0.4, type=float)
c4 = ui.prompt("C4: ", default=0, type=float)
c5 = ui.prompt("C5: ", default=5, type=float)
c6 = ui.prompt("C6: ", default=21, type=float)
x_exp = ui.prompt("X: ", default=1, type=float) # Exponent for Weibull distribution
k = ui.prompt("K: ", default=2.0, type=float) # Shape parameter for Weibull distribution
v_min = ui.prompt("Min wind speed: ", default=2, type=float)
v_max = ui.prompt("Max wind speed: ", default=30, type=float)
tipSpeed_Lower = ui.prompt("Tip speed lower bound: ", default=4, type=float)
tipSpeed_Upper = ui.prompt("Tip speed upper bound: ", default=12, type=float)
bladePitch_Lower = ui.prompt("Blade pitch lower bound: ", default=-1, type=float)
bladePitch_Upper = ui.prompt("Blade pitch upper bound: ", default=10, type=float)
radius_Lower = ui.prompt("Radius lower bound: ", default=1, type=float)
radius_Upper = ui.prompt("Radius upper bound: ", default=100, type=float)
fcr = ui.prompt("FCR: ", default=0.08, type=float) # Fixed charge rate



# Element wise problem definition
# Objectives: Minimize power and cost
# Constraints: No function
# Bounds: as stated above
# Search Space: All values are real numbers

class MyProblem(ElementwiseProblem):

    def __init__(self):
            
        super().__init__( # Inherit from element wise problem class
            n_var = 3, # Number of vars (radius, blade pitch, tip speed)
            n_obj = 2, # Number of objs
            n_ieq_constr = 0, # Number of inequality constraints
            n_eq_constr = 0, # Number of equality constraints
            xl=np.array([radius_Lower, bladePitch_Lower, tipSpeed_Lower]), # Lower bound for all
            xu=np.array([radius_Upper, bladePitch_Upper, tipSpeed_Upper]) # Upper bound for all
        )
            
    # Method for evaluation
    def _evaluate(self, x, out, *args, **kwargs):
        Hm = wt.calc_height(x[0])
        
        # Max mean turbine power (-ve to minimize)
        f1 = wt.mean_turbine_power(
            c=wt.calc_scale(v_mean, k), # Weibull scale parameter
            rho0=density_sea,
            R=x[0], # Radius
            Hm=Hm, # Height
            lam=x[2], # Tip speed
            theta=x[1], # Blade pitch
            C1=c1,
            C2=c2,
            C3=c3,
            C4=c4,
            C5=c5,
            C6=c6,
            x=x_exp,
            k=k,
            v_min=v_min,
            v_max=v_max,
            n=1000
        )
        
        f2 = wt.turbine_cost(f1/1e3, x[0], Hm, f1*8760/1e6, fcr) # Cost of turbine
        
        # Output to dictionary
        out["F"] = [-f1, f2]



problem = MyProblem() # Create instance of object

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

print("Best ASF index:", i)
print("Objectives:", F[i])
print("Chosen x1, x2, x3: ", X[i])
