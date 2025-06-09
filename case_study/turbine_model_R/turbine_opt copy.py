import json as file
import os
import numpy as np
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization # For problem definition
# For algorithm
from pymoo.algorithms.soo.nonconvex.de import DE
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
# For diff eq and integration
import scipy
import scipy.integrate
# For Parralelization
from multiprocessing.pool import ThreadPool
import nrel_model_onshore as turbine

'''
Main function to optimize for AOE and Cost.
Configured for rated power from 2.3MW to 7MW. From radius of: 56.5m -> 100m. Hub Height from: 70 -> 200
Cost function must be in W and Wh/yr
Power function is in kw
Cost of diesel comparison
'''

#
# Element wise problem definition
#
class TurbineModel(ElementwiseProblem):
    
    # Constructor 
    def __init__(self, **kwargs):
        super().__init__( # Inherit from element wise problem class
            n_var = 3, # Number of vars (R, P_r, Hm)
            n_obj = 1, # Number of objs
            n_ieq_constr = 1, # Number of inequality constraints
            n_eq_constr = 0, # Number of equality constraints
            xl=np.array([56.5, 2300, 70]), # Lower bound for both (Let R be first var) P_R in kW
            xu=np.array([100, 7000, 200]), # Upper bound for both
            **kwargs # Allows passing elementwise runner here
        )
        
    # Method for evaluation
    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate as single objective w cost of energy
        R, P_r, Hm = x
        AEP = turbine.annual_energy_production(P_r, R, Hm) # Leave drive train as standard
        cost = turbine.calculate_cost(R, P_r*1000, Hm, AEP*1000) # Leave fix charged rate as standard (P_r in W)
        f1 = turbine.costOfEnergy(cost, AEP) # In USD/kWh

        
        # Find value of constraints (PyMoo makes them <= 0, make -ve to make it > 0)
        C_p = turbine.calc_cp(turbine.V_RATED, P_r) # Takes kW input
        # Constraint to make sure that at V_rated, the turbine would actually be capable of producing the rated power, meaning
        # calculated power - P_r, if that's >= 0, turbine is capable of producing rated power.
        # For pymoo, convert to P_r - calc power
        g1 = P_r - turbine.calc_power_mech(R, turbine.V_RATED, C_p, rho0=1.225*turbine.density_adj(Hm))/1000 # Convert to kW 
        
        # Output to dictionary
        out["F"] = [f1]
        out["G"] = [g1]
        
        
# Parralelize and make instance of problem
threads = 12
pool = ThreadPool(threads) 
runner = StarmapParallelization(pool.starmap)

problem = TurbineModel (elementwise_runner=runner) # Create instance of object

#
# Algorithm Initialization
# Use differential evolution algo since it'll converge fast on a continous space problem
#
algorithm = DE(
    pop_size=50,           # moderate population
    F=0.6, # The scale factor, represents how far away from current solutions the next generation goes. Further away helps reach global best solution
    #, but could slow down progress. Usually in 0.5-0.8 range
    CR=0.9 # Cross over factor, is done by taking portions of target solution (a current gen solution) and the mutant solution
    # a solution made by mixing current gen solutions like: v = x1 + F(x2+x3) is a form, where xi are current gen solutions,
    # 0.9 means that 90% of the new solution is taken from v
)

#
# Termination
#
# Checks every 5 generations if there's been 0.25% improvement in the last 20, terminate if not
termination = RobustTermination(
    MultiObjectiveSpaceTermination(tol=0.0025, n_skip=5),
    period=20
)

# Generate the results
res = minimize(
    problem, # Pass in problem, algorithm, and termination definitions
    algorithm,
    termination,
    seed = 1, # Fix a random seed (allows us to get same results running the code multiple times)
    save_history = False, # Store population at each generation
    verbose = True # Print progress to the console each generation
)

R, P_r, Hm = res.X
CostOfEnergy = res.F
PowerAboveReated = -res.G

print(f"Best Radius: {R}")
print(f"Best Rated Power: {P_r} kW")
print(f"Hub Height: {Hm}")
print(f"Cost of Energy: {CostOfEnergy} USD/kWh")
print(f"Power Above Rated: {PowerAboveReated}")

# Save results
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path   = os.path.join(script_dir, "optimized_turbine.json")

results = {
    "Best Radius (m)"      : R,
    "Best Rated Power (kW)": float(P_r),
    "Hub Height (m)"       : Hm,
    "Cost of Energy (USD/kWh)": float(CostOfEnergy[0])
}

with open(out_path, "w") as jf:
    file.dump(results, jf, indent=2)

print(f"\nSaved results to {out_path}")

pool.close() # Close threads