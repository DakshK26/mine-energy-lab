import numpy as np
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization # For problem definition
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

# Function for objective 3
def f3_de(a: float, b: float, y0: float, T: float)->float:
    '''
    y0 : Initial Condition
    t : Time to integrate to
    '''
    
    # Make an ode to find y at a given time t. scipy treats t as independant variable and y as dependant. Expects y' as a result
    def ode(t, y):
        return a*y+b
        
    
    solution = scipy.integrate.solve_ivp(
        fun=ode, # Right hand side of system, 
        t_span=(0, T), # Interval of integration
        y0=[y0], # Initial value (Turned into 1d array)
        t_eval=[T] # Tells it to give solution only at time t
    )
    # Soluton returns .t and .y. solution.y will return a 2d array of shape (n_eqns, n_times) 
    # so it returns y(t) for equation 0 and at time (-1) which is the last index
    
    return solution.y[0, -1]

#
# Element wise problem definition
#
class MyProblem(ElementwiseProblem):
    
    # Constructor (Make class object)
    def __init__(self, **kwargs):
        super().__init__( # Inherit from element wise problem class
            n_var = 2, # Number of vars
            n_obj = 3, # Number of objs
            n_ieq_constr = 2, # Number of inequality constraints
            n_eq_constr = 0, # Number of equality constraints
            xl=np.array([-2, -2]), # Lower bound for both
            xu=np.array([2, 2]), # Upper bound for both
            **kwargs # Allows passing elementwise runner here
        )
    
    # Method for evaluation
    def _evaluate(self, x, out, *args, **kwargs):
        # Self just means I'm referring to my object (allows me to refer to object attributes)
        # x is a 1-D NumPy array of length n_var (2 in this case)
        # represents a single set of solutions to be tested
        # out is a python dictionary that takes objectives and constrains for pymoo to use
        # Find value of objectives
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2
        f3 = f3_de(x[0], x[1], 1, 2) 

        
        # Find value of constraints
        g1 = (2/0.18) * (x[0]-0.1) * (x[0]-0.9)
        g2 = (-20/4.8) * (x[0]-0.4) * (x[0]-0.6)
        
        # Output to dictionary
        out["F"] = [f1, f2, f3]
        out["G"] = [g1, g2]

#
# Parallelization
# Python automatically runs on only one core, need to parallelize to run on more
#
threads = 4
pool = ThreadPool(threads) # Creates a pool of worker threads
runner = StarmapParallelization(pool.starmap) # pool.starmap takes a function and a list of argument tuples and runs them parralel
# starmap parallelization takes the evaluate function and turns it into the input that pool.starmap expects

# Elementwise runner is now set to the runner that will parallelize instead of the basic one
problem = MyProblem(elementwise_runner=runner) # Create instance of object

#
# Algorithm Initialization
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
# Checks every 5 generations if there's been 2% improvement in the last 20, terminate if not
termination = RobustTermination(
    MultiObjectiveSpaceTermination(tol=0.02, n_skip=5),
    period=20
)

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

approx_ideal = F.min(axis=0) # Looks at each column and picks the smallest value [min f1, min f2]
approx_nadir = F.max(axis=0) # Same but w/ max

# Each objective gets normalized
nF = (F - approx_ideal) / (approx_nadir - approx_ideal) 

#
# Decomposition via ASF
# A method to find the best solution based on weights
# As in, if you think objective one is worth 20% and obj 2 is worth 80%
# It'll find the right solution based on that
#

# Define weights
weights = np.array([0.7, 0.15, 0.15])

# Get best solution
decomp = ASF()
# Decomp works with ASF(f,w) = max(i, (fi-zi*)/wi)) + small augmentation term
i = decomp.do(nF, 1/weights).argmin()

print("Best ASF index:", i)
print("Objectives:", F[i])
print("Chosen x1, x2: ", X[i])