import numpy as np
from pymoo.core.problem import ElementwiseProblem # For problem definition
# For algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
# For termination
from pymoo.termination import get_termination
# For optimization
from pymoo.optimize import minimize
# For visualization
import matplotlib.pyplot as plt


'''
Using PyMoo library to solve a bi-objective optimization problem
'''

# PyMoo requires each constraint to be <= 0
# and each objective to be minimized
# Rewrite everything as necesarry

# Objectives:
# min f(x) = 100(x1^2+x2^2)
# min f(x) = (x1-1)^2+x2^2
# Constraints:
# g(x) = (2/0.18)*(x1-0.1)(x1-0.9) <= 0
# g(x) = (-20/4.8)(x1-0.4)(x1-0.6) <= 0
# Bounds: 
# -2 <= x1 <= 2
# -2 <= x2 <= 2
# Search Space:
# x1,x2 E R

# PyMoo Supports 3 types of problem definition
# Vectorized (many solutions in one call)
# Element-wise (evaluate one solution at a time)
# Call back style (custom code optimization)

#
# Element wise problem definition
#
class MyProblem(ElementwiseProblem):
    
    # Constructor (Make class object)
    def __init__(self):
        super().__init__( # Inherit from element wise problem class
            n_var = 2, # Number of vars
            n_obj = 2, # Number of objs
            n_ieq_constr = 2, # Number of inequality constraints
            n_eq_constr = 0, # Number of equality constraints
            xl=np.array([-2, -2]), # Lower bound for both
            xu=np.array([2, 2]) # Upper bound for both
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
        
        # Find value of constraints
        g1 = (2/0.18) * (x[0]-0.1) * (x[0]-0.9)
        g2 = (-20/4.8) * (x[0]-0.4) * (x[0]-0.6)
        
        # Output to dictionary
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

problem = MyProblem() # Create instance of object

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
# PyMoo's parameters are usually well balanced, use standard parameters usually, just edited this time for practice

#
# Define Termination Criterion
#

termination = get_termination("n_gen", 40) # Terminate after 40 generations
# Small termination criteria since the problem is relatively simple

# Can use a convergence analysis to see how much progress is made each generation
# If termination criteria isn't defined, pymoo automatically terminates after progress stops being made

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
# Visualize 
#

# Plot of solution sets against the bounds of x1 and x2
xl, xu = problem.bounds() # sets xl to a numpy array [-2, -2] and xu to [2, 2]
plt.figure(figsize=(7,5)) # Create a blank window 7" by 5" tall
plt.scatter(
    X[:, 0], # All values in first column as horizontal axis (x1)
    X[:, 1], # Same but for vertical axis (x2)
    s=30, # Marker size
    facecolors='none', # No face colour, hollow circles
    edgecolors='r' # Red outline for edge
)
plt.xlim(xl[0], xu[0]) # set x axis bounds
plt.ylim(xl[1], xu[1]) # set y axis bounds
plt.title("Design Space")
plt.show()

# Plot of obj 1 vs obj 2 in solutions
plt.figure(figsize=(7,5))
plt.scatter(
    F[:, 0],
    F[:, 1],
    s=30,
    facecolors='none',
    edgecolors='blue'
)
plt.title("Objective Space")
plt.show()