import random

'''
An simple genetic algorithm to develop a solution to a single objective problem
'''


def foo(x, y, z):
    return 6*x**3 + 9 * y ** 2 + 90 * z - 25
# 6*x**3 + 9 * y ** 2 + 90 * z is the 
# function we're aiming to optimize when it's equal to 25
# So, in Foo, sub 25 and then we can aim for the solutions 
# that are closer the 0 the better they are


# Make a fitness function. Tests how will the current solution fits ideal
def fitness(x, y ,z):
    ans = foo(x,y,z)
    
    if ans == 0:
        return 99999
    else:
        return abs(1/ans) # A way of ranking our solutions,
    # the closer we are to 0 the higher this solution is in rank

# Generate 1000 initial solutions
solutions = [] # Will be a list of tuples. Tuples of form (x,y,z)
for s in range(1000):
    solutions.append( (random.uniform(0,10000), 
                       random.uniform(0,10000), 
                       random.uniform(0,10000)) )

# Create generations, max number of generations is 10000
for i in range(10000):
    rankedSolutions = []
    for s in solutions: # Iterate through initial solutions
        # Append the rank of the solutions and solution as a tupple
        rankedSolutions.append( (fitness(s[0], s[1], s[2]), s) )
    rankedSolutions.sort() # Sorts in ascending order
    rankedSolutions.reverse()  # Reverse for descending
    print(f"=== Gen [i] Best Solution ===")
    print(rankedSolutions[0])
    
    if rankedSolutions[0][0] > 999:
        break
    
    # Combine the best solutions to make new gen (Swapping chromosones)
    bestSolutions = rankedSolutions[:100]
    # Extract x,y, elements from best solutions
    goodX = []
    goodY = []
    goodZ = []
    for s in bestSolutions:
        goodX.append(s[1][0])
        goodY.append(s[1][1])
        goodZ.append(s[1][2])
    
    # Create new generation by taking elements from exctracted elements
    newGen = []
    for _ in range(1000):
        e1 = random.choice(goodX) * random.uniform(0.99, 1.01)
        e2 = random.choice(goodY) * random.uniform(0.99, 1.01) # Mutate solutions by 2% for diversity
        e3 = random.choice(goodZ) * random.uniform(0.99, 1.01)
        
        newGen.append( (e1, e2, e3) )
        
    solutions = newGen 
    

        
        
    
        

