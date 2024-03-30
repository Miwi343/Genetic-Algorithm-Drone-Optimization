from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import random
from time import perf_counter

###################################################################################################
def format(strategy):
    # round each strategy component to two decimal places and return as a string
    str_strat = ', '.join( [f"{round(i, 2)}" for i in strategy] )
    str_strat = '[' + str_strat[:-2] + ']'
    return str_strat

###################################################################################################
def calculate_coverage(drones, coverage_map):
    """
    Update the coverage_map based on drones' positions and predefined coverage radius.
    Assumes drones have a circular coverage area on the XY plane depending on their altitude.
    """
    coverage_radius = 0.5  # Define the coverage radius, adjust based on your needs
    for drone in drones:
        x, y, z = drone[:3]  # Use only position information
        # Convert drone x, y to coverage map indices
        ix, iy = int(x * 100), int(y * 100)  # Assuming coverage_map is 500x500 for 5x5 area
        # Simple coverage update, refine to account for actual coverage area
        coverage_map[ix:ix+2, iy:iy+2] = 1

###################################################################################################
def update_plot(ax, drones):
    ax.clear()
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    for drone in drones:
        ax.scatter(*drone[:3], color='blue')  # Plot using only position information
    plt.draw()

###################################################################################################
def fitness_function(strategy):
    energy = 100
    coverage_map = np.zeros((500, 500))  # Example resolution for a 5x5 area

    print(f"Strategy: {format(strategy)}")

    # Extract drones' information from strategy
    drones = [np.array(strategy[i:i+5]) for i in range(0, len(strategy), 5)]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.ion()
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 5])
    # ax.set_zlim([0, 5])

    while energy > 0:
        for i, drone in enumerate(drones):
            theta, v = drone[3], drone[4]  # Extract angle and velocity
            # Calculate movement based on angle and speed
            dx = v * np.cos(theta)
            dy = v * np.sin(theta)
            move_direction = np.array([dx, dy, 0])  # No vertical movement
            new_pos = drone[:3] + move_direction  # Update position
            # Keep within bounds
            new_pos = np.clip(new_pos, 0, 5)
            drones[i][:3] = new_pos
            energy -= v  # Update energy based on movement

        calculate_coverage([d[:3] for d in drones], coverage_map)  # Pass only position information
        
        # plt.pause(0.1)


    coverage = np.sum(coverage_map) / (500*500) * 25  # Convert to area covered in the 5x5 grid
    # print(f"Final coverage: {coverage} square units, Energy remaining: {energy}")
    return coverage

###################################################################################################
def breed( parent1, parent2 ):
    # Randomly select a crossover point
    crossover_point = np.random.randint(0, len(parent1))
    
    # Create the offspring
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    
    return child1

###################################################################################################
def mutate( child ):
    for index in len(child):       # mutate each gene in the child
        if np.random.random() < MUTATION_RATE:
            mutation_deviation = np.random.uniform(-0.5, 0.5)
            child[index] += mutation_deviation
            
            if index < 3:
                child[index] = np.random.uniform(0, 5)
            elif index == 3:
                child[index] = np.random.uniform(0, 2*math.pi)
            else:
                child[index] = np.random.uniform(0, 2)
                
    return child

###################################################################################################
def genetic_algorithm( population, generation_count ):
    
    start = perf_counter()
    # Evaluate the fitness of the population, store in a dict with the keys being the strategy vectors
    fitness = dict()
    generation_sum = 0
    
    for strategy in population:
        generation_score = fitness_function(strategy)
        
        generation_sum += generation_score
        fitness[strategy] = generation_score
    
    # Rank the strategy vectors from most fit to worst
    # create a sorted_dict that is in descending order of fitness (values of the dictionary)
    sorted_dict = dict(sorted( fitness.items(), key=operator.itemgetter(1), reverse=True ))
    ranked_strategy_list = []
        
    # Print generation results:
    for i, (strategy, score) in enumerate(sorted_dict.items()):
        print(f"Strategy {i+1}: {format(strategy)} with score {score}")
        ranked_strategy_list.append(list(strategy))
        
    end = perf_counter()
    
    print(f"Generation {generation_count} average:  {generation_sum / len(population)}.")
    print(f"{end-start} seconds elapsed to calculate fitness.")
    print(f"-----------------------------")
    input()

    # Repeat until a new generation is created of the same size as the original population with unique strategy vectors
    children_generation = set()
    
    while len(children_generation) < len(population):
        # Randomly select parents based on their fitness going down the list of most fit strategies
        parent1, parent2 = None, None
        
        for index in range(len(ranked_strategy_list)):
            if np.random.random() < PARENT_CHANCE:
                if parent1 == None:
                    parent1 = ranked_strategy_list[index]
                else:
                    parent2 = ranked_strategy_list[index]
                    break
        
        child1 = tuple( breed(parent1, parent2) )
        children_generation.add(child1)
        
    children_generation = list(children_generation)
    
    generation_count += 1
    genetic_algorithm( children_generation, generation_count)
        
    
        


        
    
        
    
    
    
    

###################################################################################################

MUTATION_RATE = 0.1  # Adjust as needed
REPRODUCTION_PROBABILITY = 0.5  # Adjust as needed
POPULATION_SIZE = 100  # Adjust as needed
PARENT_CHANCE = 0.4  # Adjust as needed

PARENT_WEIGHTS = [POPULATION_SIZE - i for i in range(POPULATION_SIZE)] #used to select parents based on their fitness

# Format: [x1, y1, z1, θ1, v1, x2, y2, z2, θ2, v2, ...]
population = []
for i in range(POPULATION_SIZE):
    strategy = [np.random.uniform(0, 5), np.random.uniform(0, 5), np.random.uniform(0, 5), np.random.uniform(0, 2*math.pi), np.random.uniform(0, 2)]  # Example strategy vector
    # 0 - 5 for x, y, z, 0 - 2pi for angle, 0 - 2 for velocity
    population.append(tuple(strategy))
    
genetic_algorithm(population, 0)