from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import random
from time import perf_counter
import pickle
from tqdm import tqdm

###################################################################################################
def polar_to_rectangular( polar_coordinates ):
    ro, theta, phi = polar_coordinates
    x = ro * math.sin(phi) * math.cos(theta)
    y = ro * math.sin(phi) * math.sin(theta)
    z = ro * math.cos(phi)
    return [x, y, z]

###################################################################################################
def format(strategy):
    # round each strategy component to two decimal places and return as a string
    str_strat = ', '.join( [f"{round(i, 2)}" for i in strategy] )
    str_strat = '[' + str_strat[:-2] + ']'
    return str_strat

###################################################################################################
def update_coverage(drones, coverage_grid, grid_scale=5):
    """
    Updates the coverage grid based on the position and coverage radius of each drone.
    
    :param drones: List of drones, each represented by its polar coordinates and movements.
    :param coverage_grid: 2D NumPy array representing the coverage grid.
    :param grid_scale: The scale of the grid (how much physical space one grid unit represents).
    """
    grid_size = coverage_grid.shape  # Assuming coverage_grid is a NumPy array
    for drone in drones:
        drone_x, drone_y, drone_z = polar_to_rectangular(drone[:3])
        radius = drone_z / math.sqrt(3)  # Coverage radius in the same units as drone_x and drone_y
        
        # Convert drone coordinates and radius to grid indices
        drone_grid_x = int((drone_x / grid_scale) * grid_size[0])
        drone_grid_y = int((drone_y / grid_scale) * grid_size[1])
        radius_in_grid_units = int((radius / grid_scale) * grid_size[0])  # Assuming square cells and uniform scaling
        
        # Calculate the bounds of the drone's coverage area
        x_min = max(0, drone_grid_x - radius_in_grid_units)
        x_max = min(grid_size[0], drone_grid_x + radius_in_grid_units + 1)
        y_min = max(0, drone_grid_y - radius_in_grid_units)
        y_max = min(grid_size[1], drone_grid_y + radius_in_grid_units + 1)
        
        # Iterate only within the bounds of the coverage area
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                # Check if the cell is within the grid's bounds
                if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                    # Update the coverage grid with a decreasing factor for each additional coverage
                    coverage_grid[x, y] += 1 / (coverage_grid[x, y] * drone_z)
    
    return coverage_grid

###################################################################################################
def fitness_function(strategy):
    
    # First, extract drones' information from strategy
    drones = []
    for i in range(NUMBER_OF_DRONES):
        # Extract drones' positions from strategy
        drones.append( strategy[i*6:i*6+6] )        
    
    coverage_grid = np.ones((GRID_WIDTH, GRID_HEIGHT))  # Initialize coverage grid
            
    energy = INITIAL_ENERGY_LEVEL
    t = 0
    while energy > 0:
        t = t + 1
        # move drones
        for i in range(NUMBER_OF_DRONES):
            drone = drones[i]
            # input(f"Drone {i+1} is at {polar_to_rectangular(drone[:3])}")
                
            # change position of spherical coordinate
            drone[0] += drone[3] # ro = ro + dro
            drone[1] += drone[4] # theta = theta + dtheta
            drone[2] += drone[5] # phi = phi + dphi
            
            # input(f"Drone {i+1} is now at {polar_to_rectangular(drone[:3])}")
            
            # update energy based on movement
            # calculate using polar displacement formula: ds = (dro)^2 + (ro)^2*(dtheta)^2 + (ro)^2*(sin(theta))^2*(dphi)^2
            ds = math.sqrt(drone[3]**2 + (drone[0]**2)*(drone[4]**2) + (drone[0]**2)*(math.sin(drone[1]))**2*(drone[5]**2))
            energy -= ds    
            
            # Update coverage based on drones' positions
            coverage_grid = update_coverage(drones, coverage_grid)        
        
    return np.sum(coverage_grid) / (GRID_WIDTH * GRID_HEIGHT)  # Average coverage of each cell

###################################################################################################
def breed( parent1, parent2 ):
    # Randomly select a crossover point
    crossover_point = np.random.randint(0, len(parent1))
    
    # Create the offspring``
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    
    return child1

###################################################################################################
def mutate( child ):
    for index in range(NUMBER_OF_DRONES):       # mutate each gene in the child
        drone = child[index*6:index*6+6]
        
        if np.random.random() < MUTATION_RATE:
            drone[0] += np.random.uniform(-0.5, 0.5)    # deviate ro
        if np.random.random() < MUTATION_RATE:
            drone[1] += np.random.uniform(-1, 1)        # deviate theta
        if np.random.random() < MUTATION_RATE:
            drone[2] += np.random.uniform(-0.5, 0.5)    # deviate phi
        # if np.random.random() < MUTATION_RATE:          
        #     drone[3] += np.random.uniform(-0.1, 0.1)    # deviate dro
        if np.random.random() < MUTATION_RATE:          
            drone[4] += np.random.uniform(-0.1, 0.1)    # deviate dtheta
        # if np.random.random() < MUTATION_RATE:
        #     drone[5] += np.random.uniform(-0.1, 0.1)    # deviate dphi
                
    return child

###################################################################################################
def display(strategy, input_energy_level):
    
    # Create a 3D plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.ion()
    
    # First, extract drones' information from strategy
    drones = []
    for i in range(NUMBER_OF_DRONES):
        # Extract drones' positions from strategy
        drones.append( strategy[i*6:i*6+6] )        
    
    energy = input_energy_level
    t = 0
    while energy > 0:
        t = t + 1
        drone = drones[i]
        
        ax.clear()
        ax.set_title(f"Energy: {energy}") 
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_zlim([0, 5])
        
        # move drones
        for i in range(len(drones)):
            # input(f"Drone {i+1} is at {drone} and moving {drone_moving}.")
                
            # change position of spherical coordinate
            drone[0] += drone[3] # ro = ro + dro
            drone[1] += drone[4] # theta = theta + dtheta
            drone[2] += drone[5] # phi = phi + dphi
            
            ax.scatter( *polar_to_rectangular(drone[:3]), color='blue')  # Plot using only position information
            plt.draw()
            plt.pause(0.05)
            
            # input(f"Drone {i+1} is now at {drone}.")
            
            # update energy based on movement
            ds = math.sqrt(drone[3]**2 + (drone[0]**2)*(drone[4]**2) + (drone[0]**2)*(math.sin(drone[1]))**2*(drone[5]**2))
            energy -= ds
    
    plt.ioff()
    plt.show()

            


###################################################################################################
def genetic_algorithm( population, generation_count ):
    
    start = perf_counter()
    # Evaluate the fitness of the population, store in a dict with the keys being the strategy vectors
    fitness = dict()
    generation_sum = 0
    
    # as we find the fitness for each strategy, update progress bar using tqdm
    for strategy in tqdm(population):
        generation_score = fitness_function( list(strategy) )
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
    
    best_strategy = ranked_strategy_list[0]
    best_strategy_score = fitness[tuple(ranked_strategy_list[0])]
        
    end = perf_counter()
    
    print(f"-----------------------------")
    print(f"Generation {generation_count} average:  {generation_sum / len(population)}.")
    print(f"Best strategy: {format(best_strategy)} with score {best_strategy_score}")
    print(f"{end-start} seconds elapsed to calculate fitness.")
    print(f"-----------------------------")
    
    # Check if user wants to visualize best strategy
    while True:
        temp = input("Do you want to view what this strategy looks like? (Y/N) ")
        if(temp.lower() == 'y' or temp.lower() == 'yes'):
            try:
                display_energy = int(input(f"At what energy level would you like to view the strategy? (Current energy level is {INITIAL_ENERGY_LEVEL}) "))
            except:
                print(f"Invalid input. Please try again.")
            display(best_strategy, display_energy)
            break
        elif(temp.lower() == 'n' or temp.lower() == 'no'):
            break
        else:
            print(f"Invalid input. Please try again.")
        

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
PARENT_CHANCE = 0.2  # Adjust as needed
NUMBER_OF_DRONES = 3
INITIAL_ENERGY_LEVEL = 100
GRID_HEIGHT, GRID_WIDTH = 500, 500

# USED IN FITNESS FUNCTION
# Coverage rate = 1/altitude
# Drone coverage radius = altitude/sqrt(3)

# Format: [x1, y1, z1, θ1, v1, x2, y2, z2, θ2, v2, ...]
population = []
for i in range(POPULATION_SIZE):
    strategy_entity = []
    for j in range(NUMBER_OF_DRONES): # spherical coordinates
        strategy_entity.append(np.random.uniform(0, 5))  # ro
        strategy_entity.append(np.random.uniform(0, 2*math.pi))  # theta
        strategy_entity.append(np.random.uniform(0, math.pi/2))  # phi
        strategy_entity.append(np.random.uniform(0, 0.5))  # dro # strategy_entity.append(np.random.uniform(0, 0.1))  # dro
        strategy_entity.append(np.random.uniform(0, math.pi/8))  # dtheta
        strategy_entity.append(np.random.uniform(0, math.pi/32))  # dphi        # strategy_entity.append(np.random.uniform(0, 0.1)) # dphi
    # 0 - 5 for x, y, z, 0 - 2pi for angle, 0 - 2 for velocity
    population.append(tuple(strategy_entity))
    
genetic_algorithm(population, 0)