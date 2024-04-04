import numpy as np
import matplotlib.pyplot as plt
import math
import operator
from time import perf_counter
import pickle
from tqdm import tqdm
import cProfile

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
    grid_height, grid_width = coverage_grid.shape
    x_indices, y_indices = np.indices((grid_height, grid_width))
    
    # Scale indices according to the grid_scale to match physical space
    x_indices = x_indices * grid_scale / grid_width
    y_indices = y_indices * grid_scale / grid_height
    
    for drone in drones:
        drone_x, drone_y, drone_z = polar_to_rectangular(drone[:3])
        radius = drone_z / math.sqrt(3)  # Coverage radius in the same units as drone_x and drone_y
        
        # Calculate squared distances to avoid sqrt for performance
        distances_squared = (x_indices - drone_x) ** 2 + (y_indices - drone_y) ** 2
        radius_squared = radius ** 2
        
        # Create a mask where distances_squared is less than or equal to radius_squared
        coverage_mask = distances_squared <= radius_squared
        
        # Use the mask to update the coverage_grid efficiently
        # Adjust this update rule as per your specific logic for updating the grid
        # For example, adding 1 to covered cells or some function of drone_z
        coverage_grid[coverage_mask] += 1 / (coverage_grid[coverage_mask] * drone_z)
    
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
        if np.random.random() < MUTATION_RATE:
            drone[5] += np.random.uniform(-0.1, 0.1)    # deviate dphi
                
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
        
        ax.clear()
        ax.set_title(f"Energy: {energy}") 
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_zlim([0, 5])
        
        # move drones
        for i in range(len(drones)):
            drone = drones[i]

            # change position of spherical coordinate
            drone[0] += drone[3] # ro = ro + dro
            drone[1] += drone[4] # theta = theta + dtheta
            drone[2] += drone[5] # phi = phi + dphi
            
            ax.scatter( *polar_to_rectangular(drone[:3]), color='blue')  # Plot using only position information
            plt.draw()
            plt.pause(0.05)
                        
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
        ranked_strategy_list.append(list(strategy))
    
    best_strategy = ranked_strategy_list[0]
    best_strategy_score = fitness[tuple(ranked_strategy_list[0])]
    
    end = perf_counter()
    print(f"{end-start} seconds elapsed to calculate fitness.")

    display_population( sorted_dict, generation_count, generation_sum, best_strategy, best_strategy_score)
    
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
            
    # Check if user wants to save the current state of the population
    while True:
        temp = input("Do you want to save the current population? (Y/N) ")
        if(temp.lower() == 'y' or temp.lower() == 'yes'):
            save_population( fitness, generation_count )
            print("Population saved successfully.")
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
    # genetic_algorithm( children_generation, generation_count)
        
###################################################################################################
def save_population( population, generation_count ):
    with open('population.pkl', 'wb') as f:
        pickle.dump(population, f)
    with open('generation_count.pkl', 'wb') as f:
        pickle.dump(generation_count, f)
        
def load_population():
    with open('population.pkl', 'rb') as f:
        population = pickle.load(f)
    with open('generation_count.pkl', 'rb') as f:
        generation_count = pickle.load(f)
        
    # retrieve the best strategy and generation sum from the population data
    ranked_strategy_list = []
    generation_sum = 0
    for i, (strategy, score) in enumerate(population.items()):
        ranked_strategy_list.append(list(strategy))
        generation_sum += score
    
    best_strategy = ranked_strategy_list[0]
    best_strategy_score = population[tuple(ranked_strategy_list[0])]
    
    print("Population loaded successfully: ")
    display_population( population, generation_count, generation_sum, best_strategy, best_strategy_score)
    return population, generation_count, generation_sum, best_strategy, best_strategy_score

def display_population( population, generation_count, generation_sum, best_strategy, best_strategy_score):
    for i, (strategy, score) in enumerate(population.items()):
        print(f"Strategy {i+1}: {format(strategy)} with score {score}")
        
    print(f"-----------------------------")
    print(f"Generation {generation_count} average:  {generation_sum / len(population)}.")
    print(f"Best strategy: {format(best_strategy)} with score {best_strategy_score}")
    print(f"-----------------------------")
    

    
    
    
    
    
    

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

# Prompt user if they want to load a previous population or start a new genetic algorithm
while True:
    temp = input("Do you want to load a previous population? (Y/N) ")
    if(temp.lower() == 'y' or temp.lower() == 'yes'):
        loaded_population, generation_count = load_population()
        print("Starting genetic algorithm with loaded population. \n \n \n")
        genetic_algorithm(loaded_population, generation_count)
        break
    elif(temp.lower() == 'n' or temp.lower() == 'no'):
        initial_population = []
        for i in range(POPULATION_SIZE):
            strategy_entity = []
            for j in range(NUMBER_OF_DRONES): # spherical coordinates
                strategy_entity.append(np.random.uniform(0, 5))  # ro
                strategy_entity.append(np.random.uniform(0, 2*math.pi))  # theta
                strategy_entity.append(np.random.uniform(0, math.pi/2))  # phi
                # strategy_entity.append(np.random.uniform(0, 0.5))  # dro # strategy_entity.append(np.random.uniform(0, 0.1))  # dro
                strategy_entity.append(0)  # dro # strategy_entity.append(np.random.uniform(0, 0.1))  # dro
                strategy_entity.append(np.random.uniform(0, math.pi/8))  # dtheta
                strategy_entity.append(np.random.uniform(0, math.pi/32))  # dphi        # strategy_entity.append(np.random.uniform(0, 0.1)) # dphi
            # 0 - 5 for x, y, z, 0 - 2pi for angle, 0 - 2 for velocity
            initial_population.append(tuple(strategy_entity))
        cProfile.run('genetic_algorithm(initial_population, 0)')
        break
        