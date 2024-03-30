from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



# Fitness Function

def fitness_function(strategy):
    # Initialize variables and drones
    energy = 1000
    coverage = 0
    
    d1 = [strategy[0], strategy[1], strategy[7]]
    d2 = [strategy[2], strategy[3], strategy[7]]
    d3 = [strategy[4], strategy[5], strategy[7]]
    
    drones = [d1, d2, d3]
    
    dv = s[6]
    z = strategy[7]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

    plt.ion()

    # Start simulation
    while energy > 0:
        # Move the drones
        for drone in drones:
            for i in range(2):
                drone[i] += dv
        
        # Update the energy
        energy -= dv
        print(f"Energy remaining: {energy}")
        
        # Update the coverage
        coverage += 1
        print(f"Coverage so far: {coverage}")

        # Update the plot
        update_plot(ax, drones)
        plt.pause(0.1)  # Pause for a short while to update plot
        
    print(f"Final coverage: {coverage}")

def coverage( drones ):
    # Calculate the coverage of the drones
    
    # First, generate 500 random points in the XY plane 5 x 5
    points = np.random.rand(500, 2) * 5
    
    # Then, for each drone, check if each point is within the drone's range
    # The drone's range is defined as a circle in the XY plane with radius r = drone position x altitude
    
    
    return 0

def update_plot(ax, drones):    
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    
    ax.clear()
    
    for drone in drones:
        ax.scatter(drone[0], drone[1], drone[2])
    
    
    plt.draw()
    plt.pause(0.1)

s = [1, 0, 2, 2, 4, 5, 0.66, 4]
# 3 pairs of coordinates, then a drone velocity, and then a drone altitude

fitness_function(s)
print("Done!")
