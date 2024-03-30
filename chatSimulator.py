#*******Randomly generated movements**********

# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_coverage(drones, coverage_map):
#     """
#     Update the coverage_map based on drones' positions and predefined coverage radius.
#     Assumes drones have a circular coverage area on the XY plane depending on their altitude.
#     """
#     coverage_radius = 0.5  # Define the coverage radius, adjust based on your needs
#     for drone in drones:
#         x, y, z = drone
#         # Convert drone x, y to coverage map indices
#         ix, iy = int(x * 100), int(y * 100)  # Assuming coverage_map is 500x500 for 5x5 area
#         # Update coverage for simplicity, consider coverage radius and altitude in a more complex model
#         coverage_map[ix:ix+2, iy:iy+2] = 1  # Simple coverage update, refine to account for actual coverage area

# def fitness_function(strategy):
#     energy = 1000
#     coverage_map = np.zeros((500, 500))  # Example resolution for a 5x5 area

#     drones = [np.array(strategy[i:i+3]) for i in range(0, len(strategy) - 2, 3)]
#     dv = strategy[-2]  # Drone velocity

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     plt.ion()
#     ax.set_xlim([0, 5])
#     ax.set_ylim([0, 5])
#     ax.set_zlim([0, 5])

#     while energy > 0:
#         for i, drone in enumerate(drones):
#             # Random movement within bounds
#             move_direction = np.random.rand(3) * 2 - 1  # Random direction
#             move_direction[2] = 0  # Ensure no vertical movement
#             new_pos = drone + move_direction * dv
#             # Keep within bounds
#             new_pos = np.clip(new_pos, 0, 5)
#             drones[i] = new_pos
#             energy -= dv  # Update energy

#         calculate_coverage(drones, coverage_map)
        
#         print( f"Energy remaining: {energy}")
#         print( f"Coverage so far: {np.sum(coverage_map)}")

#         update_plot(ax, drones)
#         plt.pause(0.1)

#     plt.ioff()
#     plt.show()

#     coverage = np.sum(coverage_map) / (500*500) * 25  # Convert to area covered in the 5x5 grid
#     print(f"Final coverage: {coverage} square units, Energy remaining: {energy}")

# def update_plot(ax, drones):
#     ax.clear()
#     ax.set_xlim([0, 5])
#     ax.set_ylim([0, 5])
#     ax.set_zlim([0, 5])
#     for drone in drones:
#         ax.scatter(*drone, color='blue')
#     plt.draw()

# # Example strategy with drones' starting positions, velocity, and altitude
# strategy = [1, 1, 2, 2, 2, 3, 0.5, 2]  # Example strategy vector

# fitness_function(strategy)


#********Movements in circle*************

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

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

def update_plot(ax, drones):
    ax.clear()
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    for drone in drones:
        ax.scatter(*drone[:3], color='blue')  # Plot using only position information
    plt.draw()

def fitness_function(strategy):
    energy = 100
    coverage_map = np.zeros((500, 500))  # Example resolution for a 5x5 area

    # Extract drones' information from strategy
    drones = [np.array(strategy[i:i+5]) for i in range(0, len(strategy), 5)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

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
        
        print(f"Energy remaining: {energy}")
        print(f"Coverage so far: {np.sum(coverage_map)}")

        update_plot(ax, drones)
        plt.pause(0.1)

    plt.ioff()
    plt.show()

    coverage = np.sum(coverage_map) / (500*500) * 25  # Convert to area covered in the 5x5 grid
    print(f"Final coverage: {coverage} square units, Energy remaining: {energy}")

# Example strategy with drones' starting positions, angles, speeds, and altitudes
# Format: [x1, y1, z1, θ1, v1, x2, y2, z2, θ2, v2, ...]
strategy = [1, 1, 2, math.pi/4, 0.1, 2, 2, 3, math.pi/2, 0.1]  # Example strategy vector

fitness_function(strategy)
