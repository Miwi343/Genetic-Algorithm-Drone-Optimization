def update_coverage(drones, coverage_grid, decay_factor=0.9):
    grid_size = coverage_grid.shape  # Assuming coverage_grid is a NumPy array
    for drone in drones:
        drone_x, drone_y, drone_z = polar_to_rectangular(drone[:3])
        radius = drone[2] / math.sqrt(3)  # Coverage radius
        
        # Convert drone coordinates to grid indices
        drone_grid_x = int((drone_x / AREA_WIDTH) * grid_size[0])
        drone_grid_y = int((drone_y / AREA_HEIGHT) * grid_size[1])
        
        # Update coverage for cells within the drone's coverage area
        for x in range(max(0, drone_grid_x - int(radius)), min(grid_size[0], drone_grid_x + int(radius) + 1)):
            for y in range(max(0, drone_grid_y - int(radius)), min(grid_size[1], drone_grid_y + int(radius) + 1)):
                # Check if the cell is within the drone's coverage radius
                if np.sqrt((x - drone_grid_x)**2 + (y - drone_grid_y)**2) <= radius:
                    coverage_grid[x, y] = max(coverage_grid[x, y] * decay_factor, 1)




def fitness_function(strategy):
    drones = [strategy[i*6:i*6+6] for i in range(NUMBER_OF_DRONES)]
    coverage_grid = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # Initialize coverage grid
    decay_factor = 0.9  # Adjust as needed
    
    for trial in range(5):
        energy = INITIAL_ENERGY_LEVEL
        trial_fitness = 0
        
        while energy > 0:
            # Assume movement code for drones is here
            
            # Update coverage based on drones' positions
            update_coverage(drones, coverage_grid, decay_factor)
            
            # Calculate trial fitness as the sum of covered cells
            # Adjust this calculation as necessary for your specific needs
            trial_fitness += np.sum(coverage_grid)
            
            # Update energy based on drone movement
            # Assume energy update code is here
        
        average_fitness += trial_fitness / 5  # Average fitness over trials
    
    return average_fitness / 5  # Average fitness over all iterations
