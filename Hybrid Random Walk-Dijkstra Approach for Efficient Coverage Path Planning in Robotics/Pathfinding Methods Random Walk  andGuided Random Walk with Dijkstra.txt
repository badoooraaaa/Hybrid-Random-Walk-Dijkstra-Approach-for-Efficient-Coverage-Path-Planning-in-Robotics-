import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# Parameters
grid_size = 20  # 20x20 grid
n_steps = 300  # Number of steps for each walk
probability = 0.7  # Probability to choose the shortest path

# Create a 2D grid graph
graph = nx.grid_2d_graph(grid_size, grid_size)
start = (0, 0)  # Starting position
goal = (grid_size - 1, grid_size - 1)  # Goal position


# Function for Dijkstra-Guided Walk
def dijkstra_guided_walk(graph, start, goal, n_steps, probability):
    path = [start]
    current_node = start
    # Precompute shortest paths from each node to the goal
    shortest_paths = nx.single_source_dijkstra_path(graph, goal)

    for _ in range(n_steps):
        neighbors = list(graph.neighbors(current_node))
        # Determine the next move based on Dijkstra guidance with probability
        if random.random() < probability:
            # Move towards the shortest path node if possible
            next_node = min(neighbors, key=lambda n: len(shortest_paths.get(n, [])))
        else:
            # Random move
            next_node = random.choice(neighbors)

        path.append(next_node)
        current_node = next_node
        if current_node == goal:
            break  # Stop if goal is reached

    return path


# Function for Random Walk
def random_walk(graph, start, n_steps):
    path = [start]
    current_node = start
    for _ in range(n_steps):
        neighbors = list(graph.neighbors(current_node))
        next_node = random.choice(neighbors)
        path.append(next_node)
        current_node = next_node
    return path


# Function to calculate touring cost and coverage loss
def calculate_metrics(path, goal, n_steps):
    # Touring cost: the length of the path or the step limit if goal not reached
    touring_cost = len(path)
    # Coverage loss: 0 if goal was reached, 1 otherwise (for simplicity)
    coverage_loss = 0 if path[-1] == goal else 1
    return touring_cost, coverage_loss


# Generate paths
random_walk_path = random_walk(graph, start, n_steps)
dijkstra_walk_path = dijkstra_guided_walk(graph, start, goal, n_steps, probability)

# Calculate metrics
random_walk_cost, random_walk_loss = calculate_metrics(random_walk_path, goal, n_steps)
dijkstra_walk_cost, dijkstra_walk_loss = calculate_metrics(dijkstra_walk_path, goal, n_steps)

# Print results
print("Random Walk Metrics:")
print(f"Total Touring Cost: {random_walk_cost}")
print(f"Coverage Loss: {random_walk_loss}")

print("\nDijkstra-Guided Walk Metrics:")
print(f"Total Touring Cost: {dijkstra_walk_cost}")
print(f"Coverage Loss: {dijkstra_walk_loss}")

# Extract x and y coordinates for plotting
random_walk_x, random_walk_y = zip(*random_walk_path)
dijkstra_walk_x, dijkstra_walk_y = zip(*dijkstra_walk_path)

# Plotting
plt.figure(figsize=(10, 10))

# Plot the random walk path
plt.subplot(1, 2, 1)
plt.plot(random_walk_x, random_walk_y, color="green", label="Random Walk (Unassisted)")
plt.scatter(*goal, color="red", label="Goal")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.title("Random Walk Only")

# Plot the guided walk path with Dijkstra
plt.subplot(1, 2, 2)
plt.plot(dijkstra_walk_x, dijkstra_walk_y, color="blue", label="Hybrid Approach ")
plt.scatter(*goal, color="red", label="Goal")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.title("Guided Random Walk with Dijkstra")

# Show plots
plt.tight_layout()
plt.show()