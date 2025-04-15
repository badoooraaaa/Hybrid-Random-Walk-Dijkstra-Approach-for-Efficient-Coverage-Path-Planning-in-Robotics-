import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from scipy.signal import savgol_filter  # For smoothing the data

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
    shortest_paths = nx.single_source_dijkstra_path(graph, goal)

    for _ in range(n_steps):
        neighbors = list(graph.neighbors(current_node))
        if random.random() < probability:
            next_node = min(neighbors, key=lambda n: len(shortest_paths.get(n, [])))
        else:
            next_node = random.choice(neighbors)

        path.append(next_node)
        current_node = next_node
        if current_node == goal:
            break

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
    touring_cost = len(path)
    coverage_loss = 0 if path[-1] == goal else 1
    return touring_cost, coverage_loss


# Function to calculate exploration speed
def calculate_exploration_speed(path, time_taken):
    unique_nodes_explored = len(set(path))
    if time_taken < 1e-6:
        exploration_speed = float('inf')
    else:
        exploration_speed = unique_nodes_explored / time_taken
    return unique_nodes_explored, exploration_speed


# Generate and time the random walk
start_time = time.time()
random_walk_path = random_walk(graph, start, n_steps)
random_walk_time = time.time() - start_time

# Generate and time the Dijkstra-guided walk
start_time = time.time()
dijkstra_walk_path = dijkstra_guided_walk(graph, start, goal, n_steps, probability)
dijkstra_walk_time = time.time() - start_time

# Calculate metrics
random_walk_cost, random_walk_loss = calculate_metrics(random_walk_path, goal, n_steps)
dijkstra_walk_cost, dijkstra_walk_loss = calculate_metrics(dijkstra_walk_path, goal, n_steps)

# Calculate exploration speed
random_walk_unique, random_walk_speed = calculate_exploration_speed(random_walk_path, random_walk_time)
dijkstra_walk_unique, dijkstra_walk_speed = calculate_exploration_speed(dijkstra_walk_path, dijkstra_walk_time)

# Create data for graphing exploration metrics
steps = list(range(1, n_steps + 1))
random_walk_speed_per_step = [random_walk_unique / (random_walk_time + i) for i in steps]
dijkstra_walk_speed_per_step = [dijkstra_walk_unique / (dijkstra_walk_time + i) for i in steps]

# Smooth the data for cleaner curves
random_walk_speed_smoothed = savgol_filter(random_walk_speed_per_step, 51, 3)
dijkstra_walk_speed_smoothed = savgol_filter(dijkstra_walk_speed_per_step, 51, 3)

# Plotting exploration graph with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left y-axis (Exploration Speed)
ax1.plot(steps, random_walk_speed_smoothed, 'g--', label="Random Walk Speed", linewidth=2)
ax1.plot(steps, dijkstra_walk_speed_smoothed, 'b-', label="Hybrid Approach", linewidth=2)
ax1.set_xlabel('Steps')
ax1.set_ylabel('Exploration Speed (nodes/second)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')

# Add title and grid
plt.title("Exploration Speed Over Steps", fontsize=14)
ax1.grid(True)

# Add annotations (Example)
ax1.annotate("Highest Speed", xy=(150, max(random_walk_speed_smoothed)),
             xytext=(180, max(random_walk_speed_smoothed) + 0.5),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=12, color='black')

# Show plot
plt.tight_layout()
plt.show()