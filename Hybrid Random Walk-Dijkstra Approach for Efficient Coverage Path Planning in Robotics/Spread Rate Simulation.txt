import numpy as np
import matplotlib.pyplot as plt

# Random Walk Function (Unassisted)
def random_walk(steps):
    position = 0
    walk = [position]
    for _ in range(steps):
        step = np.random.choice([-1, 1])
        position += step
        walk.append(position)
    return walk

# Pure Dijkstra's Walk (Direct Movement)
def dijkstra_walk(steps, target):
    position = 0
    walk = [position]
    while len(walk) < steps + 1:
        if position < target:
            position += 1  # Move directly towards the target
        elif position > target:
            position -= 1
        walk.append(position)
    return walk

# Enhanced Algorithm: Random Walk + Dijkstra (RDijkstra) for Maximum Position Reach
def enhanced_random_dijkstra_walk(steps, target):
    position = 0
    walk = [position]
    for step_num in range(steps):
        distance_to_target = abs(target - position)
        if distance_to_target < steps / 3:
            directed_prob = 0.85  # High probability of moving towards the target
        elif distance_to_target < (2 * steps / 3):
            directed_prob = 0.65  # Medium probability
        else:
            directed_prob = 0.5  # Early steps have equal chance for exploration

        if np.random.rand() < directed_prob:
            step = 1 if position < target else -1
        else:
            step = np.random.choice([-1, 1])

        position += step
        walk.append(position)

        if position in walk[:-1]:
            position += np.random.choice([1, 2])
            walk[-1] = position

    return walk

# Parameters
steps = 100
target_position = 50

# Generate walks
unassisted_walk = random_walk(steps)
enhanced_walk = enhanced_random_dijkstra_walk(steps, target_position)
dijkstra_walk_only = dijkstra_walk(steps, target_position)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Random Walk Plot
axs[0].plot(range(steps + 1), unassisted_walk, color="green", label="Random Walk (Unassisted)")
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Position")
axs[0].set_title("Random Walk (Unassisted)")
axs[0].legend()

# Enhanced Random + Dijkstra Walk Plot
axs[1].plot(range(steps + 1), enhanced_walk, color="purple", label="Enhanced Random + Dijkstra Walk")
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("Position")
axs[1].set_title("Enhanced Random + Dijkstra Walk")
axs[1].legend()

# Dijkstra's Walk Plot
axs[2].plot(range(steps + 1), dijkstra_walk_only, color="red", label="Dijkstra's Walk (Direct)")
axs[2].set_xlabel("Steps")
axs[2].set_ylabel("Position")
axs[2].set_title("Dijkstra's Walk (Direct)")
axs[2].legend()

plt.tight_layout()
plt.show()