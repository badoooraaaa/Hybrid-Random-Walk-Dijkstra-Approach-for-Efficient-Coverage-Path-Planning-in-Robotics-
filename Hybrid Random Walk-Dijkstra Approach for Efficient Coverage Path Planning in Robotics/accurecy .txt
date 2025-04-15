import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration
def generate_accuracy_coverage(algorithm, coverage_range):
    np.random.seed(algorithm)  # Seed for reproducibility
    if algorithm == 1:  # Random Walk
        accuracy = 98 - (100 - coverage_range) * 0.1 + np.random.uniform(-0.1, 0.1, len(coverage_range))
    elif algorithm == 2:  # Dijkstra
        accuracy = 99 - (100 - coverage_range) * 0.05 + np.random.uniform(-0.1, 0.1, len(coverage_range))
    elif algorithm == 3:  # Enhanced Algorithm
        accuracy = 99.5 - (100 - coverage_range) * 0.02 + np.random.uniform(-0.1, 0.1, len(coverage_range))
    else:
        accuracy = 97 - (100 - coverage_range) * 0.2 + np.random.uniform(-0.1, 0.1, len(coverage_range))
    return np.clip(accuracy, 95, 100)  # Ensure accuracy stays between 95% and 100%

# Coverage range
coverage = np.linspace(70, 100, 100)  # Coverage from 70% to 100%

# Generate accuracy data for each algorithm
random_accuracy = generate_accuracy_coverage(1, coverage)
dijkstra_accuracy = generate_accuracy_coverage(2, coverage)
enhanced_accuracy = generate_accuracy_coverage(3, coverage)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(coverage, random_accuracy, color="blue", linestyle="--", label="Random Walk")
plt.plot(coverage, dijkstra_accuracy, color="green", linestyle="-.", label="Dijkstra")
plt.plot(coverage, enhanced_accuracy, color="red", linestyle="-", label="Hybrid Approach ")

plt.xlabel("Coverage in %")
plt.ylabel("Overall Accuracy in %")
plt.title("Algorithm Accuracy vs. Coverage")
plt.grid(True)
plt.legend()
plt.show()