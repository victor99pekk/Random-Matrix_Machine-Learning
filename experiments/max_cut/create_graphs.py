import numpy as np
import random

# Parameters for dataset generation
n = 10            # number of nodes in each graph
num_samples = 10**2 # number of training samples to generate
output_file = "maxcut_dataset.csv"

def generate_maxcut_instance(n: int):
    """Generate one random Max-Cut instance with known optimal solution.
    Returns: (adjacency_matrix, solution_vector)
        adjacency_matrix: n x n symmetric matrix of edge weights.
        solution_vector: length-n binary array (0/1) indicating the partition of each node.
    """
    # Random binary assignment for n nodes (ensure not all 0 or all 1)
    while True:
        solution = [random.randint(0, 1) for _ in range(n)]
        if any(v == 0 for v in solution) and any(v == 1 for v in solution):
            break
    solution = np.array(solution, dtype=int)
    # Initialize adjacency matrix
    W = np.zeros((n, n), dtype=float)
    # Assign weights: positive for edges between partitions, negative for edges within a partition
    for i in range(n):
        for j in range(i+1, n):
            if solution[i] != solution[j]:
                # Edge between different partitions (cut edge) – assign a positive weight
                w = random.uniform(0.1, 1.0)
            else:
                # Edge within the same partition – assign a negative weight
                w = random.uniform(-1.0, -0.1)
            W[i, j] = W[j, i] = w
    # No self-loops; set diagonal to 0
    np.fill_diagonal(W, 0.0)
    return W, solution

# Generate multiple samples and save to CSV
data_rows = []
for _ in range(num_samples):
    adj_matrix, solution = generate_maxcut_instance(n)
    # Flatten the adjacency matrix and append the solution vector
    flat_adj = adj_matrix.flatten()
    row = np.concatenate([flat_adj, solution])
    data_rows.append(row)

data_array = np.array(data_rows)
# Save as CSV: each row begins with n*n adjacency entries, followed by n solution entries
np.savetxt(output_file, data_array, fmt="%.4f", delimiter=",")
print(f"Dataset saved to {output_file} (format: {n*n} adjacency values + {n} solution values per line).")
