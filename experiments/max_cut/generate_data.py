import numpy as np
import random

# Parameters for dataset generation
n = 30               # number of nodes in each graph
num_train = 1000     # number of training samples to generate
num_test = 200       # number of test samples to generate
train_file = "experiments/max_cut/data/maxcut_train.csv"
test_file  = "experiments/max_cut/data/maxcut_test.csv"

def generate_maxcut_instance(n: int):
    """Generate one random Max-Cut instance with known optimal solution."""
    # 1) random partition (not all 0 or all 1)
    while True:
        solution = [random.randint(0, 1) for _ in range(n)]
        if any(v == 0 for v in solution) and any(v == 1 for v in solution):
            break
    solution = np.array(solution, dtype=int)

    # 2) build symmetric adjacency matrix
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            if solution[i] != solution[j]:
                w = random.uniform(0.1, 1.0)    # cut edge
            else:
                w = random.uniform(-1.0, -0.1)  # non-cut edge
            W[i, j] = W[j, i] = w
    np.fill_diagonal(W, 0.0)
    return W, solution

def make_dataset(num_samples, filename):
    rows = []
    for _ in range(num_samples):
        W, sol = generate_maxcut_instance(n)
        flat_adj = W.flatten()
        row = np.concatenate([flat_adj, sol])
        rows.append(row)
    arr = np.array(rows)
    np.savetxt(filename, arr, fmt="%.4f", delimiter=",")
    print(f"Saved {num_samples} samples to {filename}")

if __name__ == "__main__":
    make_dataset(num_train, train_file)
    make_dataset(num_test,  test_file)
