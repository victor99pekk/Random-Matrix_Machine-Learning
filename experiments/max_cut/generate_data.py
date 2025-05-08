import numpy as np
import random

# Parameters for dataset generation
n = 50               # total number of nodes (must be even)
num_train = 50000     # number of training samples to generate
num_test = 200       # number of test samples to generate
train_file = "experiments/max_cut/data/maxcut_train.csv"
test_file  = "experiments/max_cut/data/maxcut_test.csv"

def generate_maxcut_instance(n: int):
    """Generate one random Max-Cut instance with a planted half/half partition.
    
    - Exactly n/2 nodes in group 1, n/2 in group 0.
    - Sample P1,P0 ~ Uniform(0.8,1.0), P ~ Uniform(0,1.0).
    - For i<j:
        if both in group 1: connect with prob P1
        if both in group 0: connect with prob P0
        if across groups:     connect with prob P
    Returns:
      W:   n×n adjacency matrix of 0/1 edges
      sol: length-n 0/1 vector (the planted partition)
    """
    assert n % 2 == 0, "n must be even"
    # 1) fixed-size partition: exactly n/2 ones
    sol = np.array([1]*(n//2) + [0]*(n//2), dtype=int)
    np.random.shuffle(sol)
    
    # 2) sample the three probabilities for this graph
    P1 = random.uniform(0.9, 1.0)   # within-group-1
    P0 = random.uniform(0.9, 1.0)   # within-group-0
    P  = random.uniform(0.0, 0.1)   # across-groups

    # 3) build symmetric adjacency (0/1) by Bernoulli trials
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if sol[i] == sol[j] == 1:
                prob = P1
            elif sol[i] == sol[j] == 0:
                prob = P0
            else:
                prob = P
            if random.random() < prob:
                W[i, j] = W[j, i] = 1
    return W, sol

def make_dataset(num_samples, filename):
    rows = []
    for _ in range(num_samples):
        W, sol = generate_maxcut_instance(n)
        flat_adj = W.flatten()
        row = np.concatenate([flat_adj, sol])
        rows.append(row)
    arr = np.array(rows)
    np.savetxt(filename, arr, fmt="%d", delimiter=",")
    print(f"Saved {num_samples} samples to {filename}")

if __name__ == "__main__":
    make_dataset(num_train, train_file)
    make_dataset(num_test,  test_file)
