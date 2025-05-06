import random
import torch

def write_dataset(filename, Q_rows, target_seq):
    """
    Write the dataset to a text file.

    Format:
    - First line: num_samples n
    - Then for each sample:
        - n lines: each row of Q_rows (n floats space-separated)
        - 1 line: target sequence (n+1 ints space-separated)
    """
    num_samples, n, _ = Q_rows.shape
    with open(filename, 'w') as f:
        f.write(f"{num_samples} {n}\n")
        for i in range(num_samples):
            # Write Q matrix rows
            for row in Q_rows[i]:
                f.write(' '.join(map(str, row.tolist())) + '\n')
            # Write target sequence
            f.write(' '.join(map(str, target_seq[i].tolist())) + '\n')

def read_dataset(filename):
    """
    Read the dataset from a text file written by write_dataset.

    Returns:
        Q_rows: torch.FloatTensor of shape [num_samples, n, n]
        target_seq: torch.LongTensor of shape [num_samples, n+1]
    """
    with open(filename, 'r') as f:
        first = f.readline().strip().split()
        num_samples, n = map(int, first)

        Q_rows = torch.zeros(num_samples, n, n, dtype=torch.float)
        target_seq = torch.zeros(num_samples, n+1, dtype=torch.long)

        for i in range(num_samples):
            # Read Q matrix
            for j in range(n):
                row_vals = list(map(float, f.readline().strip().split()))
                Q_rows[i, j] = torch.tensor(row_vals)
            # Read target sequence
            seq_vals = list(map(int, f.readline().strip().split()))
            target_seq[i] = torch.tensor(seq_vals)

    return Q_rows, target_seq


def write_ground_truth(filename, Q_rows, is_A_list):
    """
    Write the ground truth adjacency matrices to a text file.

    Format:
    - First line: num_samples n
    - Then for each sample:
        - n lines: each row of the ground truth adjacency matrix (n floats space-separated)
    """
    num_samples, n, _ = Q_rows.shape
    with open(filename, 'w') as f:
        f.write(f"{num_samples} {n}\n")
        for i in range(num_samples):
            is_A = is_A_list[i]
            ground_truth_Q = torch.zeros(n, n, dtype=torch.float)
            for u in range(n):
                for v in range(n):
                    if u != v and ((is_A[u] and is_A[v]) or (~is_A[u] and ~is_A[v])):
                        ground_truth_Q[u, v] = 1.0
            for row in ground_truth_Q:
                f.write(' '.join(map(str, row.tolist())) + '\n')


# Modify generate_sbm_dataset to return is_A_list
def generate_sbm_dataset(num_samples: int,
                         n: int,
                         x_a: float,
                         p_inter: float,
                         y_b: float) -> tuple[torch.FloatTensor, torch.LongTensor, list]:
    """
    Generate a dataset of stochastic block model graphs and ordered cut sequences.

    Returns:
        Q_rows_tensor: FloatTensor [num_samples, n, n], adjacency matrices.
        target_seq_tensor: LongTensor [num_samples, n+1], ordered node indices with EOS.
        is_A_list: List of boolean tensors indicating group membership for each graph.
    """
    EOS_index = n
    Q_rows_tensor = torch.zeros(num_samples, n, n, dtype=torch.float)
    target_seq_tensor = torch.zeros(num_samples, n+1, dtype=torch.long)
    is_A_list = []

    for i in range(num_samples):
        # Randomly determine the size of group A
        permuted_indices = torch.randperm(n)
        size_A = int(random.gauss(25, 3))
        is_A = torch.zeros(n, dtype=torch.bool)
        is_A[permuted_indices[:size_A]] = True
        is_A_list.append(is_A)

        # Sample undirected adjacency matrix
        Q = torch.zeros(n, n, dtype=torch.float)
        for u in range(n):
            for v in range(u + 1, n):
                if is_A[u] and is_A[v]:
                    p = x_a
                elif not is_A[u] and not is_A[v]:
                    p = y_b
                else:
                    p = p_inter
                if torch.rand(()) < p:
                    Q[u, v] = 1.0
                    Q[v, u] = 1.0

        Q_rows_tensor[i] = Q

        # Build ordered target sequence
        A_nodes = torch.nonzero(is_A, as_tuple=True)[0]
        B_nodes = torch.nonzero(~is_A, as_tuple=True)[0]

        seq = torch.full((n + 1,), EOS_index, dtype=torch.long)
        seq[: len(A_nodes)] = A_nodes
        seq[len(A_nodes)] = EOS_index
        seq[len(A_nodes) + 1 :] = B_nodes

        target_seq_tensor[i] = seq

    return Q_rows_tensor, target_seq_tensor, is_A_list


# Generate dataset and ground truth
Q_rows_tensor, target_seq_tensor, is_A_list = generate_sbm_dataset(
    # num_samples=500,
    num_samples=5,
    n=50,
    x_a=0.8,
    p_inter=0.2,
    y_b=0.75
)

# Write dataset
dataset_filename = 'experiments/max_cut/dataset.txt'
write_dataset(dataset_filename, Q_rows_tensor, target_seq_tensor)

# Write ground truth
ground_truth_filename = 'experiments/max_cut/ground_truth.txt'
write_ground_truth(ground_truth_filename, Q_rows_tensor, is_A_list)