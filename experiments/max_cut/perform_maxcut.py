import math
import numpy as np
import torch
import torch.nn as nn
from PointerNet import PointerNetwork

# Load the dataset from CSV
data = np.loadtxt("maxcut_dataset.csv", delimiter=",")
num_samples, total_dim = data.shape
# Deduce number of nodes n from total_dim = n*n + n  (solve n^2 + n - total_dim = 0)
n = int((-1 + math.sqrt(1 + 4 * total_dim)) / 2)
assert n * n + n == total_dim, "Invalid dataset format: cannot deduce n."

# Split data into adjacency and solution
X = data[:, :n*n].reshape(num_samples, n, n).astype(np.float32)
Y = data[:, n*n:].astype(int)

# Prepare target sequences for training (list of node index sequences including EOS)
target_sequences = []
eos_index = n  # use index n as the EOS token
for sol in Y:
    # Nodes with value 1 go in one partition, value 0 in the other
    set1_indices = [i for i, val in enumerate(sol) if val == 1]
    set0_indices = [i for i, val in enumerate(sol) if val == 0]
    set1_indices.sort()
    set0_indices.sort()
    # Output sequence: all nodes in set1, then EOS, then all nodes in set0
    seq = set1_indices + [eos_index] + set0_indices
    target_sequences.append(seq)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X)            # shape: (num_samples, n, n)
# (We'll feed target_sequences directly as Python lists to the model's forward in this example)

# Initialize the Pointer Network model
model = PointerNetwork(input_dim=n, embedding_dim=128, hidden_dim=256)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# (Optionally, use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_tensor = X_tensor.to(device)

# Training loop
model.train()
batch_size = 16
num_epochs = 30
for epoch in range(1, num_epochs+1):
    # Shuffle the training data indices for each epoch
    indices = np.random.permutation(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_X = X_tensor[batch_idx]               # (batch_size, n, n)
        batch_targets = [target_sequences[j] for j in batch_idx]  # list of sequences
        optimizer.zero_grad()
        loss = model(batch_X, target_seq=batch_targets)  # compute cross-entropy loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_idx)
    avg_loss = epoch_loss / num_samples
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Example: use the trained model to predict the Max-Cut for a new graph (or a training sample)
model.eval()
with torch.no_grad():
    sample_index = 0  # using the first training sample as an example
    sample_adj = X_tensor[sample_index:sample_index+1]  # shape: (1, n, n)
    output_seq = model(sample_adj)[0]  # predicted sequence of indices (including eos index)
    # Interpret the output sequence: split into two sets at the EOS position
    if eos_index in output_seq:
        eos_pos = output_seq.index(eos_index)
    else:
        eos_pos = len(output_seq)
    pred_set1 = output_seq[:eos_pos]
    pred_set0 = output_seq[eos_pos+1:]
    pred_solution = [1 if i in pred_set1 else 0 for i in range(n)]
    print("\nExample prediction for sample index 0:")
    print("Predicted partition vector:", pred_solution)
    print("True partition vector:     ", Y[sample_index].tolist())
