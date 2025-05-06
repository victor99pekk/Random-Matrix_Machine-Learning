import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from PointerNet import PointerNet, train_model

# --------------------------------------------------------------------------- #
# 1) Paths to your saved files                                                #
# --------------------------------------------------------------------------- #
dataset_path       = "experiments/max_cut/dataset.txt"        # ← same file your write_dataset produced
ground_truth_path  = "experiments/max_cut/ground_truth.txt"   # (optional) only needed for evaluation

# --------------------------------------------------------------------------- #
# 2) Load the dataset you previously wrote                                   #
# --------------------------------------------------------------------------- #
from create_graphs import read_dataset          # or the module where read_dataset lives

Q_rows_tensor, target_seq_tensor = read_dataset(dataset_path)   # shapes: [S,n,n], [S,n+1]
num_samples, n, _ = Q_rows_tensor.shape



# --------------------------------------------------------------------------- #
# 3) Build DataLoader                                                         #
# --------------------------------------------------------------------------- #
batch_size = 32
dataset     = TensorDataset(Q_rows_tensor, target_seq_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------------------------------------------------------------- #
# 4) Instantiate model and train                                              #
# --------------------------------------------------------------------------- #
model = PointerNet(input_dim=n, hidden_dim=256)

train_model(
    model,
    train_loader,
    n,                  # so logits.view knows the vocabulary size
    max_epochs=1_000
)

# --------------------------------------------------------------------------- #
# 5) (Optional) If you want ground-truth adjacencies for evaluation           #
# --------------------------------------------------------------------------- #
# from max_cut_dataset import read_dataset   # reader already imported
# _, gt_target_seq = read_dataset(dataset_path)           # pointer seqs already in memory
# with open(ground_truth_path, "r") as f:                  # quick sanity-load of GT adjacencies
#     f.readline()  # skip header
#     ground_truth_adj = []
#     for s in range(num_samples):
#         ground_truth_adj.append(
#             torch.tensor([list(map(float, f.readline().split())) for _ in range(n)])
#         )
# ground_truth_adj = torch.stack(ground_truth_adj)        # [S,n,n]
