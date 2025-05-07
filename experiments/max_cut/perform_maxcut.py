import math
import numpy as np
import torch
import torch.nn.functional as F
from PointerNet import PointerNetwork

#── CONFIG ───────────────────────────────────────────────────────────────────────
train_file = "experiments/max_cut/data/maxcut_train.csv"
test_file  = "experiments/max_cut/data/maxcut_test.csv"
embedding_dim = 128
hidden_dim    = 256
batch_size    = 16
num_epochs    = 30
lr            = 0.1
#───────────────────────────────────────────────────────────────────────────────────

def load_dataset(filename):
    data = np.loadtxt(filename, delimiter=",")
    num_samples, total_dim = data.shape
    # deduce n from n*n + n = total_dim
    n = int((-1 + math.sqrt(1 + 4 * total_dim)) / 2)
    assert n*n + n == total_dim, "Bad format"
    X = data[:, :n*n].reshape(num_samples, n, n).astype(np.float32)
    Y = data[:, n*n:].astype(int)
    return X, Y, n

def build_target_sequences(Y, n):
    eos = n
    sequences = []
    for sol in Y:
        set1 = sorted([i for i,v in enumerate(sol) if v==1])
        set0 = sorted([i for i,v in enumerate(sol) if v==0])
        seq = set1 + [eos] + set0
        sequences.append(seq)
    return sequences

# ── Load train & test ─────────────────────────────────────────────────────────────
X_train, Y_train, n_train = load_dataset(train_file)
X_test,  Y_test,  n_test  = load_dataset(test_file)
assert n_train == n_test, "Train/test node count mismatch"
n = n_train

train_seqs = build_target_sequences(Y_train, n)
test_seqs  = build_target_sequences(Y_test,  n)

# ── Tensors & Model ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train, device=device)  # (N_train, n, n)
X_test_t  = torch.tensor(X_test,  device=device)  # (N_test,  n, n)
model     = PointerNetwork(input_dim=n,
                           embedding_dim=embedding_dim,
                           hidden_dim=hidden_dim).to(device)
opt       = torch.optim.SGD(model.parameters(), lr=lr)

# ── Training Loop ────────────────────────────────────────────────────────────────
model.train()
N_train = X_train_t.size(0)
for epoch in range(1, num_epochs+1):
    perm = torch.randperm(N_train, device=device)
    total_loss = 0.0
    for i in range(0, N_train, batch_size):
        idx = perm[i:i+batch_size]
        batch_X = X_train_t[idx]
        batch_targets = [train_seqs[j] for j in idx.cpu().tolist()]
        opt.zero_grad()
        loss = model(batch_X, target_seq=batch_targets)
        loss.backward()
        opt.step()
        total_loss += loss.item() * idx.size(0)
    avg_loss = total_loss / N_train
    if epoch==1 or epoch%10==0:
        print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")

# ── Evaluation on Test Set ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    N_test = X_test_t.size(0)
    correct = 0
    for i in range(0, N_test, batch_size):
        batch_X = X_test_t[i:i+batch_size]
        outputs = model(batch_X)  # list of sequences, length=batch_size
        for j, out_seq in enumerate(outputs):
            # split at EOS
            eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
            set1 = set(out_seq[:eos_pos])
            # reconstruct predicted solution vector
            pred = [1 if idx in set1 else 0 for idx in range(n)]
            # compare to ground truth
            if pred == Y_test[i+j].tolist():
                correct += 1
    acc = correct / N_test * 100
    print(f"\nTest set accuracy: {correct}/{N_test} = {acc:.2f}%")
