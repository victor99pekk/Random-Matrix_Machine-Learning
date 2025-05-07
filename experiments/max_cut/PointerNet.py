import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# -----------------------------
# Data Generator for Max-Cut
# -----------------------------

def generate_maxcut_data(num_samples: int, num_nodes: int):
    """Return a list of tuples (Q, label_seq).
    * Q:   [num_nodes, num_nodes] symmetric weight matrix
    * label_seq: list of indices to select (value-1 nodes) **followed by EOS (= num_nodes)**
    """
    data = []
    for _ in range(num_samples):
        # Random symmetric matrix with weights 0-9
        Q = torch.randint(0, 10, (num_nodes, num_nodes)).float()
        Q = (Q + Q.t()) / 2  # make symmetric
        x = torch.randint(0, 2, (num_nodes,))  # random 0/1 assignment
        label = (x == 1).nonzero(as_tuple=True)[0].tolist()
        label.append(num_nodes)  # EOS token is num_nodes
        data.append((Q, label))
    return data

# -----------------------------
# Pointer Network (encoder-decoder with attention)
# -----------------------------

class PointerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.decoder_start = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, Q_rows: torch.Tensor, target_seq=None, teacher_forcing: bool = True):
        """If target_seq is provided and teacher_forcing=True, use teacher forcing.
        target_seq must be a Python list of length B, each an index list (int).
        Returns logits of shape [B, T, n+1] where T = dynamic steps.
        """
        B, n, _ = Q_rows.size()
        enc_out, (h_enc, c_enc) = self.encoder(Q_rows)

        # Append EOS representation (zeros) to encoder outputs
        eos_vec = torch.zeros(B, 1, enc_out.size(-1), device=Q_rows.device)
        enc_ext = torch.cat([enc_out, eos_vec], dim=1)  # [B, n+1, hidden]

        # Initial decoder state
        dec_inp = self.decoder_start.unsqueeze(0).expand(B, -1)  # [B, hidden]
        h_dec, c_dec = h_enc.squeeze(0), c_enc.squeeze(0)

        logits_seq = []
        mask = torch.zeros(B, n + 1, device=Q_rows.device)  # tracks used pointers

        # Determine how many decoding steps to run
        if teacher_forcing and target_seq is not None:
            max_steps = max(len(seq) for seq in target_seq)
        else:
            max_steps = n + 1  # worst-case during inference

        for step in range(max_steps):
            h_dec, c_dec = self.decoder_cell(dec_inp, (h_dec, c_dec))

            e1 = self.W1(enc_ext)            # [B, n+1, hidden]
            e2 = self.W2(h_dec).unsqueeze(1) # [B, 1,  hidden]
            scores = self.v(torch.tanh(e1 + e2)).squeeze(-1)  # [B, n+1]
            scores = scores.masked_fill(mask.bool(), -1e9)    # mask selected
            logits_seq.append(scores)

            # choose next pointer index
            if teacher_forcing and target_seq is not None:
                tgt_idx = []
                for b in range(B):
                    if step < len(target_seq[b]):
                        tgt_idx.append(target_seq[b][step])
                    else:
                        tgt_idx.append(n)  # EOS when out of labels
                idx = torch.tensor(tgt_idx, device=Q_rows.device)
            else:
                idx = scores.argmax(dim=1)

            # mark as selected
            mask[torch.arange(B), idx] = 1
            # next decoder input is the corresponding encoder output
            dec_inp = enc_ext[torch.arange(B), idx]

        logits = torch.stack(logits_seq, dim=1)  # [B, T, n+1]
        return logits

# -----------------------------
# Training routine
# -----------------------------

def train_pointer_network():
    num_nodes = 5          # size of tiny synthetic graph
    num_samples = 1000
    batch_size = 32
    hidden_dim = 128
    num_epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointerNetwork(input_dim=num_nodes, hidden_dim=hidden_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    dataset = generate_maxcut_data(num_samples, num_nodes)

    for epoch in range(1, num_epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            Q_batch = torch.stack([item[0] for item in batch]).to(device)
            y_batch = [item[1] for item in batch]  # python list of lists

            optimiser.zero_grad()
            logits = model(Q_batch, y_batch, teacher_forcing=True)  # [B, T, n+1]

            B, T, C = logits.shape
            # Build padded target tensor (EOS = num_nodes)
            tgt = torch.full((B, T), num_nodes, dtype=torch.long, device=device)  # default EOS
            for b, seq in enumerate(y_batch):
                tgt[b, : len(seq)] = torch.tensor(seq, device=device)

            loss = F.cross_entropy(logits.view(-1, C), tgt.view(-1))
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss / n_batches:.4f}")

if __name__ == "__main__":
    train_pointer_network()