import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # encoder: single‐layer LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # decoder: a single LSTMCell
        self.decoder_cell = nn.LSTMCell(input_dim+1, hidden_dim)
        # attention parameters
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v  = nn.Linear(hidden_dim, 1, bias=False)
        # learnable initial decoder input (e.g. zero vector)
        self.decoder_start = nn.Parameter(torch.zeros(input_dim+1))
        
        
    def forward(self, Q_rows, target_seq=None):
        # Q_rows: [B, n, input_dim]
        B, n, _ = Q_rows.size()
        # 1) encode
        enc_out, (h_enc, c_enc) = self.encoder(Q_rows)   # enc_out: [B, n, hidden_dim]
        # 2) decode step by step with teacher forcing
        logits = []
        h_dec, c_dec = h_enc[0], c_enc[0]                 # [B, hidden_dim]
        y_in = self.decoder_start.unsqueeze(0).expand(B, -1)
        # y_in = self.decoder_start.unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
        for i in range(n+1):
            # run one step of decoder
            h_dec, c_dec = self.decoder_cell(y_in, (h_dec, c_dec))
            # compute attention scores over encoder outputs
            # score_ij = v^T tanh(W1 x_j + W2 h_dec)
            e1 = self.W1(Q_rows)                          # [B, n, hidden_dim]
            e2 = self.W2(h_dec).unsqueeze(1)              # [B, 1, hidden_dim]
            scores = self.v(torch.tanh(e1 + e2)).squeeze(-1)  # [B, n]
            logits.append(scores)
            # next decoder input: use ground-truth embedding if provided
            if target_seq is not None:
                # target_seq one‐hot → index → use that Q_row as next y_in
                idx = target_seq[:, i]                   # [B]
                y_in = Q_rows[torch.arange(B), idx]
            else:
                # greedy during inference
                idx = scores.argmax(dim=1)
                y_in = Q_rows[torch.arange(B), idx]
        # stacked logits: [B, n+1, n]
        return torch.stack(logits, dim=1)


def train_model(model, train_loader, n, max_epochs=10**4, ):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(max_epochs):
        for Q_rows, target_seq in train_loader:
            logits = model(Q_rows, target_seq)
            loss = criterion(logits.view(-1,n), target_seq.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
