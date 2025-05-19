import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNetwork(nn.Module):
    """Pointer Network model for Max-Cut (supervised learning version). 
    Encodes an input graph (adjacency matrix) and outputs a sequence of node indices 
    indicating one partition (with a special end token separating the two partitions).
    """
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Dimension of each input element's feature vector (for Max-Cut, input_dim = n, the number of nodes).
            embedding_dim: Size of the embeddings for input nodes.
            hidden_dim: Hidden state size for the LSTM encoder and decoder.
        """
        super(PointerNetwork, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(input_dim, embedding_dim) # embedding for each row
         # for encoder to process the rows as a sequence
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # LSTM decoder generates the output sequence of node indices
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # embedding for start vector of the decoder
        self.decoder_start = nn.Parameter(torch.FloatTensor(embedding_dim))
        # Learnable EOS token
        self.enc_eos = nn.Parameter(torch.FloatTensor(hidden_dim))
        nn.init.uniform_(self.decoder_start, -0.1, 0.1)
        nn.init.uniform_(self.enc_eos, -0.1, 0.1)

    def forward(self, adj_matrix: torch.Tensor, target_seq=None):
        """
        Args:
            adj_matrix: Tensor of shape (batch_size, n, n) representing symmetric adjacency matrices of graphs.
                        Each adj_matrix[b] is an n x n matrix of edge weights for a graph with n nodes.
            target_seq: (Optional) List of target sequences (each a list of node indices including EOS represented by index n) 
                        for supervised training. If provided, the function returns the cross-entropy loss.
                        If None, the model will output a predicted sequence of node indices for each input graph.
        Returns:
            If target_seq is provided: torch.Tensor scalar loss (cross-entropy).
            If target_seq is None: a list of output sequences (each sequence is a list of node indices including EOS index).
        """
        batch_size = adj_matrix.size(0)
        n = adj_matrix.size(1)  # number of nodes
        # 1. **Encoder**: Embed each node's adjacency row and run through LSTM encoder
        node_embeds = self.input_embed(adj_matrix)              # shape: (batch_size, n, embedding_dim)
        encoder_outputs, (enc_hidden, enc_cell) = self.encoder_lstm(node_embeds)  # encoder_outputs: (batch, n, hidden_dim)
        # Initialize decoder hidden state and cell state with encoder's final state
        dec_hidden, dec_cell = enc_hidden, enc_cell
        # Prepare the initial decoder input (start token embedding, same for all batch elements)
        dec_input = self.decoder_start.unsqueeze(0).expand(batch_size, -1)  # shape: (batch_size, embedding_dim)
        # Mask to keep track of which indices have been selected. size n+1 to include EOS
        selected_mask = torch.zeros(batch_size, n+1, dtype=torch.bool, device=adj_matrix.device)

        if target_seq is not None:
            # print(target_seq)
            # **Training mode**: compute cross-entropy loss with teacher forcing
            if not isinstance(target_seq, torch.Tensor):
                # Convert list of sequences to a padded tensor (pad with -100 for ignore_index)
                max_len = max(len(seq) for seq in target_seq)
                target_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)
                for i, seq in enumerate(target_seq):
                    target_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                target_seq = target_tensor
            else:
                target_seq = target_seq.long()
            seq_len = target_seq.size(1)
            loss = 0.0
            for t in range(seq_len): #run iterations of steps of LSTM
                _, (dec_hidden, dec_cell) = self.decoder_lstm(dec_input.unsqueeze(1), (dec_hidden, dec_cell))
                # Compute attention (pointer) logits over n nodes + EOS
                # Extend encoder outputs with EOS vector for attention scoring
                eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                # Concatenate encoder outputs and EOS to shape (batch, n+1, hidden_dim)
                extended_enc = torch.cat([encoder_outputs, eos_enc], dim=1)
                # Decoder hidden state for current step
                dec_h = dec_hidden[-1]  # shape: (batch_size, hidden_dim)
                # Attention logits via dot product between dec_h and each encoder output (including EOS)
                logits = torch.bmm(extended_enc, dec_h.unsqueeze(2)).squeeze(2)  # shape: (batch_size, n+1)
                # Mask out already selected indices (including if EOS was selected earlier)
                logits.masked_fill_(selected_mask, float('-inf'))
                # True target index at this step for each sample in batch
                target_indices = target_seq[:, t].to(adj_matrix.device)  # shape: (batch_size,)
                # Compute cross-entropy loss for this step (ignoring padded positions with target -100)
                step_loss = F.cross_entropy(logits, target_indices, ignore_index=-100, reduction='sum')
                loss += step_loss
                # Update mask and decoder input for next step using the target (teacher forcing)
                # Mark selected index (from target) as used
                selected_mask = selected_mask.clone()
                for i in range(batch_size):
                    idx = int(target_indices[i].item())
                    if idx >= 0:
                        selected_mask[i, idx] = True
                # Prepare next decoder input: use the embedding of the selected node, or a zero vector if EOS was selected
                next_inputs = []
                for i in range(batch_size):
                    idx = int(target_indices[i].item())
                    if idx == n:  # EOS index (n)
                        # Use a zero vector (or could use a separate learned EOS embedding for decoder input)
                        next_inputs.append(torch.zeros(self.embedding_dim, device=adj_matrix.device))
                    else:
                        # Use the original embedding of the selected node as next decoder input
                        next_inputs.append(node_embeds[i, idx])
                dec_input = torch.stack(next_inputs, dim=0)  # shape: (batch_size, embedding_dim)
            # Average loss per sequence element
            avg_loss = loss / (batch_size * seq_len)
            return avg_loss

        else:
            # **Inference mode**: generate a sequence of node indices for each graph
            output_sequences = [[] for _ in range(batch_size)]
            for step in range(n + 1):  # maximum output length is n+1 (including all nodes and EOS)
                _, (dec_hidden, dec_cell) = self.decoder_lstm(dec_input.unsqueeze(1), (dec_hidden, dec_cell))
                dec_h = dec_hidden[-1]  # current decoder hidden state, shape: (batch_size, hidden_dim)
                eos_enc = self.enc_eos.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, self.hidden_dim)
                extended_enc = torch.cat([encoder_outputs, eos_enc], dim=1)
                logits = torch.bmm(extended_enc, dec_h.unsqueeze(2)).squeeze(2)  # shape: (batch_size, n+1)
                logits.masked_fill_(selected_mask, float('-inf'))
                # Select the index with maximum logit (highest probability) for each sample
                selected_idx = torch.argmax(logits, dim=1)  # shape: (batch_size,)
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    output_sequences[i].append(idx)
                    selected_mask[i, idx] = True
                # Prepare next decoder input (using the embedding of the selected node or zero if EOS)
                next_inputs = []
                for i in range(batch_size):
                    idx = int(selected_idx[i].item())
                    if idx == n:  # EOS selected
                        next_inputs.append(torch.zeros(self.embedding_dim, device=adj_matrix.device))
                    else:
                        next_inputs.append(node_embeds[i, idx])
                dec_input = torch.stack(next_inputs, dim=0)
            return output_sequences
