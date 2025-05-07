import random
import torch
import pathlib
from typing import Tuple


class MaxCutDatasetBuilder:
    """Generate pointer-network training files exactly like Gu & Yang (2020)."""

    def __init__(self, n: int, eos_index: int = 0):
        self.n = n
        self.eos_index = eos_index  # 0 ← EOS, 1..n ← vertices
        assert eos_index == 0, "Article format requires EOS = 0 and 1-based vertices"

    # --------------------------------------------------------------------- #
    # 1.  Simple SBM graph sampler (unchanged)                              #
    # --------------------------------------------------------------------- #
    def _sample_sbm(self, x_a: float, p_inter: float, y_b: float,
                    size_A_mu: float, size_A_sigma: float) -> Tuple[torch.Tensor, torch.BoolTensor]:
        is_A = torch.zeros(self.n, dtype=torch.bool)
        size_A = int(max(1, min(self.n - 1, random.gauss(size_A_mu, size_A_sigma))))
        is_A[torch.randperm(self.n)[:size_A]] = True

        Q = torch.zeros(self.n, self.n)
        for u in range(self.n):
            for v in range(u + 1, self.n):
                p = x_a if (is_A[u] and is_A[v]) else y_b if (~is_A[u] and ~is_A[v]) else p_inter
                if random.random() < p:
                    Q[u, v] = Q[v, u] = 1.0
        return Q, is_A

    # --------------------------------------------------------------------- #
    # 2.  Build a whole dataset                                             #
    # --------------------------------------------------------------------- #
    def build_dataset(self, num_samples: int,
                      x_a: float = 0.8, p_inter: float = 0.2, y_b: float = 0.75,
                      size_A_mu: float = 25, size_A_sigma: float = 3):
        Qs, seqs, masks = [], [], []
        for _ in range(num_samples):
            Q, is_A = self._sample_sbm(x_a, p_inter, y_b, size_A_mu, size_A_sigma)
            Qs.append(Q)
            seqs.append(self._membership_to_sequence(is_A))
            masks.append(is_A)
        self.Q_rows = torch.stack(Qs)  # [B, n, n]
        self.targets = torch.stack(seqs)  # [B, n+1]
        self.masks = masks

    # --------------------------------------------------------------------- #
    # 3.  Write / read in article format                                    #
    # --------------------------------------------------------------------- #
    def write_dataset(self, path: str):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            f.write(f"{len(self.Q_rows)} {self.n}\n")
            for Q, seq in zip(self.Q_rows, self.targets):
                for row in Q:
                    f.write(' '.join(map(str, row.tolist())) + '\n')
                f.write(' '.join(map(str, seq.tolist())) + '\n')

    @staticmethod
    def read_dataset(path: str):
        with open(path) as f:
            m, n = map(int, f.readline().split())
            Q = torch.zeros(m, n, n)
            tgt = torch.zeros(m, n + 1, dtype=torch.long)
            for i in range(m):
                for r in range(n):
                    Q[i, r] = torch.tensor(list(map(float, f.readline().split())))
                tgt[i] = torch.tensor(list(map(int, f.readline().split())))
        return Q, tgt

    def write_ground_truth_matrix(self, path: str):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            f.write(f"{len(self.Q_rows)} {self.n}\n")
            for mask in self.masks:
                M = ((mask & mask[:, None]) | (~mask & ~mask[:, None])).float()
                torch.diagonal(M).fill_(0.0)
                for row in M:
                    f.write(' '.join(map(str, row.tolist())) + '\n')

    # --------------------------------------------------------------------- #
    # 4.  Helper: build <A vertices> + EOS + padding                        #
    # --------------------------------------------------------------------- #
    def _membership_to_sequence(self, is_A: torch.BoolTensor) -> torch.LongTensor:
        A = torch.nonzero(is_A, as_tuple=True)[0] + 1  # 1-based vertex IDs
        seq = torch.full((self.n + 1,), self.eos_index, dtype=torch.long)
        seq[:len(A)] = A  # vertices in set A
        seq[len(A)] = self.eos_index  # single EOS
        # rest remain EOS (0)
        return seq


# Example usage
builder = MaxCutDatasetBuilder(n=50)  # Dynamically uses n=50
builder.build_dataset(num_samples=3)
print(builder.targets[0])  # e.g. tensor([ 1,  3, 15, 20,  0,  0,  0, … ])
builder.write_dataset('experiments/max_cut/train.txt')