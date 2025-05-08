import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from PointerNet import PointerNetwork
from train_network import load_dataset

train_file    = "experiments/max_cut/data/maxcut_train.csv"
test_file     = "experiments/max_cut/data/maxcut_test.csv"
embedding_dim = 128
hidden_dim    = 256
batch_size    = 16
num_epochs    = 5 * 10**2
lr            = 0.1
model = PointerNetwork(input_dim=n,
                       embedding_dim=embedding_dim,
                       hidden_dim=hidden_dim).to("cpu")
model.load_state_dict(torch.load("ptr_net_weights.pth"))
# X_train, Y_train, n_train = load_dataset(train_file)
X_test,  Y_test,  n_test  = load_dataset(test_file)
n = 10
model.eval()
with torch.no_grad():
    N_test = X_test.size(0)
    correct = 0
    random_samples = []  

    for i in range(0, N_test, batch_size):
        batch_X = X_test[i:i+batch_size]
        outputs = model(batch_X)  # List of length ≤ batch_size of index‐sequences
        for j, out_seq in enumerate(outputs):
            # Find EOS
            eos_pos = out_seq.index(n) if n in out_seq else len(out_seq)
            chosen = set(out_seq[:eos_pos])
            # Reconstruct predicted binary vector
            pred = [1 if idx in chosen else 0 for idx in range(n)]
            target = Y_test[i+j].tolist()

            # Compare prediction with target
            if pred == target:
                correct += 1

            # Collect random samples for printing
            if len(random_samples) < 3 and random.random() < 0.1:  # 10% chance to pick a sample
                random_samples.append((pred, target))

    # Print 3 random outputs compared to targets
    print("\nRandom Model Outputs Compared to Targets:")
    for idx, (pred, target) in enumerate(random_samples[:3]):
        print(f"Sample {idx + 1}:")
        print(f"  Model Output: {pred}")
        print(f"  Target:       {target}")
        print(f"  Match:        {'Yes' if pred == target else 'No'}")

    # Calculate and print accuracy
    accuracy = correct / N_test * 100
    print(f"\nTest Accuracy: {correct}/{N_test} = {accuracy:.2f}%")
    torch.save(model.state_dict(), "experiments/max_cut/saved_models/ptr_net_weights.pth")
    # print("Saved model in file ptr_net_weights.pth")