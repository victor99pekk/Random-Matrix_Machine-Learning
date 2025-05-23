
# `LSTM` based Pointer Network


## Network Architecture

The core of our deep learning approach is a **Pointer Network**. This model is designed to process an input graph (represented as an adjacency matrix) and output a sequence of node indices that define one partition of the Max-Cut, using a special end-of-sequence (EOS) token to separate the two partitions.

The Pointer Network consists of:
- An input embedding layer that transforms each node's adjacency vector into a learned embedding.
- An LSTM encoder that processes the sequence of node embeddings, capturing the structure and context of the entire graph.
- An LSTM decoder that generates the output sequence step by step. At each step, it uses an attention mechanism (pointer) to select the next node or the EOS token, based on the current decoder state and the encoder outputs.
- During training, the model uses teacher forcing: it is guided by the ground truth sequence, and the loss is computed using cross-entropy between the predicted logits and the true next node (or EOS) at each step.
- During inference, the model greedily selects the next node or EOS at each step, building the output partition sequence.

This architecture allows the network to learn how to construct a Max-Cut partition by sequentially "pointing" to nodes in the graph, leveraging both the graph structure and the sequence of previous selections.





## Results:
The best-performing network achieves just over `80%` accuracy in partitioning all nodes correctly on graphs with `100 nodes`. This means that, for these graphs, the model correctly classifies every node into its respective group more than 80% of the time. This network is saved in `"saved_models/n=100_82%.pth"`.

Other sizes of graphs we tried was `10` and `50` which we achieved an accuracy of over `97%` on.

We didnt try with a network bigger that a 100 nodes. And we only trained the networks on a CPU for at most couple hours.

---


## Continuation
I believe if we trained on a GPU for enough time that we could achieve near `100%` on the graph with 100 nodes, and possibly even larger.

An interesting thing to do could be to test it on the `big mac` data to compare it to the paper about maxcut in the `papers/maxcut` folder