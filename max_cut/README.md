
# Maximum cut: `Heuristic algorithm` vs `Deep Learning`
In this experiment we compare the performance of a `heuristic maximum cut` algorithm, with `deep learning` to solve the maxmimum cut problem the best we can.


## what we need:
1. function to create an adjacency graph between n number of nodes in a graph. connecting the nodes from the different groups with the correct probability
2. heuristic maximum cut algrithm
3. deep learning algorithm, several graphs for training
---

__Potential Benchmark:__ The Biq Mac Library contains a diverse collection of benchmark instances for the Max-Cut problem, and depending on the subdirectory or generator, the probability that two nodes are connected depends on the instance type and its associated edge density parameter.

Link: `https://biqmac.aau.at/biqmaclib.html`

---


## Graph spec
Each graph used in these experiments is generated with a planted partition for the Max-Cut problem:

- Each graph contains `n` nodes, split exactly in half: `n/2` nodes are assigned to group 1 and `n/2` nodes to group 0.
- For each graph, three probabilities are sampled independently: `P1` and `P0` are drawn uniformly from `[0.9, 1.0]` and represent the probability of connecting two nodes within group 1 or group 0, respectively. `P` is drawn uniformly from `[0, 0.1]` and is the probability of connecting nodes across the two groups.
- For every pair of nodes `i < j`:
    - If both nodes are in group 1, they are connected with probability `P1`.
    - If both nodes are in group 0, they are connected with probability `P0`.
    - If the nodes are in different groups, they are connected with probability `P`.
- The result is an `n x n` adjacency matrix of 0/1 edges, and a length-`n` vector indicating the planted partition (the ground truth group assignment for each node).


---