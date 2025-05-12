
# Maximum cut: `Heuristic algorithm` vs `Deep Learning`
In this experiment we compare the performance of a `heuristic maximum cut` algorithm, with `deep learning` to solve the maxmimum cut problem the best we can.


## what we need:
1. function to create an adjacency graph between n number of nodes in a graph. connecting the nodes from the different groups with the correct probability
2. heuristic maximum cut algrithm
3. deep learning algorithm, several graphs for training


__`Graph spec:`__ the graphs we use in this experiments contain nodes, where every node contain one of two groups, A or B. Two nodes from the same group are connected with probability 0.9, and nodes that arent in the same group are connected with probability 0.1. the file adj_matrices.txt contain 10 different adjacency matrices for 10 different graphs.

`Creation of graphs:` the function `create_graphs()` creates a .txt file with 10 different 20x20 adjacency matrices, graphs with 20 nodes in them. each entry is either 1 or 0 depending if the nodes are connected or not.