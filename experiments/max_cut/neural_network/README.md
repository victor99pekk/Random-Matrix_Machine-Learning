
# Maximum cut: `Heuristic algorithm` vs `Deep Learning`
In this experiment we compare the performance of a `heuristic maximum cut` algorithm, with `deep learning` to solve the maxmimum cut problem the best we can.





## Results:
The best-performing network achieves just over `80%` accuracy in partitioning all nodes correctly on graphs with `100 nodes`. This means that, for these graphs, the model correctly classifies every node into its respective group more than 80% of the time. This network is saved in `"saved_models/n=100_82%.pth"`.

Other sizes of graphs we tried was `10` and `50` which we achieved an accuracy of over `97%` on.

We didnt try with a network bigger that a 100 nodes. And we only trained the networks on a CPU for at most couple hours.

---


## Continuation
I believe if we trained on a GPU for enough time that we could achieve near `100%` on the graph with 100 nodes, and possibly even larger.

An interesting thing to do could be to test it on the `big mac` data to compare it to the paper about maxcut in the `papers/maxcut` folder