# Unbalanced (KL) Wasserstein distance

Implementation of the KL relaxed Wasserstein distance
based on Unbalanced Optimal Transport theory
introduced in (Chizat et al., 2015) using Generalized Sinkhorn algorithm.
For 2d inputs, Sinkhorn iterations can be blazinlgy accelerated using convolutions
(Solomon et al., SIGGRAPH 2015).
