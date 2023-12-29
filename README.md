# Meta-learning Optimal Reparameterisation of Probabilistic Programs

## Introduction
Existing posterior inference algorithms only work with a specific probabilistic model at a time.
We propose a meta-learning approach to learn an inference algorithm that works with a wide range of models as a white-box manner.

## Preliminary
- Probabilistic programming
- Hamiltonian Monte Carlo (HMC)
- Reparameterisation
  - Neal's funnel
- Graph Neural Network (GNN)

## Related Work
- Variationally Inferred Parameterisation (VIP) [[Gorinova et al., 2020]](http://proceedings.mlr.press/v119/gorinova20a/gorinova20a.pdf)
- Neural Transport HMC [[Hoffman et al., 2019]](https://arxiv.org/abs/1903.03704)
<!-- - Meta-Learning an Inference Algorithm for Probabilistic Programs [[Che and Yang, 2021]](https://arxiv.org/abs/2103.00737) -->

## Method
- Generate models to train
- Parsing models as computation graphs
- Construct a GNN on the computation graphs

## Experiments
- Comparison by effective sample size (ESS)
- Comparison by Gelman-Rubin (GR) diagnostic

## Discussion
- Implicit GNN