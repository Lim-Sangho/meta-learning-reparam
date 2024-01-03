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
- Neural Transport HMC (NeuTra HMC) [[Hoffman et al., 2019]](https://arxiv.org/abs/1903.03704) <br>
The paper introduces NeuTra HMC, a technique that enhances the performance of HMC in sampling from challenging posterior distributions. By incorporating inverse autoregressive flows (IAF) and a neural variational inference technique, NeuTra HMC addresses unfavorable geometry in the posterior, allowing for faster convergence and improved asymptotic effective sample sizes. The approach builds upon previous work on transport maps, adapting more powerful and scalable IAF maps to train variational autoencoders (VAEs).

- Variationally Inferred Parameterisation (VIP) [[Gorinova et al., 2020]](http://proceedings.mlr.press/v119/gorinova20a/gorinova20a.pdf) <br>
The VIP algorithm efficiently searches the space of reparameterisation through the gradient-based optimisation of a differentiable variational objective. It can be also used as a pre-processing step for other inference algorithms. <br>
The paper focuses on the parameterisation applied to normally distributed (or any location-scale family) random variables $z \sim N(\mu, \sigma)$, resulting in
$$\hat{z} \sim N(\lambda \mu, \sigma^\lambda),$$
$$ z = \mu + \sigma^{1-\lambda} (\hat{z} - \lambda \mu).$$
This is a continuous relaxation between the non-centred parameterisation (as a special case $\lambda=0$) and the centred parameterisation (as a special case $\lambda=1$). VIP finds the reparameterised model with $\lambda$ that mostly approximates a diagonal-normal. This parameterisation can be applied to any random variables in the location-scale family.

## Method
1. Generate models to train. <br>
Using a simply defined grammar of probabilistic programming language, we automatically generated various models to create a training dataset.

2. Convert models into computation graphs. <br>
We converted the generated models into computation graphs. The resulting computation graphs are simliar to the Bayesian networks, but constants are also represented as nodes, and deterministic operators (e.g., addition, exponential) are also represented as edges, and all edges between random variables are undirected.

3. Construct a GNN on the computation graphs. <br>
For a hyperparameter $d$, each constant node with constant $c$ was initialised by a $d$-dimensional feature vector $[c^0,\, c^1,\, \ldots,\, c^{d-1}]$, and each random variable node was initialised by a $d$-dimensional zero feature vector. Each edge has a type given by its parameterisation (e.g., location/scale of Normal distribution) or deterministic operator (e.g., addition, exponential). Every edge has a $d \times d$ learnable matrix, and edges with the same type share their matrices. The message passing algorithm of the GNN at step $l+1$ is defined by
$$v^{(l+1)} = \phi^{(l)} \left( \sum_{u \in N(v)} A_{(u, v)} u^{(l)} \right)$$
where each of $u, v$ is a node, and $v^{(l)}$ is the feature vector of $v$ at step $l$, and $N(v)$ is a set of neighbors of $v$, and $A_{(u, v)}$ is the learnable matrix of the edge $(u, v)$, and $\phi^{(l)}$ is a learnable function constructed by fully connected layers.

4. Train GNN with variational objectives. <br>
If a random variable node $v$ is in the location-scale family, we use its GNN output node feature $v^{(L)} \in [0, 1]$ as $\lambda$ in VIP to reparameterise $v$ in the model. As in the VIP, we trained the GNN to minimise the ELBO loss computed by the diagonal normal guide and the reparameterised model with $\lambda$'s.

5. Run HMC on the reparameterised models. <br>
Once we train the GNN, we can run the forward pass of GNN and obtain $\lambda$ values of new models that are not included in the training dataset. With these $\lambda$'s, we reparameterised the model and then applied HMC on top of it.

## Experiments
For the training, we randomly generated 1000 Neal's funnel models modified with additional observations $y \sim N(x, \sigma)$ where $y \in [-20, 20]$ and $\sigma \in [0.1, 20]$. As $y$ moves away from $0$ and $\sigma$ approaches $0$, the posterior takes on a different shape from the original Neal's funnel. For the test (i.e., obtaining $\lambda$ and running HMC), we use Neal's funnel models with observations with $y \in \{-20, -10, 0, 10, 20\}$ and $\sigma \in \{0.1, 5, 10, 15, 20\}$.

- Comparison with VIP
- Comparison by effective sample size (ESS)
- Comparison by Gelman-Rubin (GR) diagnostic

## Discussion
- Implicit GNN