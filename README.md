# Meta-learning Optimal Reparameterisation of Probabilistic Programs

## Introduction
Marcov Chain Monte Carlo (MCMC) has been one of successful tools in Bayesian inferences, which gaurantees convergence to exact target statistics.
The main concern in MCMC is to sample in high dimensional spaces, which needs exponentially wide exploration. Therefore, Hamiltonian Monte Carlo (HMC) is widely used to mitigate the obstacle from high dimensionality by using gradients of target distributions to explore the spaces. 
However, its convergence can be slow or even fail if a target density has poor geometric properties such as a multimodal shape and drastic changes of scale in the target density. 

Reparameterisation, which transforms the target distribution with bijective functions so that the target, is one of a frequently-uesd method to change the poor landscape of the target distribution into a proper landscape.
Since appropriate bijective functions for reparameterisation is specific to the target distribution, 
Finding bijective transforms manually needs deep understanding of the target distribution. 

Thanks to probabilistic programming, which treats probabilistic models as programs to which methods in programming analysis can be applicable, there were some researches which constructed parameterised bijective functions compatible to the target distribution and learned the parameters to make the functions become an proper reparmetersation.
However, those researches needs to train the parameters from scaratches for every new model, so it is hard to use their methods in model finding.

In our research, instead of finding a reparameterisation for a speicific model, we tried to meta-learn the reparameterisation; Learning a function which gets probabilistic models (in forms of probabilistic programs) and outputs proper bijective functions for reparameterisation. First, by parsing a probabilistic program, we obtained a graph representation of a probabilistic model. Then, we used GNN on the graph representation to construct bijective functions compatible to the model. Finally, We trained GNN for a wide range of models with a proper probabilistic loss to achieve smooth landscapes of reparemeterised models.

## Preliminary
- Hamiltonian Monte Carlo (HMC) <br>
Hamiltonian Monte Carlo (HMC) is MCMC methods which porposed the next sample based on a simulation of Hamiltonian dynamics.
To sample from traget unnormalised distribution $P(x)$, HMC introduces fictional Hamiltonian system with auxiliary variables $p$, which has the same dimension to $x$ for momentum in Hamiltonian system. The Hamiltonian in the system is written as follows : $H(x,p) = -ln(P(x)) + \frac{1}{2}p^TMp$, and the target distribution becomes proportional to $e^{-H(x,p)}$.
Then, the next proposal is chosen by simulating of Hamiltonian dynamics until fixed time $t$ starting from the current position with the momentum sampled from $Normal(0,M)$. The simulation is done by numerically calculating Hamiltonian equations with a leapfrog integrator. Finally, accept the next proposal with Metropolis acceptance ratio. Thanks to time reversible and volume preserving properties of the leapfrog integrator and Hamiltonian dynamics, the ratio is easy to calculate and likely to be close to 1. <br>
It is well-known that HMC works well in high-dimensional spaces, and there are some proofs of fast convergences of HMC in specific settings of high-dimensional distributions. That's why HMC becomes one of dominent tools for MCMC sampling in continuous spaces. However, HMC doesn't work well when a target distribution has a poor geometry such as multimodality, stiff changes of scales in distribution, no matter how precise the simulation is.


- Reparameterisation <br>
No matter with 

- Probabilistic programming <br>
Probabilistic programming is 

- Graph Neural Network (GNN) <br>

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
For the training, we randomly generated 1000 Neal's funnel models modified with additional observations $y \sim N(x, \sigma)$ where $y \in [-20, 20]$ and $\sigma \in [0.1, 20]$. As $y$ moves away from $0$ and $\sigma$ approaches $0$, the posterior takes on a different shape from the original Neal's funnel. For the test (i.e., obtaining $\lambda$ and running HMC), we use Neal's funnel models with observations with $y \in \{-20, -10, 0, 10, 20\}$ and $\sigma \in \{0.1, 5.075, 10.05, 15.025, 20\}$.

- Comparison with VIP
- Comparison by effective sample size (ESS)
- Comparison by Gelman-Rubin (GR) diagnostic

## Discussion
- Implicit GNN <br>
Time seires model, long range dependence <br>
Naively increasing the number of steps $(L)$ in the GNN is not a good solution as it can lead to the oversmoothing problem in GNNs.

- Meta-learning reparameterisation with more flexible transformations, such as inverse autoregressive flows (IAF).