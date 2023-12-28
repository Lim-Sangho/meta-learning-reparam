import pyro
import torch
import FlowDistributions as dist


def neals_funnel(y: torch.Tensor, sigma: torch.Tensor):
    with pyro.plate("model", y.shape[-1]):
        z = pyro.sample("z", dist.Normal(0, 1))
        x = pyro.sample("x", dist.Normal(0, z.exp()))
        pyro.sample("y", dist.Normal(x, sigma), obs=y)


def eight_schools(y: torch.Tensor, sigma: torch.Tensor):
    with pyro.plate("model", y.shape[-1]):
        mu = pyro.sample("mu", dist.Normal(0, 10))
        tau = pyro.sample("tau", dist.Normal(5, 1))
        with pyro.plate("school", y.shape[-2]):
            theta = pyro.sample("theta", dist.Normal(mu, tau.exp()))
            pyro.sample("y", dist.Normal(theta, sigma), obs=y)