import jax.numpy as np
import numpyro

import numpyro.distributions as dist

numpyro.set_platform("cpu")
numpyro.set_host_device_count(10)

J = 8

y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])

sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])



# Eight Schools example

def eight_schools(J, sigma, y=None):

    mu = numpyro.sample('mu', dist.Normal(0, 5))

    log_tau = numpyro.sample('log_tau', dist.Normal(0,5))

    with numpyro.plate('J', J):

        theta = numpyro.sample('theta', dist.Normal(mu, np.exp(log_tau)))

        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

from jax import random
from numpyro.infer import MCMC, NUTS 

from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

reparam_model = reparam(eight_schools, config={"mu": LocScaleReparam(0),"log_tau": LocScaleReparam(0),"theta": LocScaleReparam(0)})

nuts_kernel = NUTS(eight_schools)

mcmc = MCMC(nuts_kernel, num_warmup=2000, num_samples=10000 ,num_chains=10)

rng_key = random.PRNGKey(0)

mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))

mcmc.print_summary()

nuts_kernel2 = NUTS(reparam_model)

mcmc2 = MCMC(nuts_kernel2, num_warmup=2000, num_samples=10000 ,num_chains=10)

rng_key2 = random.PRNGKey(0)

mcmc2.run(rng_key2, J, sigma, y=y, extra_fields=('potential_energy',))

mcmc2.print_summary() 