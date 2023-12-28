import numpyro
from numpyro.infer.reparam import Reparam

import numpy as np

import jax
import jax.numpy as jnp

from numpyro.distributions import constraints
from numpyro.distributions.util import is_identically_one
from numpyro.util import not_jax_tracer

class LocScaleReparam2(Reparam):
    """
    Generic decentering reparameterizer [1] for latent variables parameterized
    by ``loc`` and ``scale`` (and possibly additional ``shape_params``).

    This reparameterization works only for latent variables, not likelihoods.

    **References:**

    1. *Automatic Reparameterisation of Probabilistic Programs*,
       Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)

    :param float centered: optional centered parameter. If None (default) learn
        a per-site per-element centering parameter in ``[0,1]``. If 0, fully
        decenter the distribution; if 1, preserve the centered distribution
        unchanged.
    :param shape_params: list of additional parameter names to copy unchanged from
        the centered to decentered distribution.
    :type shape_params: tuple or list
    """

    def __init__(self, centered=None, shape_params=(), batch_depth=0):
        assert centered is None or isinstance(
            centered, (int, float, np.generic, np.ndarray, jnp.ndarray, jax.core.Tracer)
        )
        assert isinstance(shape_params, (tuple, list))
        assert all(isinstance(name, str) for name in shape_params)
        if centered is not None:
            is_valid = constraints.unit_interval.check(centered)
            if not_jax_tracer(is_valid):
                if not np.all(is_valid):
                    raise ValueError(
                        "`centered` argument does not satisfy `0 <= centered <= 1`."
                    )

        self.centered = centered
        self.shape_params = shape_params
        self.batch_depth = batch_depth
    def __call__(self, name, fn, obs):
        assert obs is None, "LocScaleReparam does not support observe statements"
        centered = self.centered
        if is_identically_one(centered):
            return fn, obs
        event_shape = fn.event_shape
        batch_shape = fn.batch_shape
        fn, expand_shape, event_dim = self._unwrap(fn)

        # Apply a partial decentering transform.
        params = {key: getattr(fn, key) for key in self.shape_params}
        
        batch_length = len(batch_shape)
        reparam_shape = batch_shape[batch_length-self.batch_depth:] + event_shape
        #print("orig_loc_shape: ",fn.loc.shape)
        #print("orig_scale_shape: ",fn.scale.shape)
        #print("reparam_shape: ",reparam_shape)

        if self.centered is None:
            centered = numpyro.param(
                "{}_centered".format(name),
                jnp.full(reparam_shape, 0.5),
                # constraint=constraints.unit_interval,
            )

        if isinstance(centered, (int, float, np.generic)) and centered == 0.0:
            params["loc"] = jnp.zeros_like(fn.loc)
            params["scale"] = jnp.ones_like(fn.scale)
        else:
            params["loc"] = fn.loc * centered
            params["scale"] = fn.scale**centered
        decentered_fn = self._wrap(type(fn)(**params), expand_shape, event_dim)

        # Draw decentered noise.
        decentered_value = numpyro.sample("{}_decentered".format(name), decentered_fn)
        #print("decentered_shape: ",decentered_value.shape)
        #print("dec_loc_shape: ",decentered_fn.loc.shape)
        #print("dec_scale_shape: ",decentered_fn.scale.shape)

        # Differentiably transform.
        delta = decentered_value - centered * fn.loc
        value = fn.loc + jnp.power(fn.scale, 1 - centered) * delta

        # Simulate a pyro.deterministic() site.
        return None, value