import jax.numpy as jnp

def ld_general(mu, pv):
    i = jnp.arange(pv.shape[0])
    terms = pv[:, None] * (1.0 - mu[None, :] ** (i[:, None] + 1))
    return terms.sum(axis=0)

def ldi_general(pv):
    i = jnp.arange(pv.shape[0])
    return 2 * jnp.pi * jnp.sum(pv * (i + 1.0) / (2.0 * (i + 3.0)))
