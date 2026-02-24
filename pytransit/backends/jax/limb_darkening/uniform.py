import jax.numpy as jnp

def ld_uniform(mu, pv):
    return jnp.ones_like(mu)

def ldi_uniform(pv):
    return jnp.pi
