import jax.numpy as jnp

def ld_linear(mu, pv):
    return 1.0 - pv[0] * (1.0 - mu)

def ldi_linear(pv):
    return 2 * jnp.pi / 6 * (3 - pv[0])
