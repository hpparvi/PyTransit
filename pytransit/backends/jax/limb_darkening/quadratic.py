import jax.numpy as jnp

def ld_quadratic(mu, pv):
    return 1.0 - pv[0] * (1.0 - mu) - pv[1] * (1.0 - mu) ** 2

def ldi_quadratic(pv):
    return 2 * jnp.pi / 12 * (-2 * pv[0] - pv[1] + 6)
