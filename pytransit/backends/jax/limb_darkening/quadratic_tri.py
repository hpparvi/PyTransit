import jax.numpy as jnp

def ld_quadratic_tri(mu, pv):
    a, b = jnp.sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1.0 - b)
    return 1.0 - u * (1.0 - mu) - v * (1.0 - mu) ** 2

def ldi_quadratic_tri(pv):
    a, b = jnp.sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1.0 - b)
    return 2 * jnp.pi / 12 * (-2 * u - v + 6)
