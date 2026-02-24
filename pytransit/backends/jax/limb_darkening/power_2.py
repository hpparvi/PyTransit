import jax.numpy as jnp

def ld_power_2(mu, pv):
    return 1.0 - pv[0] * (1.0 - mu ** pv[1])

def ldi_power_2(pv):
    return jnp.pi * (1.0 - pv[0] * pv[1] / (pv[1] + 2.0))
