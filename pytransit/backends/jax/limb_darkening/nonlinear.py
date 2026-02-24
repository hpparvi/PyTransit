import jax.numpy as jnp

def ld_nonlinear(mu, pv):
    return (1.0 - pv[0] * (1.0 - mu**0.5) - pv[1] * (1.0 - mu)
            - pv[2] * (1.0 - mu**1.5) - pv[3] * (1.0 - mu**2))

def ldi_nonlinear(pv):
    return 2 * jnp.pi * (0.5 - pv[0]/10 - pv[1]/6 - 3*pv[2]/14 - pv[3]/4)
