import jax.numpy as jnp

def ld_uniform(mu, pv):
    """Uniform (constant) limb darkening: I(mu) = 1.

    Parameters
    ----------
    mu : jax array
        Array of mu (= cos(theta)) values.
    pv : jax array
        Limb darkening coefficients (unused, empty array).

    Returns
    -------
    intensity : jax array
        Ones array of the same size as ``mu``.
    """
    return jnp.ones_like(mu)

def ldi_uniform(pv):
    """Disk-integrated intensity for the uniform limb darkening model.

    Parameters
    ----------
    pv : jax array
        Limb darkening coefficients (unused).

    Returns
    -------
    intensity : float
        Integrated intensity, equal to pi.
    """
    return jnp.pi
