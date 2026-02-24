import jax.numpy as jnp

def ld_general(mu, pv):
    """General limb darkening model: I(mu) = 1 - sum(c_i * (1 - mu^(i+1))).

    Supports an arbitrary number of coefficients.

    Parameters
    ----------
    mu : jax array
        Array of mu values.
    pv : jax array
        Limb darkening coefficients ``[c_1, c_2, ..., c_n]``.

    Returns
    -------
    intensity : jax array
        Limb darkening profile evaluated at each mu.
    """
    i = jnp.arange(pv.shape[0])
    terms = pv[:, None] * (1.0 - mu[None, :] ** (i[:, None] + 1))
    return terms.sum(axis=0)

def ldi_general(pv):
    """Disk-integrated intensity for the general limb darkening model.

    Parameters
    ----------
    pv : jax array
        Limb darkening coefficients ``[c_1, c_2, ..., c_n]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    i = jnp.arange(pv.shape[0])
    return 2 * jnp.pi * jnp.sum(pv * (i + 1.0) / (2.0 * (i + 3.0)))
