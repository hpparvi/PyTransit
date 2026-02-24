import jax.numpy as jnp

def ld_quadratic(mu, pv):
    """Quadratic limb darkening model: I(mu) = 1 - a*(1 - mu) - b*(1 - mu)^2.

    Parameters
    ----------
    mu : jax array
        Array of mu values.
    pv : jax array
        Limb darkening coefficients ``[a, b]``.

    Returns
    -------
    intensity : jax array
        Limb darkening profile evaluated at each mu.
    """
    return 1.0 - pv[0] * (1.0 - mu) - pv[1] * (1.0 - mu) ** 2

def ldi_quadratic(pv):
    """Disk-integrated intensity for the quadratic limb darkening model.

    Parameters
    ----------
    pv : jax array
        Limb darkening coefficients ``[a, b]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return 2 * jnp.pi / 12 * (-2 * pv[0] - pv[1] + 6)
