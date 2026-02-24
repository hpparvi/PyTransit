import jax.numpy as jnp

def ld_linear(mu, pv):
    """Linear limb darkening model: I(mu) = 1 - u*(1 - mu).

    Parameters
    ----------
    mu : jax array
        Array of mu values.
    pv : jax array
        Limb darkening coefficients, where ``pv[0]`` is the linear
        coefficient u.

    Returns
    -------
    intensity : jax array
        Limb darkening profile evaluated at each mu.
    """
    return 1.0 - pv[0] * (1.0 - mu)

def ldi_linear(pv):
    """Disk-integrated intensity for the linear limb darkening model.

    Parameters
    ----------
    pv : jax array
        Limb darkening coefficients, where ``pv[0]`` is the linear
        coefficient u.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return 2 * jnp.pi / 6 * (3 - pv[0])
