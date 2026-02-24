import jax.numpy as jnp

def ld_power_2(mu, pv):
    """Power-2 limb darkening model: I(mu) = 1 - c*(1 - mu^alpha).

    Parameters
    ----------
    mu : jax array
        Array of mu values.
    pv : jax array
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    intensity : jax array
        Limb darkening profile evaluated at each mu.
    """
    return 1.0 - pv[0] * (1.0 - mu ** pv[1])

def ldi_power_2(pv):
    """Disk-integrated intensity for the power-2 limb darkening model.

    Parameters
    ----------
    pv : jax array
        Limb darkening coefficients ``[c, alpha]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return jnp.pi * (1.0 - pv[0] * pv[1] / (pv[1] + 2.0))
