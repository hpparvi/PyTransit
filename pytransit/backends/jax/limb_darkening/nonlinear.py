import jax.numpy as jnp

def ld_nonlinear(mu, pv):
    """Nonlinear limb darkening model: I(mu) = 1 - sum(c_n * (1 - mu^(n/2))).

    Uses four coefficients for n = 1, 2, 3, 4.

    Parameters
    ----------
    mu : jax array
        Array of mu values.
    pv : jax array
        Limb darkening coefficients ``[c1, c2, c3, c4]``.

    Returns
    -------
    intensity : jax array
        Limb darkening profile evaluated at each mu.
    """
    return (1.0 - pv[0] * (1.0 - mu**0.5) - pv[1] * (1.0 - mu)
            - pv[2] * (1.0 - mu**1.5) - pv[3] * (1.0 - mu**2))

def ldi_nonlinear(pv):
    """Disk-integrated intensity for the nonlinear limb darkening model.

    Parameters
    ----------
    pv : jax array
        Limb darkening coefficients ``[c1, c2, c3, c4]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    return 2 * jnp.pi * (0.5 - pv[0]/10 - pv[1]/6 - 3*pv[2]/14 - pv[3]/4)
