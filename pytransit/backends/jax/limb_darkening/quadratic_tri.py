import jax.numpy as jnp

def ld_quadratic_tri(mu, pv):
    """Quadratic limb darkening with Kipping (2013) triangular parameterization.

    Reparameterizes the standard quadratic coefficients (u, v) in terms of
    q1 and q2 via ``u = sqrt(q1) * 2*q2`` and ``v = sqrt(q1) * (1 - 2*q2)``.

    Parameters
    ----------
    mu : jax array
        Array of mu values.
    pv : jax array
        Reparameterized coefficients ``[q1, q2]``.

    Returns
    -------
    intensity : jax array
        Limb darkening profile evaluated at each mu.
    """
    a, b = jnp.sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1.0 - b)
    return 1.0 - u * (1.0 - mu) - v * (1.0 - mu) ** 2

def ldi_quadratic_tri(pv):
    """Disk-integrated intensity for the triangular quadratic model.

    Parameters
    ----------
    pv : jax array
        Reparameterized coefficients ``[q1, q2]``.

    Returns
    -------
    intensity : float
        Integrated intensity over the stellar disk.
    """
    a, b = jnp.sqrt(pv[0]), 2 * pv[1]
    u, v = a * b, a * (1.0 - b)
    return 2 * jnp.pi / 12 * (-2 * u - v + 6)
