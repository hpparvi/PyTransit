"""RoadRunner limb-darkened transit model for JAX.

JAX-compatible (jit + grad) RoadRunner transit model. Uses pre-computed weight
tables for efficient limb-darkening integration, with branchless jnp.where
control flow for full differentiability.

Unlike the uniform-disk model, no @jax.custom_jvp is needed — standard JAX
tracing with jnp.where produces correct gradients via jax.grad / jax.jacobian.
"""

import jax
import jax.numpy as jnp

from meepmeep.backends.jax.ts2d import solve_xy_p5
from .ccintersection import ccia


def _pd_t15(tc, t0, p, c):
    """Projected planet-star distance from Taylor coefficients.

    Plain forward-only version (no explicit derivatives) since JAX autodiff
    handles differentiation through this function.
    """
    epoch = jnp.floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    return jnp.sqrt(px ** 2 + py ** 2)


def _interpolate_ldm(g, dg, ldm):
    """Branchless linear interpolation of the mean limb darkening profile."""
    ng = ldm.shape[0]
    ig = jnp.clip(jnp.floor(g / dg).astype(int), 0, ng - 2)
    alpha = jnp.clip((g - ig * dg) / dg, 0.0, 1.0)
    val = (1.0 - alpha) * ldm[ig] + alpha * ldm[ig + 1]
    return jnp.where((g >= 0.0) & (g <= 1.0), val, 0.0)


def rr_simple(times, k, t0, p, a, i, e, w,
              nsamples, exptimes, ldp, ldg, ldi, dldi,
              weights, dk, kmin, kmax, dg, ze):
    """RoadRunner limb-darkened transit model (JAX).

    Compatible with jax.jit and jax.grad / jax.jacobian.

    Parameters
    ----------
    times : array
        Observation times.
    k : float
        Planet-to-star radius ratio.
    t0 : float
        Mid-transit time.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis (a/R*).
    i : float
        Orbital inclination [rad].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [rad].
    ldp : array, shape (nmu,)
        Limb darkening profile values at radial zone midpoints.
    ldi : float
        Disk-integrated limb darkening intensity.
    weights : array, shape (nk, ng, nmu)
        Pre-computed weight table for limb darkening integration.
    dk : float
        Radius ratio step size in the weight table.
    kmin : float
        Minimum radius ratio in the weight table.
    kmax : float
        Maximum radius ratio in the weight table.
    dg : float
        Grazing parameter step size.
    ze : array
        Radial zone edges (unused in JAX version, kept for API compatibility).

    Returns
    -------
    array, shape (npt,)
        Model flux (1.0 out of transit).
    """
    nk = weights.shape[0]

    # Interpolate weight table to get mean limb darkening profile for this k
    ik = jnp.clip(jnp.floor((k - kmin) / dk).astype(int), 0, nk - 2)
    ak = jnp.clip((k - kmin - ik * dk) / dk, 0.0, 1.0)
    ldm = (1.0 - ak) * jnp.dot(weights[ik], ldp[0,0]) + ak * jnp.dot(weights[ik + 1], ldp[0,0])

    # Compute Taylor coefficients for the orbit
    cf = solve_xy_p5(0.0, p, a, i, e, w)

    # Compute transit half-window
    b_imp = a * jnp.cos(i) * (1.0 - e ** 2) / (1.0 + e * jnp.sin(w))
    ae = jnp.sqrt(1.0 - e ** 2) / (1.0 + e * jnp.sin(w))
    sin_arg = jnp.sqrt(jnp.maximum((1.0 + k) ** 2 - b_imp ** 2, 0.0)) / (a * jnp.sin(i))
    half_window = 0.025 + 0.5 * p / jnp.pi * jnp.arcsin(jnp.minimum(sin_arg, 1.0)) * ae

    def _single_time(tc):
        epoch = jnp.floor((tc - t0 + 0.5 * p) / p)
        dt = tc - (t0 + epoch * p)
        z = _pd_t15(tc, t0, p, cf)
        g = z / (1.0 + k)
        iplanet = _interpolate_ldm(g, dg, ldm)
        aplanet = ccia(1.0, k, z)
        flux = (ldi - iplanet * aplanet) / ldi
        in_window = jnp.abs(dt) < half_window
        return jnp.where(in_window, flux, 1.0)

    return jax.vmap(_single_time)(times)
