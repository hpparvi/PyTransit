"""Uniform-disk transit model for JAX.

JAX-compatible (jit + grad) uniform stellar disk transit model.
Uses jnp.where-based branchless ccia from ccintersection.py,
so all code paths are traced and the function remains differentiable.
"""

from meepmeep.backends.jax.ts2d import solve_xy_p5_d, pd_t15_d
from .ccintersection import ccia
import jax.numpy as jnp
import jax


def uniform_model(times, k, t0, p, a, i, e, w):
    """Uniform-disk transit model (JAX). Returns flux deviation array.

    Compatible with jax.jit and jax.grad (w.r.t. scalar params).

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

    Returns
    -------
    array
        Flux deviation (negative during transit, zero out of transit).
    """
    cf, dcf = solve_xy_p5_d(0.0, p, a, i, e, w)

    # Compute transit half-window (T14/2 with safety margin).
    # The Taylor expansion in pd_t15_d is only valid near transit center,
    # so we must zero out results for times outside the window.
    b = a * jnp.cos(i) * (1.0 - e**2) / (1.0 + e * jnp.sin(w))
    ae = jnp.sqrt(1.0 - e**2) / (1.0 + e * jnp.sin(w))
    sin_arg = jnp.sqrt(jnp.maximum((1.0 + k)**2 - b**2, 0.0)) / (a * jnp.sin(i))
    half_window = 0.025 + 0.5 * p / jnp.pi * jnp.arcsin(jnp.minimum(sin_arg, 1.0)) * ae

    def _single_time(tc):
        epoch = jnp.floor((tc - t0 + 0.5 * p) / p)
        dt = tc - (t0 + epoch * p)
        z, _ = pd_t15_d(tc, t0, p, cf, dcf)
        z = jnp.abs(z)
        area = ccia(1.0, k, z)
        flux = -area / jnp.pi
        return jnp.where(jnp.abs(dt) < half_window, flux, 0.0)

    return jax.vmap(_single_time)(times)
