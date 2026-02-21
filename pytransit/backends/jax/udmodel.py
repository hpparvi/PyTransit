"""Uniform-disk transit model for JAX.

JAX-compatible (jit + grad) uniform stellar disk transit model.
Uses jnp.where-based branchless ccia from ccintersection.py,
so all code paths are traced and the function remains differentiable.

Derivative propagation uses @jax.custom_jvp: the explicit Taylor-coefficient
derivative chain (dcf -> dz -> dflux) is registered as the JVP rule so that
jax.grad works transparently without tracing through the full computation graph.
"""

from meepmeep.backends.jax.ts2d import solve_xy_p5_d, pd_t15_d
from .ccintersection import ccia_and_grad
import jax.numpy as jnp
import jax


def _uniform_model_fwd(times, k, t0, p, a, i, e, w):
    """Forward pass returning flux and per-parameter Jacobian.

    Returns
    -------
    flux : array (npt,)
        Flux deviation (negative during transit, zero outside).
    dflux : array (npt, 7)
        Jacobian rows: dflux/d(k, t0, p, a, i, e, w).
    """
    cf, dcf = solve_xy_p5_d(0.0, p, a, i, e, w)

    # Transit half-window (T14/2 with safety margin)
    b_imp = a * jnp.cos(i) * (1.0 - e**2) / (1.0 + e * jnp.sin(w))
    ae = jnp.sqrt(1.0 - e**2) / (1.0 + e * jnp.sin(w))
    sin_arg = jnp.sqrt(jnp.maximum((1.0 + k)**2 - b_imp**2, 0.0)) / (a * jnp.sin(i))
    half_window = 0.025 + 0.5 * p / jnp.pi * jnp.arcsin(jnp.minimum(sin_arg, 1.0)) * ae

    def _single_time(tc):
        epoch = jnp.floor((tc - t0 + 0.5 * p) / p)
        dt = tc - (t0 + epoch * p)

        z, dz = pd_t15_d(tc, t0, p, cf, dcf)
        z = jnp.abs(z)

        area, dadk, dadz = ccia_and_grad(1.0, k, z)
        flux = -area / jnp.pi

        # dz layout: (dz/dphase, dz/dp, dz/da, dz/di, dz/de, dz/dw)
        # d(phase)/d(t0) = -1, so dflux/dt0 picks up a sign flip.
        dflux_k  = -dadk / jnp.pi
        dflux_t0 =  dadz * dz[0] / jnp.pi
        dflux_p  = -dadz * dz[1] / jnp.pi
        dflux_a  = -dadz * dz[2] / jnp.pi
        dflux_i  = -dadz * dz[3] / jnp.pi
        dflux_e  = -dadz * dz[4] / jnp.pi
        dflux_w  = -dadz * dz[5] / jnp.pi

        jac = jnp.array([dflux_k, dflux_t0, dflux_p, dflux_a,
                          dflux_i, dflux_e, dflux_w])

        in_window = jnp.abs(dt) < half_window
        flux = jnp.where(in_window, flux, 0.0)
        jac = jnp.where(in_window, jac, 0.0)

        return flux, jac

    return jax.vmap(_single_time)(times)


@jax.custom_jvp
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
    flux, _ = _uniform_model_fwd(times, k, t0, p, a, i, e, w)
    return flux


@uniform_model.defjvp
def uniform_model_jvp(primals, tangents):
    times, k, t0, p, a, i, e, w = primals
    times_dot, k_dot, t0_dot, p_dot, a_dot, i_dot, e_dot, w_dot = tangents

    flux, dflux = _uniform_model_fwd(times, k, t0, p, a, i, e, w)

    # dflux columns: (k, t0, p, a, i, e, w)
    flux_dot = (dflux[:, 0] * k_dot
              + dflux[:, 1] * t0_dot
              + dflux[:, 2] * p_dot
              + dflux[:, 3] * a_dot
              + dflux[:, 4] * i_dot
              + dflux[:, 5] * e_dot
              + dflux[:, 6] * w_dot)

    return flux, flux_dot
