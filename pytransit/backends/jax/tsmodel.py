"""Transmission spectroscopy transit model for JAX.

JAX-compatible (jit + grad) transmission spectroscopy model supporting
multiple passbands with per-passband radius ratios. Computes geometry once
at kmean and adjusts per passband for efficiency.

Uses branchless jnp.where control flow for full differentiability.
No @jax.custom_jvp needed — standard JAX tracing produces correct gradients.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

from meepmeep.backends.jax.ts2d import solve_xy_p5
from .ccintersection import ccia
from .rrmodel import _compute_half_window, _pd_t15, _interpolate_ldm


def tsmodel(times, k, t0, p, a, i, e, w,
            nsamples, exptimes, ldp, istar,
            weights, dk, kmin, kmax, dg, ze,
            npb, max_ns):
    """Transmission spectroscopy transit model (JAX) for a single parameter vector.

    Parameters
    ----------
    times : array (npt,)
        Observation times.
    k : array (npb,)
        Planet-to-star radius ratio per passband.
    t0, p, a, i, e, w : float
        Orbital parameters (scalars for single PV).
    nsamples : int
        Number of supersamples (scalar, homogeneous).
    exptimes : float
        Exposure time (scalar, homogeneous).
    ldp : array (npb, nmu)
        Limb darkening profile per passband.
    istar : array (npb,)
        Disk-integrated intensity per passband.
    weights : array (nk, ng, nmu)
        Pre-computed weight table.
    dk : float
        Radius ratio step size in weight table.
    kmin : float
        Minimum radius ratio in weight table.
    kmax : float
        Maximum radius ratio in weight table.
    dg : float
        Grazing parameter step size.
    ze : array
        Radial zone edges (unused, API compatibility).
    npb : int
        Number of passbands (static).
    max_ns : int
        Maximum number of supersamples (static, for fori_loop bound).

    Returns
    -------
    array (npb, npt)
        Model flux (1.0 out of transit).
    """
    nk = weights.shape[0]

    # Mean and max radius ratios
    kmean = jnp.mean(k)
    kmax_val = jnp.max(k)
    afac = k ** 2 / kmean ** 2

    # Taylor coefficients for the orbit
    cf = solve_xy_p5(0.0, p, a, i, e, w)

    # Transit half-window
    half_window = _compute_half_window(kmax_val, p, a, i, e, w)

    # Pre-compute LD means per passband by interpolating weight table at kmean
    ik = jnp.clip(jnp.floor((kmean - kmin) / dk).astype(int), 0, nk - 2)
    ak = jnp.clip((kmean - kmin - ik * dk) / dk, 0.0, 1.0)

    def _compute_ldm_pb(ldp_pb):
        return (1.0 - ak) * jnp.dot(weights[ik], ldp_pb) + ak * jnp.dot(weights[ik + 1], ldp_pb)

    ldm_all = jax.vmap(_compute_ldm_pb)(ldp)  # (npb, ng)

    # dA/dk at kmean, computed via autodiff
    _dadk_fn = jax.grad(ccia, argnums=1)

    def _single_time(tc):
        epoch = jnp.floor((tc - t0 + 0.5 * p) / p)
        dt = tc - (t0 + epoch * p)

        def body(isample, acc):
            offset = exptimes * ((isample + 0.5) / nsamples - 0.5)
            z = _pd_t15(tc + offset, t0, p, cf)

            # Overlap area and its k-derivative at kmean
            ap0 = ccia(1.0, kmean, z)
            dadk = _dadk_fn(1.0, kmean, z)

            g = z / (1.0 + kmean)
            fully_contained = z <= 1.0 - kmax_val

            def _per_passband(k_pb, ldm_pb, istar_pb, afac_pb):
                iplanet = _interpolate_ldm(g, dg, ldm_pb)
                area_adj = jnp.where(fully_contained,
                                     ap0 * afac_pb,
                                     ap0 + (k_pb - kmean) * dadk)
                return (istar_pb - iplanet * area_adj) / istar_pb

            sample_flux = jax.vmap(_per_passband)(k, ldm_all, istar, afac)
            return acc + jnp.where(isample < nsamples, sample_flux, 0.0)

        flux = lax.fori_loop(0, max_ns, body, jnp.zeros(npb)) / nsamples
        in_window = jnp.abs(dt) < half_window
        return jnp.where(in_window, flux, 1.0)

    return jax.vmap(_single_time)(times).T  # (npb, npt)
