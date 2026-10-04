"""Uniform-disk transit model for JAX.

JAX-compatible (jit + grad) uniform stellar disk transit model supporting
heterogeneous light curves with multiple passbands, epochs, and supersampling.

The flux is a plain JAX function of the parameters (`_flux`), and its Jacobian is
obtained from it by forward-mode autodiff (`_udmodel_fwd`). `udmodel` registers
that Jacobian as a @jax.custom_jvp rule. The rule is linear in the tangents, so
jax.grad works on `udmodel` even though the supersampling loop has a traced trip
count, which reverse-mode autodiff cannot differentiate through directly.
"""

import functools

import jax
import jax.numpy as jnp

from meepmeep.jax2d import solve2d, pos_c
from .ccintersection import ccia


def _compute_half_window(k_max, p, a, inc, e, w):
    """Compute transit half-window (T14/2 with safety margin)."""
    b_imp = a * jnp.cos(inc) * (1.0 - e**2) / (1.0 + e * jnp.sin(w))
    ae = jnp.sqrt(1.0 - e**2) / (1.0 + e * jnp.sin(w))
    sin_arg = jnp.sqrt(jnp.maximum((1.0 + k_max)**2 - b_imp**2, 0.0)) / (a * jnp.sin(inc))
    return 0.025 + 0.5 * p / jnp.pi * jnp.arcsin(jnp.minimum(sin_arg, 1.0)) * ae


def _point_indices(lcids, pbids, epids, nep):
    """Passband, transit-centre and orbit index of every time point."""
    itcs = epids[lcids]
    ieps = itcs if nep > 1 else jnp.zeros_like(lcids)
    return pbids[lcids], itcs, ieps


def _flux(times, k, t0, p, a, i, e, w,
          lcids, pbids, epids, nsamples, exptimes,
          npb, nep):
    """Flux deficit, shape (npt,). See `udmodel` for the parameters."""
    # Taylor coefficients of the sky-plane position, one set per orbit
    cfs = jax.vmap(lambda pp, aa, ii, ee, ww: solve2d(0.0, pp, aa, ii, ee, ww))(p, a, i, e, w)

    # Compute half-window per orbit using max k across passbands
    k_max = jnp.max(k)
    half_windows = jax.vmap(
        lambda pp, aa, ii, ee, ww: _compute_half_window(k_max, pp, aa, ii, ee, ww)
    )(p, a, i, e, w)

    ipbs, itcs, ieps = _point_indices(lcids, pbids, epids, nep)

    def _single_time(tc, k_val, t0_val, p_val, cf, hw, ns, et):
        # Fold time. The epoch has a zero derivative, so the period derivative picks
        # up the -epoch * p term of the folded time through autodiff.
        epoch = jnp.floor((tc - t0_val + 0.5 * p_val) / p_val)
        dt = tc - (t0_val + epoch * p_val)

        def body(isample, acc):
            offset = et * ((isample + 0.5) / ns - 0.5)
            x, y = pos_c(dt + offset, cf)
            # The separation has no derivative at zero. Clamping keeps it finite
            # there, and the flux does not depend on z while the planet is fully
            # inside the disk.
            z = jnp.sqrt(jnp.maximum(x * x + y * y, 1e-30))
            return acc - ccia(1.0, k_val, z) / jnp.pi

        flux = jax.lax.fori_loop(0, ns, body, jnp.zeros_like(dt)) / ns
        return jnp.where(jnp.abs(dt) < hw, flux, 0.0)

    return jax.vmap(_single_time)(times, k[ipbs], t0[itcs], p[ieps], cfs[ieps],
                                  half_windows[ieps], nsamples[lcids], exptimes[lcids])


def _udmodel_fwd(times, k, t0, p, a, i, e, w,
                 lcids, pbids, epids, nsamples, exptimes,
                 npb, nep):
    """Forward pass returning flux and per-parameter Jacobian.

    Parameters
    ----------
    times : array (npt,)
    k : array (npb,)
    t0 : array (ntc,)
    p, a, i, e, w : array (nep,)
    lcids : array (npt,)
    pbids : array (nlc,)
    epids : array (nlc,)
    nsamples : array (nlc,)
    exptimes : array (nlc,)
    npb, nep : int

    Returns
    -------
    flux : array (npt,)
        Flux deviation (negative during transit, zero outside).
    dflux : array (npt, 7)
        Jacobian rows: dflux/d(k, t0, p, a, i, e, w), each taken with respect to
        the time point's own passband, transit centre or orbit.
    """
    params = (k, t0, p, a, i, e, w)
    flux, jvp = jax.linearize(
        lambda *x: _flux(times, *x, lcids, pbids, epids, nsamples, exptimes, npb, nep), *params)

    # A time point depends on one element of each parameter array, so a tangent of
    # ones along a parameter gives the derivative with respect to that element.
    def column(j):
        return jvp(*[jnp.ones_like(x) if n == j else jnp.zeros_like(x) for n, x in enumerate(params)])

    return flux, jnp.stack([column(j) for j in range(7)], axis=1)


@functools.partial(jax.custom_jvp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14))
def udmodel(times, k, t0, p, a, i, e, w,
            lcids, pbids, epids, nsamples, exptimes,
            npb, nep):
    """Uniform-disk transit model (JAX). Returns flux deviation array.

    Parameters
    ----------
    times : array (npt,)
        Observation times.
    k : array (npb,)
        Planet-to-star radius ratio.
    t0 : array (ntc,)
        Mid-transit time, one per epoch.
    p : array (nep,)
        Orbital period.
    a : array (nep,)
        Scaled semi-major axis (a/R*).
    i : array (nep,)
        Orbital inclination [rad].
    e : array (nep,)
        Eccentricity.
    w : array (nep,)
        Argument of periastron [rad].
    lcids : array (npt,)
        Light curve index per time point.
    pbids : array (nlc,)
        Passband index per light curve.
    epids : array (nlc,)
        Epoch index per light curve.
    nsamples : array (nlc,)
        Number of supersamples per light curve.
    exptimes : array (nlc,)
        Exposure time per light curve.
    npb : int
        Number of passbands.
    nep : int
        Number of epochs.

    Returns
    -------
    array (npt,)
        Flux deviation (negative during transit, zero out of transit).
    """
    flux, _ = _udmodel_fwd(times, k, t0, p, a, i, e, w,
                           lcids, pbids, epids, nsamples, exptimes,
                           npb, nep)
    return flux


@udmodel.defjvp
def udmodel_jvp(lcids, pbids, epids, nsamples, exptimes, npb, nep,
                primals, tangents):
    times, k, t0, p, a, i, e, w = primals
    _, k_dot, t0_dot, p_dot, a_dot, i_dot, e_dot, w_dot = tangents

    flux, dflux = _udmodel_fwd(times, k, t0, p, a, i, e, w,
                               lcids, pbids, epids, nsamples, exptimes,
                               npb, nep)

    ipbs, itcs, ieps = _point_indices(lcids, pbids, epids, nep)

    # Contract Jacobian with tangent vectors
    flux_dot = (dflux[:, 0] * k_dot[ipbs]
              + dflux[:, 1] * t0_dot[itcs]
              + dflux[:, 2] * p_dot[ieps]
              + dflux[:, 3] * a_dot[ieps]
              + dflux[:, 4] * i_dot[ieps]
              + dflux[:, 5] * e_dot[ieps]
              + dflux[:, 6] * w_dot[ieps])

    return flux, flux_dot


def udmodel_grad(times, k, t0, p, a, i, e, w,
                 lcids, pbids, epids, nsamples, exptimes,
                 npb, nep):
    """Uniform-disk transit model with gradient (JAX).

    Returns
    -------
    flux : array (npt,)
    dflux : array (npt, 7)
    """
    return _udmodel_fwd(times, k, t0, p, a, i, e, w,
                        lcids, pbids, epids, nsamples, exptimes,
                        npb, nep)
