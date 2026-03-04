"""Uniform-disk transit model for JAX.

JAX-compatible (jit + grad) uniform stellar disk transit model supporting
heterogeneous light curves with multiple passbands, epochs, and supersampling.

Derivative propagation uses @jax.custom_jvp: the explicit Taylor-coefficient
derivative chain (dcf -> dz -> dflux) is registered as the JVP rule so that
jax.grad works transparently without tracing through the full computation graph.
"""

import functools

import jax
import jax.numpy as jnp

from meepmeep.backends.jax.ts2d import solve_xy_p5_d, pd_t15_d
from .ccintersection import ccia_and_grad


def _compute_half_window(k_max, p, a, inc, e, w):
    """Compute transit half-window (T14/2 with safety margin)."""
    b_imp = a * jnp.cos(inc) * (1.0 - e**2) / (1.0 + e * jnp.sin(w))
    ae = jnp.sqrt(1.0 - e**2) / (1.0 + e * jnp.sin(w))
    sin_arg = jnp.sqrt(jnp.maximum((1.0 + k_max)**2 - b_imp**2, 0.0)) / (a * jnp.sin(inc))
    return 0.025 + 0.5 * p / jnp.pi * jnp.arcsin(jnp.minimum(sin_arg, 1.0)) * ae


def _udmodel_fwd(times, k, t0, p, a, i, e, w,
                 lcids, pbids, epids, nsamples, exptimes,
                 npv, npb, nep):
    """Forward pass returning flux and per-parameter Jacobian.

    Parameters
    ----------
    times : array (npt,)
    k : array (npv, npb)
    t0 : array (npv, nep)
    p, a, i, e, w : array (npv, nep)
    lcids : array (npt,)
    pbids : array (nlc,)
    epids : array (nlc,)
    nsamples : array (nlc,)
    exptimes : array (nlc,)
    npv, npb, nep : int

    Returns
    -------
    flux : array (npv, npt)
        Flux deviation (negative during transit, zero outside).
    dflux : array (npv, npt, 7)
        Jacobian rows: dflux/d(k, t0, p, a, i, e, w).
    """
    all_flux = []
    all_dflux = []

    for ipv in range(npv):
        k_pv = k[ipv]       # (npb,)
        t0_pv = t0[ipv]     # (nep,)
        p_pv = p[ipv]       # (nep,)
        a_pv = a[ipv]       # (nep,)
        i_pv = i[ipv]       # (nep,)
        e_pv = e[ipv]       # (nep,)
        w_pv = w[ipv]       # (nep,)

        # Pre-compute Taylor coefficients for all epochs
        cfs, dcfs = jax.vmap(
            lambda pp, aa, ii, ee, ww: solve_xy_p5_d(0.0, pp, aa, ii, ee, ww)
        )(p_pv, a_pv, i_pv, e_pv, w_pv)
        # cfs: (nep, 2, 5), dcfs: (nep, 6, 2, 5)

        # Compute half-window per epoch using max k across passbands
        k_max = jnp.max(k_pv)
        half_windows = jax.vmap(
            lambda pp, aa, ii, ee, ww: _compute_half_window(k_max, pp, aa, ii, ee, ww)
        )(p_pv, a_pv, i_pv, e_pv, w_pv)

        # Gather per-time-point parameters using index arrays
        ilcs = lcids
        ipbs = pbids[ilcs]
        if nep > 1:
            ieps = epids[ilcs]
        else:
            ieps = jnp.zeros_like(ilcs)

        k_pt = k_pv[ipbs]
        t0_pt = t0_pv[ieps]
        p_pt = p_pv[ieps]
        cf_pt = cfs[ieps]
        dcf_pt = dcfs[ieps]
        hw_pt = half_windows[ieps]
        ns_pt = nsamples[ilcs]
        et_pt = exptimes[ilcs]

        def _single_time(tc, k_val, t0_val, p_val, cf, dcf, hw, ns, et):
            # Fold time
            epoch = jnp.floor((tc - t0_val + 0.5 * p_val) / p_val)
            dt = tc - (t0_val + epoch * p_val)

            # Supersampling via lax.fori_loop
            def body(isample, acc):
                flux_acc, dflux_acc = acc
                offset = et * ((isample + 0.5) / ns - 0.5)

                z, dz = pd_t15_d(tc + offset, t0_val, p_val, cf, dcf)
                z = jnp.abs(z)

                area, dadk, dadz = ccia_and_grad(1.0, k_val, z)
                flux_sample = -area / jnp.pi

                dflux_sample = jnp.array([
                    -dadk / jnp.pi,
                     dadz * dz[0] / jnp.pi,
                    -dadz * dz[1] / jnp.pi,
                    -dadz * dz[2] / jnp.pi,
                    -dadz * dz[3] / jnp.pi,
                    -dadz * dz[4] / jnp.pi,
                    -dadz * dz[5] / jnp.pi,
                ])

                return (flux_acc + flux_sample, dflux_acc + dflux_sample)

            init = (jnp.float64(0.0), jnp.zeros(7))
            flux, dflux = jax.lax.fori_loop(0, ns, body, init)
            flux = flux / ns
            dflux = dflux / ns

            # Window mask
            in_window = jnp.abs(dt) < hw
            flux = jnp.where(in_window, flux, 0.0)
            dflux = jnp.where(in_window, dflux, jnp.zeros(7))

            return flux, dflux

        flux_pv, dflux_pv = jax.vmap(_single_time)(
            times, k_pt, t0_pt, p_pt, cf_pt, dcf_pt, hw_pt, ns_pt, et_pt
        )
        all_flux.append(flux_pv)
        all_dflux.append(dflux_pv)

    return jnp.stack(all_flux), jnp.stack(all_dflux)


@functools.partial(jax.custom_jvp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15))
def udmodel(times, k, t0, p, a, i, e, w,
            lcids, pbids, epids, nsamples, exptimes,
            npv, npb, nep):
    """Uniform-disk transit model (JAX). Returns flux deviation array.

    Parameters
    ----------
    times : array (npt,)
        Observation times.
    k : array (npv, npb)
        Planet-to-star radius ratio.
    t0 : array (npv, nep)
        Mid-transit time.
    p : array (npv, nep)
        Orbital period.
    a : array (npv, nep)
        Scaled semi-major axis (a/R*).
    i : array (npv, nep)
        Orbital inclination [rad].
    e : array (npv, nep)
        Eccentricity.
    w : array (npv, nep)
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
    npv : int
        Number of parameter vectors.
    npb : int
        Number of passbands.
    nep : int
        Number of epochs.

    Returns
    -------
    array (npv, npt)
        Flux deviation (negative during transit, zero out of transit).
    """
    flux, _ = _udmodel_fwd(times, k, t0, p, a, i, e, w,
                           lcids, pbids, epids, nsamples, exptimes,
                           npv, npb, nep)
    return flux


@udmodel.defjvp
def udmodel_jvp(lcids, pbids, epids, nsamples, exptimes, npv, npb, nep,
                primals, tangents):
    times, k, t0, p, a, i, e, w = primals
    _, k_dot, t0_dot, p_dot, a_dot, i_dot, e_dot, w_dot = tangents

    flux, dflux = _udmodel_fwd(times, k, t0, p, a, i, e, w,
                               lcids, pbids, epids, nsamples, exptimes,
                               npv, npb, nep)

    # Gather per-time-point tangent indices
    ipbs = pbids[lcids]
    if nep > 1:
        ieps = epids[lcids]
    else:
        ieps = jnp.zeros_like(lcids)

    # Contract Jacobian with tangent vectors
    flux_dot = (dflux[:, :, 0] * k_dot[:, ipbs]
              + dflux[:, :, 1] * t0_dot[:, ieps]
              + dflux[:, :, 2] * p_dot[:, ieps]
              + dflux[:, :, 3] * a_dot[:, ieps]
              + dflux[:, :, 4] * i_dot[:, ieps]
              + dflux[:, :, 5] * e_dot[:, ieps]
              + dflux[:, :, 6] * w_dot[:, ieps])

    return flux, flux_dot


def udmodel_grad(times, k, t0, p, a, i, e, w,
                 lcids, pbids, epids, nsamples, exptimes,
                 npv, npb, nep):
    """Uniform-disk transit model with gradient (JAX).

    Returns
    -------
    flux : array (npv, npt)
    dflux : array (npv, npt, 7)
    """
    return _udmodel_fwd(times, k, t0, p, a, i, e, w,
                        lcids, pbids, epids, nsamples, exptimes,
                        npv, npb, nep)
