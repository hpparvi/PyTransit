"""RoadRunner limb-darkened transit model for JAX.

JAX-compatible (jit + grad) RoadRunner transit model supporting heterogeneous
light curves with multiple passbands, epochs, and supersampling configurations.

Uses pre-computed weight tables for efficient limb-darkening integration, with
branchless jnp.where control flow for full differentiability.

No @jax.custom_jvp is needed — standard JAX tracing with jnp.where produces
correct gradients via jax.grad / jax.jacobian.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

from meepmeep.backends.jax.ts2d import solve_xy_p5
from .ccintersection import ccia


def _compute_half_window(k_max, p, a, inc, e, w):
    """Compute transit half-window (T14/2 with safety margin)."""
    b_imp = a * jnp.cos(inc) * (1.0 - e**2) / (1.0 + e * jnp.sin(w))
    ae = jnp.sqrt(1.0 - e**2) / (1.0 + e * jnp.sin(w))
    sin_arg = jnp.sqrt(jnp.maximum((1.0 + k_max)**2 - b_imp**2, 0.0)) / (a * jnp.sin(inc))
    return 0.025 + 0.5 * p / jnp.pi * jnp.arcsin(jnp.minimum(sin_arg, 1.0)) * ae


def _pd_t15(tc, t0, p, c):
    """Projected planet-star distance from Taylor coefficients."""
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


def rrmodel(times, k, t0, p, a, i, e, w,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, ldg, ldi, dldi,
            weights, dk, kmin, kmax, dg, ze,
            npb, nep, max_ns=1):
    """RoadRunner limb-darkened transit model (JAX) for a single parameter vector.

    Parameters
    ----------
    times : array (npt,)
        Observation times.
    k : array (npb,)
        Planet-to-star radius ratio per passband.
    t0 : array (ntc,)
        Mid-transit time per epoch.
    p, a, i, e, w : array (nep,)
        Orbital parameters per epoch.
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
    ldp : array (npb, nmu)
        Limb darkening profile per passband.
    ldg : array
        LD gradient (unused, kept for API compatibility).
    ldi : array (npb,)
        Disk-integrated intensity per passband.
    dldi : array
        Derivative of ldi (unused, kept for API compatibility).
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
        Radial zone edges (unused in JAX, kept for API compatibility).
    npb : int
        Number of passbands.
    nep : int
        Number of epochs.
    max_ns : int
        Maximum number of supersamples (static, for JAX tracing).

    Returns
    -------
    array (npt,)
        Model flux (1.0 out of transit).
    """
    nk = weights.shape[0]

    # Pre-compute Taylor coefficients for all epochs
    cfs = jax.vmap(
        lambda pp, aa, ii, ee, ww: solve_xy_p5(0.0, pp, aa, ii, ee, ww)
    )(p, a, i, e, w)

    # Pre-compute half-window per epoch
    k_max = jnp.max(k)
    half_windows = jax.vmap(
        lambda pp, aa, ii, ee, ww: _compute_half_window(k_max, pp, aa, ii, ee, ww)
    )(p, a, i, e, w)

    # Pre-compute LD means per passband by interpolating the weight table
    def _compute_ldm_pb(kv, ldp_pb):
        ik = jnp.clip(jnp.floor((kv - kmin) / dk).astype(int), 0, nk - 2)
        ak = jnp.clip((kv - kmin - ik * dk) / dk, 0.0, 1.0)
        return (1.0 - ak) * jnp.dot(weights[ik], ldp_pb) + ak * jnp.dot(weights[ik + 1], ldp_pb)

    ldm_all = jax.vmap(_compute_ldm_pb)(k, ldp)  # (npb, ng)

    # Gather per-time-point parameters using index arrays
    ilcs = lcids
    ipbs = pbids[ilcs]
    ieps = jnp.where(nep > 1, epids[ilcs], jnp.zeros_like(ilcs))

    k_pt = k[ipbs]
    t0_pt = t0[ieps]
    p_pt = p[ieps]
    cf_pt = cfs[ieps]
    hw_pt = half_windows[ieps]
    ns_pt = nsamples[ilcs]
    et_pt = exptimes[ilcs]
    ldm_pt = ldm_all[ipbs]    # (npt, ng)
    ldi_pt = ldi[ipbs]         # (npt,)

    def _single_time(tc, k_val, t0_val, p_val, cf, hw, ns, et, ldm, ldi_val):
        epoch = jnp.floor((tc - t0_val + 0.5 * p_val) / p_val)
        dt = tc - (t0_val + epoch * p_val)

        # Use static max_ns loop bound (required for JAX reverse-mode AD).
        # Samples beyond the actual ns are masked out.
        def body(isample, acc):
            offset = et * ((isample + 0.5) / ns - 0.5)
            z = _pd_t15(tc + offset, t0_val, p_val, cf)
            g = z / (1.0 + k_val)
            iplanet = _interpolate_ldm(g, dg, ldm)
            aplanet = ccia(1.0, k_val, z)
            sample_flux = (ldi_val - iplanet * aplanet) / ldi_val
            return acc + jnp.where(isample < ns, sample_flux, 0.0)

        flux = lax.fori_loop(0, max_ns, body, 0.0) / ns
        in_window = jnp.abs(dt) < hw
        return jnp.where(in_window, flux, 1.0)

    return jax.vmap(_single_time)(
        times, k_pt, t0_pt, p_pt, cf_pt, hw_pt,
        ns_pt, et_pt, ldm_pt, ldi_pt
    )
