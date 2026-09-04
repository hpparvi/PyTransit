"""RoadRunner limb-darkened transit model for JAX.

JAX-compatible (jit + grad) RoadRunner transit model supporting heterogeneous
light curves with multiple passbands, epochs, and supersampling configurations.

The mean stellar intensity under the planet is integrated for every radius ratio with
the geometry-matched Gauss quadrature of `pytransit.backends.numba.ldquad`, written here
in vectorised, branchless form: the three integration regimes are all evaluated and the
right one selected with jnp.where, and every operation is kept finite for every input so
that the unselected regimes cannot poison the gradients. The result is tabulated against
the grazing parameter in two segments split at the limb contact and read with a
per-interval cubic, so jax.grad differentiates through the table build, including the
dependence of the nodes on the radius ratio.

No @jax.custom_jvp is needed — standard JAX tracing with jnp.where produces
correct gradients via jax.grad / jax.jacobian.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from meepmeep.backends.jax.ts2d import solve_xy_p5

from .ccintersection import ccia
from ..numba.ldquad import CUBIC_MATRICES

_TWO_PI = 2.0 * jnp.pi


def _safe_sqrt(x):
    """Square root with a finite gradient at (and below) zero."""
    return jnp.sqrt(jnp.maximum(x, 1e-30))


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


def _planet_angular_extent(z, b, k):
    """Angular extent of the planet disk at stellar radius z, finite for any input."""
    full = (z <= k - b) | ((b < 1e-12) & (z < k))
    outside = (z < b - k) | (z > b + k)
    partial = ~full & ~outside & (b >= 1e-12) & (z > 0.0)
    zs = jnp.where(partial, z, 1.0)
    bs = jnp.where(partial, b, 1.0)
    c = (zs * zs + bs * bs - k * k) / (2.0 * zs * bs)
    c = jnp.clip(jnp.where(partial, c, 0.0), -1.0 + 1e-12, 1.0 - 1e-12)
    return jnp.where(full, _TWO_PI, jnp.where(partial, 2.0 * jnp.arccos(c), 0.0))


def _ldm_nodes(k, gs, rules):
    """Quadrature nodes (as mu) and geometric factors for every g in `gs`, shapes (ng, 2 nq).

    The three regimes share the node columns: the first nq columns hold the single
    Gauss-Jacobi segment of the on-disk and limb-crossing regimes or the Gauss-Legendre
    part of the centre-covering regime, and the last nq columns the centre-covering
    regime's Gauss-Jacobi part (zero-weighted otherwise). The regime is selected on the
    inputs of the angular extent, so the transcendentals are evaluated once per column.
    """
    nq = rules.shape[2]
    tl, wl = rules[0, 0], rules[0, 1]
    tj, wj = rules[1, 0], rules[1, 1]
    b = (jnp.minimum(gs, 1.0 - 1e-9) * (1.0 + k))[:, None]
    on_disk = b + k <= 1.0
    centre = on_disk & (b < k)

    # Star centre inside the planet: Gauss-Legendre in mu over the fully covered centre ...
    mu_mid = _safe_sqrt(1.0 - (k - b) ** 2)
    c, h = 0.5 * (mu_mid + 1.0), 0.5 * (1.0 - mu_mid)
    m1 = c + h * tl
    z1 = _safe_sqrt(1.0 - m1 * m1)
    pref1 = wl * m1 * h
    # Planet fully on the disk: z = (b + k) - s**2 with Gauss-Jacobi.
    ch2 = 0.5 * jnp.sqrt(2.0 * k)
    s2 = ch2 + ch2 * tj
    z2 = b + k - s2 * s2
    m2 = _safe_sqrt(1.0 - z2 * z2)
    pref2 = wj * z2 * 2.0 * s2 * ch2
    # Planet crossing the limb: Gauss-Jacobi in mu.
    ch3 = 0.5 * _safe_sqrt(1.0 - (b - k) ** 2)
    m3 = ch3 + ch3 * tj
    z3 = _safe_sqrt(1.0 - m3 * m3)
    pref3 = wj * m3 * ch3

    z_a = jnp.where(on_disk, jnp.where(centre, z1, z2), z3)
    mu_a = jnp.where(on_disk, jnp.where(centre, m1, m2), m3)
    wf_a = jnp.where(on_disk, jnp.where(centre, pref1, pref2), pref3) * _planet_angular_extent(z_a, b, k)

    # ... then z = (k - b) + s**2 with Gauss-Jacobi for the rest of the planet.
    ch = 0.5 * _safe_sqrt(2.0 * b)
    s1 = ch + ch * tj
    z1b = k - b + s1 * s1
    z_b = jnp.where(centre, z1b, 1.0)
    mu_b = jnp.where(centre, _safe_sqrt(1.0 - z1b * z1b), 1.0)
    wf_b = jnp.where(centre, wj * z1b * 2.0 * s1 * ch * _planet_angular_extent(z_b, b, k), 0.0)

    return jnp.concatenate([mu_a, mu_b], axis=1), jnp.concatenate([wf_a, wf_b], axis=1)


def _profile_at(m, t0, dt, ldp):
    """Cubic interpolation of a profile tabulated on the grid ``mu = (t0 + dt * arange(n))**2``."""
    n = ldp.shape[0]
    x = jnp.clip((_safe_sqrt(m) - t0) / dt, 0.0, n - 1.0)
    i = jnp.clip(jnp.floor(x).astype(int) - 1, 0, n - 4)
    u = x - i
    w0 = -(u - 1.0) * (u - 2.0) * (u - 3.0) / 6.0
    w1 = u * (u - 2.0) * (u - 3.0) / 2.0
    w2 = -u * (u - 1.0) * (u - 3.0) / 2.0
    w3 = u * (u - 1.0) * (u - 2.0) / 6.0
    return w0 * ldp[i] + w1 * ldp[i + 1] + w2 * ldp[i + 2] + w3 * ldp[i + 3]


def _ldm_table(mu, wf, t0, dt, ldp):
    """The mean intensity under the planet at every g from the nodes and a tabulated profile."""
    # The barrier keeps XLA from fusing the profile gathers into the reduction, which
    # serialises them and costs several times the whole table build.
    prof = lax.optimization_barrier(_profile_at(mu, t0, dt, ldp))
    return jnp.sum(wf * prof, axis=1) / jnp.sum(wf, axis=1)


def _cubic_coefficients(tab):
    """Per-interval cubic coefficients for a table on a uniform grid, shape (n - 1, 4)."""
    n = tab.shape[0]
    i = np.arange(n - 1)
    j = np.clip(i - 1, 0, n - 4)
    vals = tab[j[:, None] + np.arange(4)]
    return jnp.einsum('irm,im->ir', jnp.asarray(CUBIC_MATRICES)[i - j], vals)


def _split_cubic_coefficients(tab, n1):
    """Cubic coefficients of a g table split at node ``n1 - 1``, shape (ng - 2, 4)."""
    return jnp.concatenate([_cubic_coefficients(tab[:n1]), _cubic_coefficients(tab[n1:])])


def _ldm_lookup(g, gc, n1, n2, coef, ipb):
    """The mean intensity under the planet at grazing parameter g from the split cubic table.

    `coef` is the flattened ``(npb, ng - 2, 4)`` coefficient table; four scalar gathers are
    considerably cheaper than one two-index gather on the CPU backend.
    """
    x1 = g / gc * (n1 - 1)
    i1 = jnp.clip(jnp.floor(x1).astype(int), 0, n1 - 2)
    x2 = (g - gc) / (1.0 - gc) * (n2 - 1)
    i2 = jnp.clip(jnp.floor(x2).astype(int), 0, n2 - 2)
    first = g < gc
    i = jnp.where(first, i1, i2 + n1 - 1)
    a = jnp.where(first, x1 - i1, x2 - i2)
    base = (ipb * (n1 + n2 - 2) + i) * 4
    return coef[base] + a * (coef[base + 1] + a * (coef[base + 2] + a * coef[base + 3]))


def rrmodel(times, k, t0, p, a, i, e, w,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, ldg, ldi, dldi,
            rules, pt0, pdt, ng, npb, nep, max_ns=1):
    """RoadRunner limb-darkened transit model (JAX) for a single parameter vector.

    Parameters
    ----------
    times : array (npt,)
        Observation times.
    k : array (npb,)
        Planet-to-star radius ratio per passband.
    t0 : array (ntc,)
        Transit centre per epoch.
    p, a, i, e, w : array (nep,)
        Orbital period, scaled semi-major axis, inclination [rad], eccentricity
        and argument of periastron [rad] per epoch.
    lcids : array (npt,)
        Light curve index for each time stamp.
    pbids : array (nlc,)
        Passband index for each light curve.
    epids : array (nlc,)
        Epoch index for each light curve.
    nsamples : array (nlc,)
        Number of supersamples per light curve.
    exptimes : array (nlc,)
        Exposure time per light curve.
    ldp : array (npb, nmu)
        Limb darkening profiles on the `ldquad.profile_grid` mu grid.
    ldg : unused
        Kept for signature compatibility with the numba kernels.
    ldi : array (npb,)
        Disk-integrated intensity per passband.
    dldi : unused
        Kept for signature compatibility with the numba kernels.
    rules : array (2, 2, nq)
        Quadrature rules from `ldquad.quadrature_rules`.
    pt0 : float
        First sqrt(mu) node of the profile grid.
    pdt : float
        The sqrt(mu) spacing of the profile grid.
    ng : int
        Number of grazing parameter nodes (static).
    npb : int
        Number of passbands (static).
    nep : int
        Number of epochs (static).
    max_ns : int
        Maximum number of supersamples (static, for JAX tracing).

    Returns
    -------
    array (npt,)
        Model flux (1.0 out of transit).
    """
    n1 = ng // 2
    n2 = ng - n1

    # Pre-compute Taylor coefficients for all epochs
    cfs = jax.vmap(
        lambda pp, aa, ii, ee, ww: solve_xy_p5(0.0, pp, aa, ii, ee, ww)
    )(p, a, i, e, w)

    # Pre-compute half-window per epoch
    k_max = jnp.max(k)
    half_windows = jax.vmap(
        lambda pp, aa, ii, ee, ww: _compute_half_window(k_max, pp, aa, ii, ee, ww)
    )(p, a, i, e, w)

    # Mean intensity under the planet as a split cubic table in g, per passband
    def _tables(kv, ldp_pb):
        gc = (1.0 - kv) / (1.0 + kv)
        gs = jnp.concatenate([jnp.linspace(0.0, gc, n1), jnp.linspace(gc, 1.0, n2)])
        mu, wf = _ldm_nodes(kv, gs, rules)
        tab = _ldm_table(mu, wf, pt0, pdt, ldp_pb)
        return gc, _split_cubic_coefficients(tab, n1)

    gcs, coef = jax.vmap(_tables)(k, ldp)  # (npb,), (npb, ng - 2, 4)
    coef = coef.reshape(-1)

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
    ldi_pt = ldi[ipbs]
    gc_pt = gcs[ipbs]

    def _single_time(tc, ipb, k_val, t0_val, p_val, cf, hw, ns, et, ldi_val, gc):
        epoch = jnp.floor((tc - t0_val + 0.5 * p_val) / p_val)
        dt = tc - (t0_val + epoch * p_val)

        # Use static max_ns loop bound (required for JAX reverse-mode AD).
        # Samples beyond the actual ns are masked out.
        def body(isample, acc):
            offset = et * ((isample + 0.5) / ns - 0.5)
            z = _pd_t15(tc + offset, t0_val, p_val, cf)
            iplanet = _ldm_lookup(z / (1.0 + k_val), gc, n1, n2, coef, ipb)
            aplanet = ccia(1.0, k_val, z)
            sample_flux = -(iplanet * aplanet) / ldi_val
            return acc + jnp.where(isample < ns, sample_flux, 0.0)

        flux = lax.fori_loop(0, max_ns, body, 0.0) / ns
        in_window = jnp.abs(dt) < hw
        return jnp.where(in_window, flux, 0.0)

    return jax.vmap(_single_time)(
        times, ipbs, k_pt, t0_pt, p_pt, cf_pt, hw_pt,
        ns_pt, et_pt, ldi_pt, gc_pt
    )
