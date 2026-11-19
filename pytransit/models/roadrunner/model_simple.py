from math import floor

from meepmeep.backends.numba.point2d import pos_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, ndarray, isnan, full, nan, empty, sqrt, pi

from .common import (g_nodes, table_sizes, ldm_nodes, ldm_table, split_cubic_coefficients, split_point, ldm_lookup,
                     ldm_lookup_full,
                     profile_at)
from .common import circle_circle_intersection_area_kite as ccia


def rr_simple(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
              parallelize: bool, splimit: float, nsamples: int, exptimes: float, ldp: ndarray,
              pt0: float, pdt: float, istar: float, rules: ndarray, ng: int):
    """Simplified RoadRunner model for a single homogeneous light curve.

    The per-orbit quantities are calculated first and the flux loop dispatches to a serial or a
    parallel version compiled from a shared implementation. For radius ratios at or below
    `splimit`, the model uses a small-planet approximation where the mean intensity blocked by
    the planet is approximated by the stellar intensity at the planet's center, and the mean
    intensity table is skipped altogether.

    Notes
    -----
    This dispatcher is deliberately a plain Python function: compiling it with numba makes the
    flux kernels it calls run measurably (up to three times) slower when invoked from
    ``RoadRunnerModel.evaluate``, for reasons that could not be pinned down (the effect persists
    with identical argument types and values, and disappears when the dispatch happens in
    Python). Keep it in Python unless the measurements are redone.
    """
    bad, small_planet, gc, n1, coef, xyc, bt1, bt4 = rr_simple_precompute(k, p, a, i, e, w, splimit, ldp,
                                                                          pt0, pdt, rules, ng)
    if bad:
        return full(times.size, nan)

    bt1 -= 0.003 + exptimes
    bt4 += 0.003 + exptimes

    if parallelize:
        return rr_simple_flux_parallel(times, k, t0, p, xyc, bt1, bt4, gc, n1, coef, ldp, pt0, pdt,
                                       small_planet, nsamples, exptimes, istar)
    else:
        return rr_simple_flux_serial(times, k, t0, p, xyc, bt1, bt4, gc, n1, coef, ldp, pt0, pdt,
                                     small_planet, nsamples, exptimes, istar)


@njit
def rr_simple_precompute(k: float, p: float, a: float, i: float, e: float, w: float, splimit: float,
                         ldp: ndarray, pt0: float, pdt: float, rules: ndarray, ng: int):
    """Precompute the per-orbit quantities needed by the flux stage.

    Builds the mean intensity table under the planet as a function of the grazing parameter
    (skipped in the small-planet mode) and its cubic coefficients, the Taylor series
    expansion coefficients for the planet position, and the transit bounding box.
    """
    bad = isnan(a) or (a <= 1.0) or (e < 0.0) or isnan(ldp[0]) or not (0.0 < k <= 1.0)
    small_planet = k <= splimit
    nq = rules.shape[2]
    n1 = table_sizes(ng)[0]
    coef = zeros((ng - 3, 4))

    # Nothing below is defined for a bad parameter vector; the caller returns NaN fluxes.
    if bad:
        return bad, small_planet, 0.0, n1, coef, zeros((2, 5)), 0.0, 0.0

    gc = split_point(k)

    # -------------------------------------------#
    # Mean intensity under the planet, ldm(g)    #
    # -------------------------------------------#
    if not bad and not small_planet:
        gs, n1 = g_nodes(k, ng)
        mu = empty((ng, 2 * nq))
        wf = zeros((ng, 2 * nq))
        ldm_nodes(k, gs, rules, mu, wf)
        ldm = empty(ng)
        ldm_table(mu, wf, pt0, pdt, ldp, ldm)
        split_cubic_coefficients(ldm, n1, coef)

    # ------------------------------------------------------------------------------#
    # Calculate the Taylor series expansions for the orbit and the transit bounding #
    # box (the box is padded by the caller)                                         #
    # ------------------------------------------------------------------------------#
    xyc = solve2d(0.0, p, a, i, e, w)
    bt1, bt4 = bounding_box(k, xyc)
    return bad, small_planet, gc, n1, coef, xyc, bt1, bt4


def _rr_simple_flux(times: ndarray, k: float, t0: float, p: float, xyc: ndarray,
                    bt1: float, bt4: float, gc: float, n1: int, coef: ndarray, ldp: ndarray,
                    pt0: float, pdt: float, small_planet: bool, nsamples: int, exptimes: float, istar: float):
    """Calculate the model fluxes for all time samples.

    Compiled both in serial and in parallel; in the parallel version the loop over the time
    samples is distributed over the numba threads.
    """
    # The input arrays are copied locally because numba generates measurably faster code for
    # the flux loop with locally allocated arrays than with array arguments.
    xyc = xyc.copy()
    coef = coef.copy()
    ldp = ldp.copy()

    # The divisions are hoisted out of the loops as reciprocals.
    inv_p = 1.0 / p
    inv_k1 = 1.0 / (1.0 + k)
    scale = 1.0 / (istar * nsamples)
    dt = exptimes / nsamples
    t_first = 0.5 * (dt - exptimes)      # Offset of the first supersample from the exposure centre

    # A planet fully on the disk blocks its whole area, and its mean intensity table is laid out
    # in a coordinate that follows from the squared projected distance, so those samples need
    # neither the distance itself nor the overlap area calculation.
    z2c = (1.0 - k) ** 2
    inv_z2c = 1.0 / z2c if z2c > 0.0 else 0.0
    area_full = pi * k * k

    npt = times.size
    flux = zeros(npt)
    for ipt in prange(npt):
        epoch = floor((times[ipt] - t0) * inv_p + 0.5)
        tc = times[ipt] - (t0 + epoch * p)
        if not (bt1 <= tc <= bt4):
            flux[ipt] = 1.0
        else:
            fsum = 0.0
            for isample in range(nsamples):
                px, py = pos_c(tc + t_first + isample * dt, xyc)
                z2 = px * px + py * py
                if small_planet:
                    fsum += profile_at(sqrt(max(0.0, 1.0 - z2)), pt0, pdt, ldp) * ccia(1.0, k, sqrt(z2))[0]
                elif z2 < z2c:
                    fsum += ldm_lookup_full(z2, inv_z2c, n1, coef) * area_full
                else:
                    z = sqrt(z2)
                    fsum += ldm_lookup(z * inv_k1, gc, n1, coef) * ccia(1.0, k, z)[0]
            flux[ipt] = 1.0 - fsum * scale
    return flux


rr_simple_flux_serial = njit(parallel=False)(_rr_simple_flux)
rr_simple_flux_parallel = njit(parallel=True)(_rr_simple_flux)
