from math import floor

from meepmeep.backends.numba.point2d import sep_c, solve2d, bounding_box
from numba import njit, prange
from numpy import zeros, dot, ndarray, isnan, full, nan

from .common import (calculate_weights_2d, interpolate_mean_limb_darkening_s,
                     interpolate_limb_darkening_s)
from .common import circle_circle_intersection_area_kite as ccia


def rr_simple(times: ndarray, k: float, t0: float, p: float, a: float, i: float, e: float, w: float,
              parallelize: bool, splimit: float, nsamples: int, exptimes: float, ldp: ndarray, istar: float,
              weights: ndarray, dk: float, kmin: float, kmax: float, dg: float, z_edges: ndarray, zm: ndarray):
    """Simplified RoadRunner model for a single homogeneous light curve.

    The per-orbit quantities are calculated first and the flux loop dispatches to a serial or a
    parallel version compiled from a shared implementation. For radius ratios at or below
    `splimit`, the model uses a small-planet approximation where the mean intensity blocked by
    the planet is approximated by the stellar intensity at the planet's center, and the limb
    darkening weighting is skipped altogether.

    Notes
    -----
    This dispatcher is deliberately a plain Python function: compiling it with numba makes the
    flux kernels it calls run measurably (up to three times) slower when invoked from
    ``RoadRunnerModel.evaluate``, for reasons that could not be pinned down (the effect persists
    with identical argument types and values, and disappears when the dispatch happens in
    Python). Keep it in Python unless the measurements are redone.
    """
    bad, small_planet, ldm, xyc, bt1, bt4 = rr_simple_precompute(k, p, a, i, e, w, splimit, ldp,
                                                                 weights, dk, kmin, kmax, z_edges)
    if bad:
        return full(times.size, nan)

    bt1 -= 0.003 + exptimes
    bt4 += 0.003 + exptimes

    if parallelize:
        return rr_simple_flux_parallel(times, k, t0, p, xyc, bt1, bt4, ldm, ldp, zm, small_planet,
                                       nsamples, exptimes, istar, dg)
    else:
        return rr_simple_flux_serial(times, k, t0, p, xyc, bt1, bt4, ldm, ldp, zm, small_planet,
                                     nsamples, exptimes, istar, dg)


@njit
def rr_simple_precompute(k: float, p: float, a: float, i: float, e: float, w: float, splimit: float,
                         ldp: ndarray, weights: ndarray, dk: float, kmin: float, kmax: float,
                         z_edges: ndarray):
    """Precompute the per-orbit quantities needed by the flux stage.

    Calculates the mean limb darkening profile (skipped in the small-planet mode), the Taylor
    series expansion coefficients for the planet position, and the transit bounding box.
    """
    ng = weights.shape[1]
    bad = isnan(a) or (a <= 1.0) or (e < 0.0) or isnan(ldp[0])

    # ----------------------------------#
    # Calculate the limb darkening mean #
    # ----------------------------------#
    small_planet = k <= splimit
    ldm = zeros(ng)
    if not bad and not small_planet:
        if kmin <= k <= kmax:
            ik = int(floor((k - kmin) / dk))
            ak = (k - kmin - ik * dk) / dk
            ldm[:] = (1.0 - ak) * dot(weights[ik], ldp) + ak * dot(weights[ik + 1], ldp)
        else:
            _, _, wg = calculate_weights_2d(k, z_edges, ng)
            ldm[:] = dot(wg, ldp)

    # ------------------------------------------------------------------------------#
    # Calculate the Taylor series expansions for the orbit and the transit bounding #
    # box (the box is padded by the caller)                                         #
    # ------------------------------------------------------------------------------#
    xyc = solve2d(0.0, p, a, i, e, w)
    bt1, bt4 = bounding_box(k, xyc)
    return bad, small_planet, ldm, xyc, bt1, bt4


def _rr_simple_flux(times: ndarray, k: float, t0: float, p: float, xyc: ndarray,
                    bt1: float, bt4: float, ldm: ndarray, ldp: ndarray, zm: ndarray,
                    small_planet: bool, nsamples: int, exptimes: float, istar: float, dg: float):
    """Calculate the model fluxes for all time samples.

    Compiled both in serial and in parallel; in the parallel version the loop over the time
    samples is distributed over the numba threads.
    """
    # The input arrays are copied locally because numba generates measurably faster code for
    # the flux loop with locally allocated arrays than with array arguments.
    xyc = xyc.copy()
    ldm = ldm.copy()
    ldp = ldp.copy()
    zm = zm.copy()

    npt = times.size
    flux = zeros(npt)
    for ipt in prange(npt):
        epoch = floor((times[ipt] - t0 + 0.5 * p) / p)
        tc = times[ipt] - (t0 + epoch * p)
        if not (bt1 <= tc <= bt4):
            flux[ipt] = 1.0
        else:
            fsum = 0.0
            for isample in range(1, nsamples + 1):
                time_offset = exptimes * ((isample - 0.5) / nsamples - 0.5)
                z = sep_c(tc + time_offset, xyc)
                if small_planet:
                    iplanet = interpolate_limb_darkening_s(z, zm, ldp)
                else:
                    iplanet = interpolate_mean_limb_darkening_s(z / (1.0 + k), dg, ldm)
                aplanet = ccia(1.0, k, z)[0]
                fsum += (istar - iplanet * aplanet) / istar
            flux[ipt] = fsum / nsamples
    return flux


rr_simple_flux_serial = njit(parallel=False)(_rr_simple_flux)
rr_simple_flux_parallel = njit(parallel=True)(_rr_simple_flux)
