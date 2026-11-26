#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Transit over an optically thin shell (Schlawin, Agol, Walkowicz, Covey & Lloyd, ApJL 722, L75, 2010).

The surface brightness of the shell is ``1 / sqrt(1 - r^2)``, which integrates to ``2 pi`` over the
disk. The blocked flux has a closed form in the complete elliptic integrals of all three kinds,
evaluated together by `ellkepi`.
"""

from numba import njit, prange
from numpy import pi, sqrt, arctan, abs, zeros, atleast_1d, ndarray, floor, inf, isnan, copysign, full, empty, int64

from meepmeep.backends.numba.point2d import solve2d, sep_c, bounding_box

from .ma_quadratic_nb import (ellke, ellkepi, supersampling_steps, bounding_box_padding, largest_radius_ratio,
                              parallel_block_layout, too_short_for_threads, infinite_flux)

TWO_PI = 2.0 * pi
FOUR_PI = 4.0 * pi
INV_FOUR_PI = 0.25 / pi

# The general formula stays accurate right up to its two singular points, so the windows around
# them only need to catch the distances where it divides by zero.
EDGE_TOL = 1e-12   #: |z - k| below which the occulter's edge is taken to pass through the disk centre
LIMB_TOL = 1e-12   #: |z - (1 - k)| below which the occulter is taken to touch the limb from the inside
HALF_TOL = 1e-10   #: |k - 0.5| below which an occulter with its edge at the disk centre also touches the limb


@njit(cache=False, fastmath=True)
def shell_flux(z: float, k: float) -> float:
    """The optically thin shell model for a non-negative distance.

    Parameters
    ----------
    z: float
        Normalized distance, z >= 0
    k: float
        Planet-star radius ratio

    Returns
    -------
    float
        Normalized flux: one for an unocculted source, zero for a completely occulted one.
    """
    # The source is unocculted
    if z >= 1.0 + k:
        return 1.0

    # The source is completely occulted
    if k >= 1.0 and z <= k - 1.0:
        return 0.0

    d = z - k
    zk = z * k

    # The edge of the occulting object lies at the origin
    if abs(d) < EDGE_TOL:
        if abs(k - 0.5) < HALF_TOL:
            b = TWO_PI - 4.0
        elif k < 0.5:
            Kk, Ek = ellke(2.0 * k)
            b = TWO_PI - 4.0 * Ek
        else:
            Kk, Ek = ellke(0.5 / k)
            b = TWO_PI - 8.0 * k * Ek + (8.0 * k - 2.0 / k) * Kk

    # The occulting object touches the limb of the source from the inside
    elif abs(z + k - 1.0) < LIMB_TOL:
        s = sqrt(zk)
        if d > 0.0:
            b = 4.0 * arctan(2.0 * s / d) - 8.0 * s
        else:
            b = FOUR_PI - 4.0 * arctan(-2.0 * s / d) - 8.0 * s

    else:
        zmk2 = d * d
        izmk2 = 1.0 / zmk2

        # The occulting object is fully inside the disk of the source
        if z + k < 1.0:
            rs = 1.0 / sqrt(1.0 - zmk2)
            Kk, Ek, Pk = ellkepi(4.0 * zk * izmk2, 2.0 * sqrt(zk) * rs)
            b = 4.0 * rs * ((zmk2 - 1.0) * Ek - (z + k) * d * Kk + (z + k) * d * izmk2 * Pk)

        # Ingress and egress
        else:
            rzk = 1.0 / sqrt(zk)
            Kk, Ek, Pk = ellkepi(izmk2 - 1.0, 0.5 * sqrt(1.0 - zmk2) * rzk)
            b = 2.0 * rzk * (-4.0 * zk * Ek + (2.0 * k * (z + k) - 1.0) * Kk + (z + k) * d * izmk2 * Pk)

        if k > z:
            b += FOUR_PI

    return 1.0 - b * INV_FOUR_PI


@njit(cache=False, fastmath=True)
def chromosphere_z_s(z: float, k: float) -> float:
    """Optically thin chromosphere model presented in
       Schlawin, Agol, Walkowicz, Covey & Lloyd (2010) for a scalar distance.

    A negative distance, including a negative zero, marks an unocculted source.
    """
    if copysign(1.0, z) < 0.0:
        return 1.0
    return shell_flux(z, k)


@njit(cache=False)
def chromosphere_z_v(zs: ndarray, k: float) -> ndarray:
    """Optically thin chromosphere model presented in
       Schlawin, Agol, Walkowicz, Covey & Lloyd (2010) for an array of distances."""
    flux = empty(zs.size)
    for i in range(zs.size):
        flux[i] = chromosphere_z_s(zs[i], k)
    return flux


# Chromosphere model for vector parameters
# ----------------------------------------
@njit(parallel=True, fastmath=False)
def chromosphere_model_v(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes):
    t0, p, a, i, e, w = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)

    npv = k.shape[0]
    nk = k.shape[1]
    npt = t.size

    ss = supersampling_steps(nsamples, exptimes)
    pad = bounding_box_padding(nsamples, exptimes)

    flux = zeros((npv, npt))
    for ipv in prange(npv):
        if (isnan(t0[ipv]) or isnan(p[ipv]) or isnan(a[ipv]) or isnan(i[ipv]) or isnan(e[ipv]) or isnan(w[ipv])
                or isnan(k[ipv]).any()):
            flux[ipv, :] = inf
            continue

        c = solve2d(0.0, p[ipv], a[ipv], i[ipv], e[ipv], w[ipv])
        bt1, bt4 = bounding_box(largest_radius_ratio(k[ipv]), c)
        bt1 -= pad
        bt4 += pad
        inv_p = 1.0 / p[ipv]

        for j in range(npt):
            epoch = floor((t[j] - t0[ipv] + 0.5 * p[ipv]) * inv_p)
            tc = t[j] - (t0[ipv] + epoch * p[ipv])

            if not (bt1 <= tc <= bt4):
                flux[ipv, j] = 1.0
            else:
                ilc = lcids[j]
                if nk == 1:
                    _k = k[ipv, 0]
                else:
                    _k = k[ipv, pbids[ilc]]

                if nsamples[ilc] == 1:
                    flux[ipv, j] = shell_flux(sep_c(tc, c), _k)
                else:
                    dt, t_first = ss[ilc, 0], ss[ilc, 1]
                    fsum = 0.0
                    for isample in range(nsamples[ilc]):
                        fsum += shell_flux(sep_c(tc + t_first + isample * dt, c), _k)
                    flux[ipv, j] = fsum * ss[ilc, 2]
    return flux


# Chromosphere model for scalar parameters
# ----------------------------------------
@njit(cache=False)
def too_short_for_threads_s(t, t0, p, bt1, bt4, lcids, nsamples) -> bool:
    """`too_short_for_threads` for a model with a single transit centre."""
    return too_short_for_threads(t, full(1, t0), p, bt1, bt4, lcids, zeros(nsamples.size, int64), nsamples)


def _create_chromosphere_model_s(parallel: bool, serial_model=None):
    def chromosphere_model_s(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes):
        """The chromosphere model for a single parameter vector.

        Compiled both in serial and in parallel; in the parallel version the time samples are
        distributed over the numba threads in interleaved blocks (see `parallel_block_layout`).
        """
        k = atleast_1d(k)
        npt = t.size

        # A NaN orbital parameter cannot be solved for an orbit.
        if isnan(t0) or isnan(p) or isnan(a) or isnan(i) or isnan(e) or isnan(w):
            return infinite_flux(npt)

        c = solve2d(0.0, p, a, i, e, w)
        pad = bounding_box_padding(nsamples, exptimes)
        bt1, bt4 = bounding_box(largest_radius_ratio(k), c)
        bt1 -= pad
        bt4 += pad

        if parallel and too_short_for_threads_s(t, t0, p, bt1, bt4, lcids, nsamples):
            return serial_model(t, k, t0, p, a, i, e, w, lcids, pbids, nsamples, exptimes)

        inv_p = 1.0 / p
        ss = supersampling_steps(nsamples, exptimes)

        flux = zeros(npt)
        block, nblocks, nthreads = parallel_block_layout(npt, parallel)
        for ithread in prange(nthreads):
            for iblock in range(ithread, nblocks, nthreads):
                for j in range(iblock * block, min((iblock + 1) * block, npt)):
                    epoch = floor((t[j] - t0 + 0.5 * p) * inv_p)
                    tc = t[j] - (t0 + epoch * p)

                    if not (bt1 <= tc <= bt4):
                        flux[j] = 1.0
                    else:
                        ilc = lcids[j]
                        if k.size == 1:
                            _k = k[0]
                        else:
                            _k = k[pbids[ilc]]

                        if isnan(_k):
                            flux[j] = inf
                        elif nsamples[ilc] == 1:
                            flux[j] = shell_flux(sep_c(tc, c), _k)
                        else:
                            dt, t_first = ss[ilc, 0], ss[ilc, 1]
                            fsum = 0.0
                            for isample in range(nsamples[ilc]):
                                fsum += shell_flux(sep_c(tc + t_first + isample * dt, c), _k)
                            flux[j] = fsum * ss[ilc, 2]
        return flux
    return njit(parallel=parallel, fastmath=False)(chromosphere_model_s)


chromosphere_model_s = _create_chromosphere_model_s(False)
chromosphere_model_s_parallel = _create_chromosphere_model_s(True, chromosphere_model_s)
