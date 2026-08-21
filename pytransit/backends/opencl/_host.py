#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
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

"""Host-side orbit solving for the OpenCL backend.

The Taylor coefficients are solved on the CPU with MeepMeep and uploaded to the
device, so Kepler's equation and the contact-point bisection never cross the
device boundary and `bounding_box` stays bit-identical to the Numba backend.
The work is tiny (once per parameter vector and epoch, versus once per data
point on the device) but it is jitted rather than looped in Python: for a large
parameter-vector population a Python loop over `solve2d` would dominate the
evaluation and negate the GPU entirely.
"""

from meepmeep.numba2d import bounding_box, solve2d, solve2d_d
from numba import njit
from numpy import zeros

__all__ = ['solve_coefficients', 'solve_coefficients_d']


@njit
def solve_coefficients(k, p, a, i, e, w, nor):
    """Solve the Taylor coefficients and transit window for every parameter vector.

    Mirrors the per-parameter-vector preamble of `backends.numba.udmodel`,
    including its use of passband 0's radius ratio and epoch 0's coefficients
    for the bounding box, and the fixed 0.003 d padding.

    Returns
    -------
    xyc : ndarray
        Taylor coefficients, shape ``(npv, nor, 2, 5)``.
    bt1, bt4 : ndarray
        Padded first and fourth contact times, shape ``(npv,)``.
    """
    npv = k.shape[0]
    xyc = zeros((npv, nor, 2, 5))
    bt1 = zeros(npv)
    bt4 = zeros(npv)

    for ipv in range(npv):
        for iep in range(nor):
            xyc[ipv, iep, :, :] = solve2d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep],
                                          e[ipv, iep], w[ipv, iep])
        t1, t4 = bounding_box(k[ipv, 0], xyc[ipv, 0])
        bt1[ipv] = t1 - 0.003
        bt4[ipv] = t4 + 0.003
    return xyc, bt1, bt4


@njit
def solve_coefficients_d(k, p, a, i, e, w, nor):
    """Solve the Taylor coefficients, their derivatives, and the transit window.

    As `solve_coefficients`, but also returns the coefficient derivatives.
    MeepMeep returns a ``(7, 2, 5)`` tensor ordered ``(tc, p, a, i, e, w, lan)``;
    the transit model ignores `lan`, so only the first six rows are kept and the
    device never sees the seventh.

    Returns
    -------
    xyc : ndarray
        Taylor coefficients, shape ``(npv, nor, 2, 5)``.
    dxyc : ndarray
        Coefficient derivatives, shape ``(npv, nor, 6, 2, 5)``.
    bt1, bt4 : ndarray
        Padded first and fourth contact times, shape ``(npv,)``.
    """
    npv = k.shape[0]
    xyc = zeros((npv, nor, 2, 5))
    dxyc = zeros((npv, nor, 6, 2, 5))
    bt1 = zeros(npv)
    bt4 = zeros(npv)

    for ipv in range(npv):
        for iep in range(nor):
            c, dc = solve2d_d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep],
                              e[ipv, iep], w[ipv, iep])
            xyc[ipv, iep, :, :] = c
            dxyc[ipv, iep, :, :, :] = dc[:6]
        t1, t4 = bounding_box(k[ipv, 0], xyc[ipv, 0])
        bt1[ipv] = t1 - 0.003
        bt4[ipv] = t4 + 0.003
    return xyc, dxyc, bt1, bt4
