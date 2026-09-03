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
from numpy import any as any_, int32, isnan, zeros

from ..numba.ldquad import rr_tables, rr_tables_grad

__all__ = ['solve_coefficients', 'solve_coefficients_d', 'rr_precompute', 'rr_precompute_d']


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


@njit
def _is_valid(a, e, ldp):
    """Per-parameter-vector validity, matching `backends.numba.rrmodel`.

    Note it inspects passband 0's limb-darkening profile only, and epoch 0's
    orbital parameters -- a quirk of the reference implementation.
    """
    return not (isnan(a[0]) or (a[0] <= 1.0) or (e[0] < 0.0) or any_(isnan(ldp[0])))


@njit
def rr_precompute(k, p, a, i, e, w, ldp, rules, pt0, pdt, ng, nor):
    """Host-side preamble for the OpenCL RoadRunner forward model.

    The mean intensity under the planet is tabulated against the grazing parameter with
    `ldquad.rr_tables`, the same function the numba kernel uses, so the two backends read
    identical tables.

    Returns
    -------
    gcs : ndarray
        Limb contacts, shape ``(npv, npb)``.
    coef : ndarray
        Split cubic tables, shape ``(npv, npb, ng - 2, 4)``.
    xyc : ndarray
        Taylor coefficients, shape ``(npv, nor, 2, 5)``.
    bt1, bt4 : ndarray
        Padded contact times, shape ``(npv,)``.
    valid : ndarray
        Per-parameter-vector validity flags, shape ``(npv,)``.
    """
    npv, npb = k.shape
    gcs = zeros((npv, npb))
    coef = zeros((npv, npb, ng - 2, 4))
    xyc = zeros((npv, nor, 2, 5))
    bt1 = zeros(npv)
    bt4 = zeros(npv)
    valid = zeros(npv, dtype=int32)
    for ipv in range(npv):
        if not _is_valid(a[ipv], e[ipv], ldp[ipv]):
            continue
        valid[ipv] = 1
        rr_tables(k[ipv], ldp[ipv], rules, pt0, pdt, ng, gcs[ipv], coef[ipv])
        for iep in range(nor):
            xyc[ipv, iep, :, :] = solve2d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep],
                                          e[ipv, iep], w[ipv, iep])
        t1, t4 = bounding_box(k[ipv, 0], xyc[ipv, 0])
        bt1[ipv] = t1 - 0.003
        bt4[ipv] = t4 + 0.003
    return gcs, coef, xyc, bt1, bt4, valid


@njit
def rr_precompute_d(k, p, a, i, e, w, ldp, ldg, rules, pt0, pdt, ng, nldc, nor):
    """Host-side preamble for the OpenCL RoadRunner gradient model.

    As `rr_precompute`, plus the tables the gradient needs (`ldquad.rr_tables_grad`) and
    the Taylor coefficient derivatives.

    Returns
    -------
    gcs, coef : ndarray
        Shapes ``(npv, npb)`` and ``(npv, npb, ng - 2, 4)``.
    kcoef : ndarray
        Total k derivative of the mean intensity along the node trajectory, shape
        ``(npv, npb, ng - 2, 4)``.
    dcoef : ndarray
        Limb darkening coefficient derivatives, shape ``(npv, npb, nldc, ng - 2, 4)``.
    xyc, dxyc : ndarray
        Shapes ``(npv, nor, 2, 5)`` and ``(npv, nor, 6, 2, 5)``.
    bt1, bt4, valid : ndarray
        Shape ``(npv,)``.
    """
    npv, npb = k.shape
    gcs = zeros((npv, npb))
    coef = zeros((npv, npb, ng - 2, 4))
    kcoef = zeros((npv, npb, ng - 2, 4))
    dcoef = zeros((npv, npb, nldc, ng - 2, 4))
    xyc = zeros((npv, nor, 2, 5))
    dxyc = zeros((npv, nor, 6, 2, 5))
    bt1 = zeros(npv)
    bt4 = zeros(npv)
    valid = zeros(npv, dtype=int32)
    for ipv in range(npv):
        if not _is_valid(a[ipv], e[ipv], ldp[ipv]):
            continue
        valid[ipv] = 1
        rr_tables_grad(k[ipv], ldp[ipv], ldg[ipv], rules, pt0, pdt, ng,
                       gcs[ipv], coef[ipv], kcoef[ipv], dcoef[ipv])
        for iep in range(nor):
            c, dc = solve2d_d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep],
                              e[ipv, iep], w[ipv, iep])
            xyc[ipv, iep, :, :] = c
            dxyc[ipv, iep, :, :, :] = dc[:6]
        t1, t4 = bounding_box(k[ipv, 0], xyc[ipv, 0])
        bt1[ipv] = t1 - 0.003
        bt4[ipv] = t4 + 0.003
    return gcs, coef, kcoef, dcoef, xyc, dxyc, bt1, bt4, valid
