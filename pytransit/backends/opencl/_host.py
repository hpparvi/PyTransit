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
from numpy import any as any_, ascontiguousarray, floor, dot, int32, isnan, zeros

from ..numba.rrmodel import calculate_weights_2d

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
def _passband_weights(kv, weights, dk, kmin, kmax, z_edges, ng):
    """The (ng, nmu) weight matrix for one radius ratio.

    Interpolates the tabulated weights in k, or rebuilds them from scratch when
    k falls outside the tabulated range -- the same fallback the Numba reference
    uses. The fallback costs ~ng*nmu circle-circle intersections, so it is
    computed once per radius ratio rather than once per limb-darkening
    coefficient.
    """
    if kmin <= kv <= kmax:
        ik = int(floor((kv - kmin) / dk))
        ak = (kv - kmin - ik * dk) / dk
        return (1.0 - ak) * weights[ik] + ak * weights[ik + 1]
    _, _, wg = calculate_weights_2d(kv, z_edges, ng)
    return wg


@njit
def _ld_means(k, ldp, weights, dk, kmin, kmax, z_edges, ng):
    """Contract the weight table with a limb-darkening profile table.

    Mirrors the per-passband preamble of `backends.numba.rrmodel`, including the
    `calculate_weights_2d` fallback when the radius ratio falls outside the
    tabulated range. Keeping this on the host means the 8 MB weight table never
    reaches the device and the device code stays limb-darkening-agnostic.

    Parameters
    ----------
    k : ndarray
        Radius ratios for one parameter vector, shape ``(npb,)``.
    ldp : ndarray
        Limb-darkening profiles for one parameter vector, shape ``(npb, nmu)``.

    Returns
    -------
    ldm : ndarray
        Mean intensity under the planet disk versus the grazing parameter,
        shape ``(npb, ng)``.
    """
    npb = k.size
    ldm = zeros((npb, ng))
    for ipb in range(npb):
        wg = _passband_weights(k[ipb], weights, dk, kmin, kmax, z_edges, ng)
        ldm[ipb, :] = dot(wg, ldp[ipb])
    return ldm


@njit
def _ld_means_and_derivatives(k, ldp, ldg, weights, dk, kmin, kmax, z_edges, ng, nldc):
    """`_ld_means` plus the limb-darkening coefficient derivatives.

    Shares one weight matrix per passband across the profile and all its
    coefficient derivatives; `ldg` row 0 is dI/dmu, which the transit model does
    not use, so rows ``1..nldc`` are the ones contracted.
    """
    npb = k.size
    ldm = zeros((npb, ng))
    dldm = zeros((npb, nldc, ng))
    for ipb in range(npb):
        wg = _passband_weights(k[ipb], weights, dk, kmin, kmax, z_edges, ng)
        ldm[ipb, :] = dot(wg, ldp[ipb])
        for j in range(nldc):
            # `ldg[ipb, j+1]` is a strided view; copy it so `dot` runs contiguous.
            dldm[ipb, j, :] = dot(wg, ascontiguousarray(ldg[ipb, j + 1]))
    return ldm, dldm


@njit
def _is_valid(a, e, ldp):
    """Per-parameter-vector validity, matching `backends.numba.rrmodel`.

    Note it inspects passband 0's limb-darkening profile only, and epoch 0's
    orbital parameters -- a quirk of the reference implementation.
    """
    return not (isnan(a[0]) or (a[0] <= 1.0) or (e[0] < 0.0) or any_(isnan(ldp[0])))


@njit
def rr_precompute(k, p, a, i, e, w, ldp, weights, dk, kmin, kmax, z_edges, ng, nor):
    """Host-side preamble for the OpenCL RoadRunner forward model.

    Returns
    -------
    ldm : ndarray
        Mean limb-darkening tables, shape ``(npv, npb, ng)``.
    xyc : ndarray
        Taylor coefficients, shape ``(npv, nor, 2, 5)``.
    bt1, bt4 : ndarray
        Padded contact times, shape ``(npv,)``.
    valid : ndarray
        Per-parameter-vector validity flags, shape ``(npv,)``.
    """
    npv, npb = k.shape
    ldm = zeros((npv, npb, ng))
    xyc = zeros((npv, nor, 2, 5))
    bt1 = zeros(npv)
    bt4 = zeros(npv)
    valid = zeros(npv, dtype=int32)

    for ipv in range(npv):
        if not _is_valid(a[ipv], e[ipv], ldp[ipv]):
            continue
        valid[ipv] = 1
        ldm[ipv] = _ld_means(k[ipv], ldp[ipv], weights, dk, kmin, kmax, z_edges, ng)
        for iep in range(nor):
            xyc[ipv, iep, :, :] = solve2d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep],
                                          e[ipv, iep], w[ipv, iep])
        t1, t4 = bounding_box(k[ipv, 0], xyc[ipv, 0])
        bt1[ipv] = t1 - 0.003
        bt4[ipv] = t4 + 0.003
    return ldm, xyc, bt1, bt4, valid


@njit
def rr_precompute_d(k, p, a, i, e, w, ldp, ldg, weights, dk, kmin, kmax, z_edges, ng, nldc, nor):
    """Host-side preamble for the OpenCL RoadRunner gradient model.

    As `rr_precompute`, plus the limb-darkening coefficient derivatives of the
    mean profile and the Taylor coefficient derivatives. `ldg` row 0 is `dI/dmu`,
    which the transit model does not use, so rows ``1..nldc`` are the ones
    contracted here.

    Returns
    -------
    ldm : ndarray
        Shape ``(npv, npb, ng)``.
    dldm : ndarray
        Shape ``(npv, npb, nldc, ng)``.
    xyc, dxyc : ndarray
        Shapes ``(npv, nor, 2, 5)`` and ``(npv, nor, 6, 2, 5)``.
    bt1, bt4, valid : ndarray
        Shape ``(npv,)``.
    """
    npv, npb = k.shape
    ldm = zeros((npv, npb, ng))
    dldm = zeros((npv, npb, nldc, ng))
    xyc = zeros((npv, nor, 2, 5))
    dxyc = zeros((npv, nor, 6, 2, 5))
    bt1 = zeros(npv)
    bt4 = zeros(npv)
    valid = zeros(npv, dtype=int32)

    for ipv in range(npv):
        if not _is_valid(a[ipv], e[ipv], ldp[ipv]):
            continue
        valid[ipv] = 1
        ldm[ipv], dldm[ipv] = _ld_means_and_derivatives(
            k[ipv], ldp[ipv], ldg[ipv], weights, dk, kmin, kmax, z_edges, ng, nldc)
        for iep in range(nor):
            c, dc = solve2d_d(0.0, p[ipv, iep], a[ipv, iep], i[ipv, iep],
                              e[ipv, iep], w[ipv, iep])
            xyc[ipv, iep, :, :] = c
            dxyc[ipv, iep, :, :, :] = dc[:6]
        t1, t4 = bounding_box(k[ipv, 0], xyc[ipv, 0])
        bt1[ipv] = t1 - 0.003
        bt4[ipv] = t4 + 0.003
    return ldm, dldm, xyc, dxyc, bt1, bt4, valid
