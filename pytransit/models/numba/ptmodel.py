#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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
from numba import njit
from numpy import arccos, sqrt, linspace, zeros, arange, dot, floor, pi, ndarray

from pytransit.models.numba.ma_uniform_nb import uniform_z_v


@njit
def cc_area(r, k, b):
    """Area of the intersection of two circles.
    """
    if b <= k - r:
        return pi * r ** 2
    elif b <= r - k:
        return pi * k ** 2
    elif b >= r + k:
        return 0.0
    else:
        return (k ** 2 * arccos((b ** 2 + k ** 2 - r ** 2) / (2 * b * k)) +
                r ** 2 * arccos((b ** 2 + r ** 2 - k ** 2) / (2 * b * r)) -
                0.5 * sqrt((-b + k + r) * (b + k - r) * (b - k + r) * (b + k + r)))


@njit
def create_z_grid(zcut=0.7, nin=15, nedge=15):
    mucut = sqrt(1.0 - zcut ** 2)
    dz = zcut / nin
    dmu = mucut / nedge

    z_edges = zeros(nin + nedge)
    z_means = zeros(nin + nedge)

    for i in range(nin - 1):
        z_edges[i] = (i + 1) * dz

    for i in range(nedge + 1):
        z_edges[-i - 1] = sqrt(1 - (i * dmu) ** 2)

    for i in range(nin + nedge - 1):
        z_means[i + 1] = 0.5 * (z_edges[i] + z_edges[i + 1])

    return z_edges, z_means

@njit
def calculate_weights(k: float, ze: ndarray, ng: int = 50):
    """Calculate a 2D limb darkening weight array.

    Parameters
    ----------
    k: float
        Radius ratio
    ng: int
        Grazing parameter resolution
    nmu: int
        Mu resolution

    Returns
    -------

    """
    gs = linspace(0, 1 - 1e-5, ng)
    nz = ze.size
    weights = zeros((ng, nz))

    for ig in range(ng):
        b = gs[ig] * (1.0 + k)
        a0 = cc_area(ze[0], k, b)
        weights[ig, 0] = a0
        s = weights[ig, 0]
        for i in range(1, nz):
            a1 = cc_area(ze[i], k, b)
            weights[ig, i] = a1 - a0
            a0 = a1
            s += weights[ig, i]
        weights[ig] /= s
    return gs, gs[1] - gs[0], weights


@njit
def im_p_s(g: float, dg, weights, ldp):
    """The average limb darkening intensity blocked by the planet.

    Parameters
    ----------
    g: float
        Grazing parameter
    dg: float
        Grazing parameter array step size
    weights: ndarray
        Limb darkening weight array
    ldp: ndarray
        Limb darkening values

    Returns
    -------

    """
    ng = g / dg
    ig = int(floor(ng))
    ag = ng - ig
    return (1.0 - ag) * dot(weights[ig], ldp) + ag * dot(weights[ig + 1], ldp)

@njit
def im_p_v(gs, dg, weights, ldp):
    """The average limb darkening intensity blocked by the planet.

    Parameters
    ----------
    g: float
        Grazing parameter
    dg: float
        Grazing parameter array step size
    weights: ndarray
        Limb darkening weight array
    ldp: ndarray
        Limb darkening values

    Returns
    -------

    """
    ldw = dot(weights, ldp)
    im = zeros(gs.size)
    for i in range(gs.size):
        ng = gs[i] / dg
        ig = int(floor(ng))
        ag = ng - ig
        im[i] = (1.0 - ag) * ldw[ig] + ag * ldw[ig + 1]
    return im

@njit
def ptmodel_z_v(z, k, ldi, dg, istar, weights):
    iplanet = im_p_v(z / (1. + k), dg, weights, ldi)
    aplanet = 1. - uniform_z_v(z, k)
    return (istar - iplanet * aplanet) / istar


@njit
def ptmodel_z_v_full(z, k, ldl, ldc, istar, ng=50, nzin=20, nzedge=20):
    ze, zm = create_z_grid(nin=nzin, nedge=nzedge)
    gs, dg, weights = calculate_weights(k, ze, ng)
    mus = sqrt(1. - zm**2)
    ldp = ldl(mus, ldc)
    iplanet = im_p_v(z / (1. + k), dg, weights, ldp)
    aplanet = 1.0 - uniform_z_v(z, k)
    return (istar - iplanet * aplanet) / istar