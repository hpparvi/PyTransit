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
from numba import njit, prange
from numpy import any, all, sum, ndarray, zeros, array, exp, log, atleast_2d, ones, arange, atleast_1d, isnan, inf, unique

from ...orbits.orbits_py import z_ip_s, z_ip_v


MODE_NORMAL = 0
MODE_TRSPEC = 1

# Gamma and log gamma
# -------------------

lanczos_g = 6.024680040776729583740234375
lanczos_g_minus_half = 5.524680040776729583740234375

lanczos_num_coeffs = array([
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408])

lanczos_den_coeffs = array([
    0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0,
    13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0])

lanczos_n = lanczos_num_coeffs.size

gamma_integral = array([
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
    1307674368000.0, 20922789888000.0, 355687428096000.0,
    6402373705728000.0, 121645100408832000.0, 2432902008176640000.0,
    51090942171709440000.0, 1124000727777607680000.0])


@njit
def lanczos_sum(x):
    num, den = 0.0, 0.0
    if x < 0.0:
        raise ValueError
    if (x < 5.0):
        for i in range(lanczos_n - 1, -1, -1):
            num = num * x + lanczos_num_coeffs[i]
            den = den * x + lanczos_den_coeffs[i]
    else:
        for i in range(lanczos_n):
            num = num / x + lanczos_num_coeffs[i]
            den = den / x + lanczos_den_coeffs[i]
    return num / den


@njit
def loggamma(x):
    """Unsafe Log Gamma function for positive x.

    Notes
    -----
    The code is modified from the implementation in the python
    math library, but with all the safety checking removed.

    """
    absx = abs(x)
    if (absx < 1e-20):
        return -log(absx)

    r = log(lanczos_sum(absx)) - lanczos_g
    r += (absx - 0.5) * (log(absx + lanczos_g - 0.5) - 1)
    return r


@njit
def gamma(x):
    """Unsafe Gamma function for positive x.

    Notes
    -----
    The code is modified from the implementation in the python
    math library, but with all the safety checking removed.
    """
    return exp(loggamma(x))


def jcoeff(alpha, beta, ri):
    res = zeros(4)
    res[0] = 2. * ri * (ri + alpha + beta) * (2. * ri - 2. + alpha + beta)
    res[1] = (2. * ri - 1. + alpha + beta) * (2. * ri + alpha + beta) * (2. * ri - 2. + alpha + beta)
    res[2] = (2. * ri - 1. + alpha + beta) * (alpha + beta) * (alpha - beta)
    res[3] = - 2. * (ri - 1. + alpha) * (ri - 1. + beta) * (2. * ri + alpha + beta)
    return res


def init_arrays(npol, nldc):
    anm, avl = zeros((npol, nldc + 1)), zeros((npol, nldc + 1))
    ajd, aje = zeros((4, npol, nldc + 1)), zeros((4, npol, nldc + 1))

    for j in range(nldc + 1):
        nu = (j + 2.0) / 2.0
        for i in range(npol):
            nm1 = exp(loggamma(nu + i + 1.0) - loggamma(i + 2.0))
            avl[i, j] = (-1) ** (i) * (2.0 + 2.0 * i + nu) * nm1
            anm[i, j] = exp(loggamma(i + 1.0) + loggamma(nu + 1.0) - loggamma(i + 1.0 + nu))

    for j in range(nldc + 1):
        nu = (2.0 + j) / 2.0
        for i in range(npol):
            ajd[:, i, j] = jcoeff(0.0, 1.0 + nu, i + 1)
            aje[:, i, j] = jcoeff(nu, 1.0, i + 1)

    return anm, avl, ajd, aje


@njit
def jacobi(npol, alpha, beta, x, i_ld, j_c):
    """Jacobi polynomial

    Notes
    -----

    Adapted from the Jacobi polynomials routine by J. Burkardt. The
    only major difference is that the routine computes the values for
    multiple x at the same time.
    """
    j = i_ld
    cx = zeros((x.size, npol))
    cx[:, 0] = 1.0
    cx[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)
    for i in range(1, npol - 1):
        for k in range(x.size):
            cx[k, i + 1] = ((j_c[2, i, j] + j_c[1, i, j] * x[k]) * cx[k, i] + j_c[3, i, j] * cx[k, i - 1]) / j_c[
                0, i, j]
    return cx


@njit
def jacobi_s(npol, alpha, beta, x, i_ld, j_c):
    """Jacobi polynomial

    Notes
    -----

    Adapted from the Jacobi polynomials routine by J. Burkardt. The
    only major difference is that the routine computes the values for
    multiple x at the same time.
    """
    j = i_ld
    cx = zeros(npol)
    cx[0] = 1.0
    cx[1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)
    for i in range(1, npol - 1):
        cx[i + 1] = ((j_c[2, i, j] + j_c[1, i, j] * x) * cx[i] + j_c[3, i, j] * cx[i - 1]) / j_c[0, i, j]
    return cx


@njit
def alpha(b: float, c: ndarray, n: int, npol: int, nldc: int, anm: ndarray, avl: ndarray, ajd: ndarray, aje: ndarray):
    sm = zeros(c.size)
    nu = (2.0 + n) / 2.0
    norm = b * b * (1.0 - c * c) ** (1.0 + nu) / (nu * gamma(1.0 + nu))

    e = jacobi_s(npol, nu, 1.0, 1.0 - 2 * (1.0 - b), n, aje)
    for i in range(npol):
        e[i] = (e[i] * anm[i, n]) ** 2

    d = jacobi(npol, 0.0, 1.0 + nu, 1.0 - 2 * c ** 2, n, ajd)
    for i in range(npol):
        sm += avl[i, n] * d[:, i] * e[i]

    a = norm * sm
    return a


@njit
def alpha_s(b: float, c: float, n: int, npol: int, nldc: int, anm: ndarray, avl: ndarray, ajd: ndarray, aje: ndarray):
    sm = 0.0
    nu = (2.0 + n) / 2.0
    norm = b * b * (1.0 - c * c) ** (1.0 + nu) / (nu * gamma(1.0 + nu))

    e = jacobi_s(npol, nu, 1.0, 1.0 - 2 * (1.0 - b), n, aje)
    for i in range(npol):
        e[i] = (e[i] * anm[i, n]) ** 2

    d = jacobi_s(npol, 0.0, 1.0 + nu, 1.0 - 2 * c ** 2, n, ajd)
    for i in range(npol):
        sm += avl[i, n] * d[i] * e[i]

    a = norm * sm
    return a


@njit
def general_model_vz(z: ndarray, k: float, u, npol, nldc, npb, anm, avl, ajd, aje):
    u = atleast_2d(u)
    assert u.shape == (npb, nldc)
    b = k / (1.0 + k)
    c = z / (1.0 + k)

    zmask = c <= 1.0
    nz = zmask.sum()

    a = zeros((nz, nldc + 1))
    for j in range(nldc + 1):
        a[:, j] = alpha(b, c[zmask], j, npol, nldc, anm, avl, ajd, aje)

    cn = ones((npb, nldc + 1))
    n = arange(nldc + 1)
    for ui in range(npb):
        cn[ui, 0] = (1. - u[ui].sum()) / (1.0 - sum(u[ui] * n[1:] / (n[1:] + 2.0)))
        cn[ui, 1:] = u[ui] / (1.0 - sum(n[1:] * u[ui] / (n[1:] + 2.0)))

    model = zeros((npb, nz))
    for i in range(nz):
        for j in range(npb):
            model[j, i] = -sum(a[i, :] * cn[j], 0)

    flux = ones((npb, z.size))
    flux[:, zmask] += model
    return flux


@njit
def general_model_z(z, k, u, npol, nldc, npb, anm, avl, ajd, aje):
    u = atleast_2d(u)
    assert u.shape == (npb, nldc)

    b = k / (1.0 + k)
    c = z / (1.0 + k)

    if c > 1.0:
        return ones(npb)
    else:
        a = zeros(nldc + 1)
        for j in range(nldc + 1):
            a[j] = alpha_s(b, c, j, npol, nldc, anm, avl, ajd, aje)

        cn = ones((npb, nldc + 1))
        n = arange(nldc + 1)
        for ui in range(npb):
            cn[ui, 0] = (1. - u[ui].sum()) / (1.0 - sum(u[ui] * n[1:] / (n[1:] + 2.0)))
            cn[ui, 1:] = u[ui] / (1.0 - sum(n[1:] * u[ui] / (n[1:] + 2.0)))

        flux = zeros(npb)
        for j in range(npb):
            flux[j] = -sum(a * cn[j], 0)
        return flux + 1


@njit(parallel=True)
def general_model_pz(z, k, u, npol, nldc, npb, anm, avl, ajd, aje):
    flux = zeros((npb, z.size))

    for i in prange(z.size):
        flux[:, i] = general_model_z(z[i], k, u, npol, nldc, npb, anm, avl, ajd, aje)
    return flux


@njit(parallel=True, fastmath=True)
def general_model_v(t: ndarray, k: ndarray, t0: ndarray, p: ndarray, a: ndarray, i: ndarray, e: ndarray, w: ndarray, ldc: ndarray,
                    mode: int, lcids: ndarray, pbids: ndarray, nsamples: ndarray, exptimes: ndarray,
                    npb: int, npol: int, nldc: int, es: ndarray, ms: ndarray, tae: ndarray,
                    anm: ndarray, avl: ndarray, ajd: ndarray, aje: ndarray):

    k = atleast_2d(k)
    t0, p, a, i, e, w = atleast_1d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)

    ldc = atleast_2d(ldc)
    if ldc.shape[1] != nldc*npb:
        raise ValueError("The quadratic model needs two limb darkening coefficients per passband")

    npv = k.shape[0]
    npt = t.size

    if mode == MODE_NORMAL:
        flux = zeros((npv, 1, npt))
        npc = 1
    elif mode == MODE_TRSPEC:
        flux = zeros((npv, npb, npt))
        npc = npb
    else:
        raise NotImplementedError

    ulcs = unique(lcids)
    nlcs = ulcs.size
    for ipv in prange(npv):

        fpv = zeros((flux.shape[1], flux.shape[2]))

        if isnan(a[ipv]) or isnan(i[ipv]):
            flux[ipv, :, :] = inf
            continue

        for ilc in range(nlcs):
            ipb = pbids[ilc]

            if k.shape[1] == 1:
                _k = k[ipv, 0]
            else:
                _k = k[ipv, ipb]

            if mode == MODE_NORMAL:
                lldc = ldc[ipv, nldc * ipb: nldc * (ipb + 1)].reshape((1, nldc))
            else:
                lldc = ldc[ipv, :].reshape((npb, nldc))

            msk = lcids == ilc

            tlc = t[msk]
            for isample in range(1, nsamples[ilc] + 1):
                time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                z = z_ip_v(tlc + time_offset, t0[ipv], p[ipv], a[ipv], i[ipv], e[ipv], w[ipv], es, ms, tae)
                fpv[:, msk] += general_model_vz(z, _k, lldc, npol, nldc, npc, anm, avl, ajd, aje)
            fpv[:, msk] /= nsamples[ilc]
        flux[ipv, :, :] = fpv

    return flux



@njit(parallel=True, fastmath=False)
def general_model_s(t: ndarray, k: ndarray, t0: float, p: float, a: float, i: float, e: float, w: float, ldc: ndarray,
                    mode: int, lcids: ndarray, pbids: ndarray, nsamples: ndarray, exptimes: ndarray,
                    npb: int, npol: int, nldc: int, es: ndarray, ms: ndarray, tae: ndarray,
                    anm: ndarray, avl: ndarray, ajd: ndarray, aje: ndarray):
    """

    Parameters
    ----------
    t: ndarray
        Mid-exposure times
    k: ndarray
        Planet-star radius ratio(s)
    t0: float
        Zero epoch
    p: float
        Orbital period
    a: float
        Semi-major axis divided by the stellar radius
    i: float
        inclination [rad]
    e: float
        Eccentricity
    w: float
        Argument of peri-astron [rad]
    ldc: ndarray
        Limb darkening coefficients, should have a size of npb*nldc.
    mode: int
        Either `0` for normal behaviour or `1` for transmission spectroscopy. The transmission spectroscopy mode
        ignores the passband indices (`pbids`) and evaluates the model over all the passbands for all the exposures.
        The evaluation is done using optimizations presented in Parviainen (2015) and is much faster than evaluating
        the model in normal mode for transmission spectroscopy. However, the approach assumes that the photometry
        has been created from a spectroscopic time series so that all the passbands are observed simultaneously.
    lcids: ndarray
        Light curve indices
    pbids: ndarray
        Passband indices
    nsamples: ndarray
        Number of samples per exposure
    exptimes: ndarray
        Exposure times
    npb: int
        Number of passbands
    npol: int
        Number of polynomials
    nldc: int
        Number of limb darkening coefficients
    es
    ms
    tae
    anm
    avl
    ajd
    aje

    Returns
    -------

    """
    ldc = atleast_1d(ldc)
    k = atleast_1d(k)
    npt = t.size

    if mode == MODE_NORMAL:
        flux = zeros((1, npt))
        npc = 1
    elif mode == MODE_TRSPEC:
        flux = zeros((npb, npt))
        npc = npb
    else:
        raise NotImplementedError

    ulcs = unique(lcids)
    nlcs = ulcs.size

    if any(isnan(k)) or isnan(a) or isnan(i):
        flux[:, :] = inf
        return flux

    for ilc in prange(nlcs):
        ipb = pbids[ilc]

        if k.size == 1:
            _k = k[0]
        else:
            _k = k[ipb]

        if mode == MODE_NORMAL:
            lldc = ldc[nldc * ipb: nldc * (ipb + 1)].reshape((1, nldc))
        else:
            lldc = ldc.reshape((npb, nldc))

        msk = lcids == ilc

        tlc = t[msk]
        for isample in range(1, nsamples[ilc] + 1):
            time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
            z = z_ip_v(tlc + time_offset, t0, p, a, i, e, w, es, ms, tae)
            flux[:, msk] += general_model_vz(z, _k, lldc, npol, nldc, npc, anm, avl, ajd, aje)
        flux[:, msk] /= nsamples[ilc]
    return flux

@njit(parallel=True, fastmath=False)
def general_model_pv(t: ndarray, pvp: ndarray, ldc: ndarray,
                    mode: int, lcids: ndarray, pbids: ndarray, nsamples: ndarray, exptimes: ndarray,
                    npb: int, npol: int, nldc: int, es: ndarray, ms: ndarray, tae: ndarray,
                    anm: ndarray, avl: ndarray, ajd: ndarray, aje: ndarray):

    pvp = atleast_2d(pvp)
    nk = pvp.shape[1] - 6
    return general_model_v(t, pvp[:,:nk], t0=pvp[:,nk], p=pvp[:,nk+1], a=pvp[:,nk+2], i=pvp[:,nk+3], e=pvp[:,nk+4], w=pvp[:,nk+5],
                        ldc=ldc, mode=mode, lcids=lcids, pbids=pbids, nsamples=nsamples, exptimes=exptimes,
                        npb=npb, npol=npol, nldc=nldc, es=es, ms=ms, tae=tae, anm=anm, avl=avl, ajd=ajd, aje=aje)


@njit(parallel=True, fastmath=True)
def general_model_s_slow(t, k, t0, p, a, i, e, w, ldc, lcids, pbids, nsamples, exptimes, npb, es, ms, tae,
                    npol, nldc, anm, avl, ajd, aje):
    ldc = atleast_1d(ldc)
    k = atleast_1d(k)

    npt = t.size
    flux = zeros(npt)
    for j in prange(npt):
        ilc = lcids[j]
        ipb = pbids[ilc]

        if k.size == 1:
            _k = k[0]
        else:
            _k = k[ipb]

        if isnan(_k) or isnan(a) or isnan(i):
            flux[j] = inf
        else:
            for isample in range(1, nsamples[ilc] + 1):
                time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                z = z_ip_s(t[j] + time_offset, t0, p, a, i, e, w, es, ms, tae)
                if z > 1.0 + _k:
                    flux[j] += 1.
                else:
                    flux[j] += general_model_z(z, _k, ldc[nldc * ipb: nldc * (ipb + 1)], npol, nldc, npb, anm, avl, ajd, aje)[0]
            flux[j] /= nsamples[ilc]
    return flux
