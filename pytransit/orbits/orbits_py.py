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
from numpy import (pi, arccos, arctan2, sin, cos, sqrt, sign, copysign, mod, zeros_like, zeros, linspace, floor, arcsin,
                   int, around, nan, full)
from scipy.constants import G

HALF_PI = 0.5 * pi
TWO_PI = 2.0 * pi
INV_PI = 1.0 / pi

D_H = 24.
D_M = 60 * D_H
D_S = 60 * D_M

au,   au_e             = 1.496e11, 0.0
msun, msun_e           = 1.9891e30, 0.0
rsun, rsun_e           = 0.5*1.392684e9, 0.0

# Utilities
# =========

cache = False

def epoch(time, zero_epoch, period):
    return around((time - zero_epoch) / period).astype(int)

@njit
def mean_anomaly_offset(e, w):
    mean_anomaly_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    mean_anomaly_offset -= e*sin(mean_anomaly_offset)
    return mean_anomaly_offset

@njit
def z_from_ta_s(Ta, a, i, e, w):
    z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
    z *= copysign(1.0, sin(w+Ta))
    return z

@njit(parallel=True)
def z_from_ta_v(Ta, a, i, e, w):
    z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
    z *= sign(1.0, sin(w+Ta))
    return z

@njit
def rclip(v, vmin, vmax):
    return min(max(v, vmin), vmax)

@njit
def iclip(v, vmin, vmax):
    return int(min(max(v, vmin), vmax))

# Mean Anomaly
# ============

@njit(cache=cache)
def mean_anomaly(t, t0, p, e, w):
    offset = mean_anomaly_offset(e, w)
    Ma = mod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI)
    return Ma

@njit(parallel=True)
def mean_anomaly_p(t, t0, p, e, w):
    offset = mean_anomaly_offset(e, w)
    Ma = mod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI)
    return Ma


# Ecccentric anomaly
# ==================

@njit(cache=cache)
def ea_newton_v(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ea = Ma.copy()
    for j in range(len(t)):
        err = 0.05
        k = 0
        while abs(err) > 1e-8 and k<1000:
            err   = Ea[j] - e*sin(Ea[j]) - Ma[j]
            Ea[j] = Ea[j] - err/(1.0-e*cos(Ea[j]))
            k += 1
    return Ea

@njit(cache=cache)
def ea_newton_s(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ea = Ma
    err = 0.05
    k = 0
    while abs(err) > 1e-8 and k<1000:
        err   = Ea - e*sin(Ea) - Ma
        Ea = Ea - err/(1.0-e*cos(Ea))
        k += 1
    return Ea

@njit(cache=cache)
def ea_iter_s(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    ec = e*sin(Ma)/(1.0 - e*cos(Ma))
    for k in range(15):
        ect = ec
        ec  = e*sin(Ma+ec)
        if (abs(ect-ec) < 1e-4):
            break
    Ea  = Ma + ec
    return Ea

@njit(parallel=True)
def ea_iter_v(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    ec = e*sin(Ma)/(1.0 - e*cos(Ma))
    for j in prange(len(t)):
        for k in range(15):
            ect   = ec[j]
            ec[j] = e*sin(Ma[j]+ec[j])
            if (abs(ect-ec[j]) < 1e-4):
                break
    Ea  = Ma + ec
    return Ea

# True Anomaly
# ============

@njit(parallel=True)
def ta_from_ea_v(Ea, e):
    sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
    cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
    Ta  = arctan2(sta, cta)
    return Ta

@njit(cache=cache)
def ta_from_ea_s(Ea, e):
    sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
    cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
    Ta  = arctan2(sta, cta)
    return Ta

@njit(cache=cache)
def ta_newton_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_newton_s(t, t0, p, e, w), e)

@njit
def ta_newton_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_newton_v(t, t0, p, e, w), e)

@njit(cache=cache)
def ta_iter_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_iter_s(t, t0, p, e, w), e)

@njit
def ta_iter_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_iter_v(t, t0, p, e, w), e)

@njit(parallel=True)
def ta_ps3(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = (Ma + (2.0*e - 0.25*e**3)*sin(Ma)
             + 1.25*e**2*sin(2.0*Ma)
             + 13.0/12.0*e**3*sin(3.0*Ma))
    return Ta

@njit(parallel=True)
def ta_ps5(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = (Ma + (2.0*e - 0.25*e**3 + 5.0/96.0*e**5) * sin(Ma)
             + (1.25*e**2 - 11.0/24.0*e**4) * sin(2.0*Ma)
             + (13.0/12.0 * e**3 - 43.0/64.0 * e**5) * sin(3.0*Ma)
             + 103.0/96.0 * e**4 * sin(4.0*Ma)
             + 1097.0/960.0 * e**5 * sin(5.0*Ma))
    return Ta

@njit
def ta_from_ma(Ma, e, cache=cache):
    Ta = zeros_like(Ma)
    for j in range(len(Ma)):
        ec = e*sin(Ma[j])/(1.0 - e*cos(Ma[j]))
        for k in range(15):
            ect = ec
            ec  = e*sin(Ma[j]+ec)
            if abs(ect-ec) < 1e-4:
                break
        Ea = Ma[j] + ec
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta[j] = arctan2(sta, cta)
    return Ta

@njit
def ta_ip_calculate_table(ne=256, nm=512, cache=cache):
    es = linspace(0, 0.95, ne)
    ms = linspace(0,   pi, nm)
    tae = zeros((ne, nm))
    for i,e in enumerate(es):
        tae[i,:]  = ta_from_ma(ms, e)
        tae[i,:] -= ms
    return tae, es, ms

@njit(parallel=True)
def ta_ip(t, t0, p, e, w, es, ms, tae):
    ne = es.size
    nm = ms.size
    de = es[1] - es[0]
    dm = ms[1] - ms[0]

    ie = iclip(         floor(e/de),      0,   ne-1)
    ae = rclip((e - de*(ie-1)) / de,    0.0,    1.0)
    tae2 = tae[ie:ie+2,:]

    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = zeros_like(Ma)

    for i in range(len(t)):
        if Ma[i] < pi:
            im = iclip(floor(Ma[i]/dm), 0, nm-1)
            am = rclip((Ma[i] - dm*(im-1)) / dm, 0.0, 1.0)
            s = 1.0
        else:
            im = iclip(floor((TWO_PI - Ma[i])/dm), 0, nm-1)
            am = rclip((TWO_PI - (Ma[i] - dm*(im-1))) / dm, 0.0, 1.0)
            s = -1.0

        Ta[i] = ( tae2[0,im  ]*(1.0-ae)*(1.0-am)
                + tae2[1,im  ]*     ae *(1.0-am)
                + tae2[0,im+1]*(1.0-ae)*     am
                + tae2[1,im+1]*     ae *     am  )
        Ta[i] = Ma[i] + s * Ta[i]

        if (Ta[i] < 0.0):
            Ta[i] = Ta[i] + TWO_PI

    return Ta

# Projected distance Z
# ====================
# These functions calculate the projected distance (z) using different ways to calculate the
# tue anomaly (Ta). The functions have different versions optimized for different use-cases
#
#  - z_*_s  : scalar time
#  - z_*_v  : vector time
#  - z_*_p  : vector time, parallelized and usually fastest
#  - z_*_mp : vector time and two-dimensional parameter array, can be faster than the others
#

# Z: Newton's method
# ------------------


@njit(cache=cache)
def z_circular(t, pv):
    t0, p, a, i, e, w = pv
    cosph = cos(TWO_PI*(t-t0)/p)
    z = sign(cosph) * a*sqrt(1.0 - cosph*cosph*sin(i)**2)
    return z


@njit(fastmath=True, parallel=False)
def z_ip_v(t, t0, p, a, i, e, w, es, ms, tae):
    if e < 0. or e >= es[-1]:
        return full(t.size, nan)

    Ma = mean_anomaly(t, t0, p, e, w)
    if e < 0.01:
        Ta = Ma
    else:
        de = es[1] - es[0]
        dm = ms[1] - ms[0]

        ie = int(floor(e / de))
        ae = (e - de * ie) / de
        tae2 = tae[ie:ie + 2, :]

        Ta = zeros_like(Ma)

        for j in range(len(t)):
            if Ma[j] < pi:
                x = Ma[j]
                s = 1.
            else:
                x = TWO_PI - Ma[j]
                s = -1.

            im = int(floor(x / dm))
            am = (x - im * dm) / dm

            Ta[j] = (tae2[0, im] * (1.0 - ae) * (1.0 - am)
                     + tae2[1, im] * ae * (1.0 - am)
                     + tae2[0, im + 1] * (1.0 - ae) * am
                     + tae2[1, im + 1] * ae * am)
            Ta[j] = Ma[j] + s * Ta[j]

            if (Ta[j] < 0.0):
                Ta[j] = Ta[j] + TWO_PI

    z = a * (1.0 - e ** 2) / (1.0 + e * cos(Ta)) * sqrt(1.0 - sin(w + Ta) ** 2 * sin(i) ** 2)
    z *= copysign(1.0, sin(w + Ta))
    return z


@njit(fastmath=True)
def z_ip_s(t, t0, p, a, i, e, w, es, ms, tae):
    if e < 0. or e >= es[-1]:
        return nan

    Ma = mean_anomaly(t, t0, p, e, w)
    if e < 0.01:
        Ta = Ma
    else:
        de = es[1] - es[0]
        dm = ms[1] - ms[0]

        ie = int(floor(e / de))
        ae = (e - de * ie) / de

        if Ma < pi:
            x = Ma
            s = 1.
        else:
            x = TWO_PI - Ma
            s = -1.

        im = int(floor(x / dm))
        am = (x - im * dm) / dm

        Ta = (tae[ie, im] * (1.0 - ae) * (1.0 - am)
              + tae[ie + 1, im] * ae * (1.0 - am)
              + tae[ie, im + 1] * (1.0 - ae) * am
              + tae[ie + 1, im + 1] * ae * am)
        Ta = Ma + s * Ta

        if (Ta < 0.0):
            Ta = Ta + TWO_PI

    z = a * (1.0 - e ** 2) / (1.0 + e * cos(Ta)) * sqrt(1.0 - sin(w + Ta) ** 2 * sin(i) ** 2)
    z *= copysign(1.0, sin(w + Ta))
    return z


@njit(cache=cache)
def z_newton_s(t, pv):
    """Normalized projected distance for scalar t.

    pv = [t0, p, a, i, e, w]
    """
    t0, p, a, i, e, w = pv
    Ta = ta_newton_s(t, t0, p, e, w)
    return z_from_ta_s(Ta, a, i, e, w)

@njit(cache=cache)
def z_newton_v(ts, pv):
    t0, p, a, i, e, w = pv
    Ta = ta_newton_v(ts, t0, p, e, w)
    return z_from_ta_v(Ta, a, i, e, w)

@njit(parallel=True, fastmath=True)
def z_newton_p(ts, pv):
    t0, p, a, i, e, w = pv
    zs = zeros_like(ts)
    for j in prange(len(ts)):
        t = ts[j]
        ma_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
        ma_offset -= e*sin(ma_offset)
        Ma = mod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI)
        Ea = Ma
        err = 0.05
        k = 0
        while abs(err) > 1e-8 and k<1000:
            err   = Ea - e*sin(Ea) - Ma
            Ea = Ea - err/(1.0-e*cos(Ea))
            k += 1
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta  = arctan2(sta, cta)
        z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
        z *= copysign(1.0, sin(w+Ta))
        zs[j] = z
    return zs

@njit(parallel=True, fastmath=True)
def z_newton_mp(ts, pvs):
    zs = zeros((pvs.shape[0], ts.size))
    for j in prange(zs.size):
        ipv, ipt = j // ts.size, j % ts.size
        t0, p, a, i, e, w = pvs[ipv]
        t = ts[ipt]
        ma_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
        ma_offset -= e*sin(ma_offset)
        Ma = mod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI)
        Ea = Ma
        err = 0.05
        k = 0
        while abs(err) > 1e-8 and k<1000:
            err   = Ea - e*sin(Ea) - Ma
            Ea = Ea - err/(1.0-e*cos(Ea))
            k += 1
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta  = arctan2(sta, cta)
        z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
        z *= copysign(1.0, sin(w+Ta))
        zs[ipv,ipt] = z
    return zs



# Z: Iteration
# ------------


@njit(cache=cache)
def z_iter_s(t, pv):
    t0, p, a, i, e, w = pv
    Ta = ta_iter_s(t, t0, p, e, w)
    return z_from_ta_s(Ta, a, i, e, w)


@njit(cache=cache)
def z_iter_v(ts, pv):
    t0, p, a, i, e, w = pv
    Ta = ta_iter_v(ts, t0, p, e, w)
    return z_from_ta_v(Ta, a, i, e, w)


@njit(parallel=True, fastmath=True)
def z_iter_p(ts, pv):
    t0, p, a, i, e, w = pv
    ma_offset = arctan2(sqrt(1.0 - e ** 2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    ma_offset -= e * sin(ma_offset)
    zs = zeros_like(ts)
    for j in prange(len(ts)):
        t = ts[j]
        Ma = mod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI)
        ec = e*sin(Ma)/(1.0 - e*cos(Ma))
        for k in range(15):
            ect   = ec
            ec = e*sin(Ma+ec)
            if (abs(ect-ec) < 1e-4):
                break
        Ea  = Ma + ec
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta  = arctan2(sta, cta)
        z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
        z *= copysign(1.0, sin(w+Ta))
        zs[j] = z
    return zs

# Z: Series expansion
# -------------------


@njit(cache=cache)
def z_ps3(t, pv):
    t0, p, a, i, e, w = pv
    return z_from_ta_v(ta_ps3(t, t0, p, e, w), a, i, e, w)


@njit(cache=cache)
def z_ps5(t, pv):
    t0, p, a, i, e, w = pv
    return z_from_ta_v(ta_ps5(t, t0, p, e, w), a, i, e, w)


# Utility functions
# -----------------

@njit(cache=cache)
def impact_parameter(a, i):
    return a * cos(i)


@njit(cache=cache)
def impact_parameter_ec(a, i, e, w, tr_sign):
    return a * cos(i) * ((1.-e**2) / (1.+tr_sign*e*sin(w)))


@njit(cache=cache)
def duration_eccentric(p, k, a, i, e, w, tr_sign):
    b  = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = sqrt(1.-e**2)/(1.+tr_sign*e*sin(w))
    return p/pi  * arcsin(sqrt((1.+k)**2-b**2)/(a*sin(i))) * ae


@njit(cache=cache)
def eclipse_phase(p, i, e, w):
    """ Phase for the secondary eclipse center.

    Exact secondary eclipse center phase, good for all eccentricities.
    """
    etr = arctan2(sqrt(1. - e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    eec = arctan2(sqrt(1. - e**2) * sin(HALF_PI + pi - w), e + cos(HALF_PI + pi - w))
    mtr = etr - e * sin(etr)
    mec = eec - e * sin(eec)
    phase = (mec - mtr) * p / TWO_PI
    return phase if phase > 0. else p + phase


@njit(cache=cache)
def eclipse_phase_ap(p, i, e, w):
    """Approximate phase for the secondary eclipse center.

    Approximate phase for the center of the secondary eclipse (ok for e < 0.5)
    (Kallrath, J., "Eclipsing Binary Stars: Modeling and Analysis", p. 85)
    """
    return 0.5 * p + (p * INV_PI * e * cos(w) * (1. + 1. / sin(i)**2))


@njit(cache=cache)
def p_from_am(a=1., ms=1.):
    """Orbital period from the semi-major axis and stellar mass.

    Parameters
    ----------

      a    : semi-major axis [AU]
      ms   : stellar mass    [M_Sun]

    Returns
    -------

      p    : Orbital period  [d]
    """
    return sqrt((4*pi**2*(a*au)**3)/(G*ms*msun)) / D_S


@njit(cache=cache)
def a_from_mp(ms, period):
    """Semi-major axis from the stellar mass and planet's orbital period.

    Parameters
    ----------

      ms     : stellar mass    [M_Sun]
      period : orbital period  [d]

    Returns
    -------

      a : semi-major axis [AU]
    """
    return ((G * (ms*msun) * (period * D_S)**2) / (4 * pi**2))**(1 / 3) / au


@njit(cache=cache)
def as_from_rhop(rho, period):
    """Scaled semi-major axis from the stellar density and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]

    Returns
    -------

      as : scaled semi-major axis [R_star]
    """
    return (G/(3*pi))**(1/3) * ((period * D_S)**2 * 1e3 * rho)**(1 / 3)


@njit(cache=cache)
def rho_from_asp(a, period):
    """Stellar density from the scaled semi-major axis and orbital period.
    """
    return (3*pi*a**3)/(G * (period * D_S)**2) * 1e-3


@njit(cache=cache)
def a_from_rhoprs(rho, period, rstar):
    """Semi-major axis from the stellar density, stellar radius, and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]
      rstar  : stellar radius  [R_Sun]

    Returns
    -------

      a : semi-major axis [AU]
    """
    return as_from_rhop(rho, period)*rstar*rsun/au


@njit(cache=cache)
def af_transit(e, w):
    """Calculates the -- factor during the transit"""
    return (1.0-e**2)/(1.0 + e*sin(w))


@njit(cache=cache)
def i_from_baew(b, a, e, w):
    """Orbital inclination from the impact parameter, scaled semi-major axis, eccentricity and argument of periastron

    Parameters
    ----------

      b  : impact parameter       [-]
      a  : scaled semi-major axis [R_Star]
      e  : eccentricity           [-]
      w  : argument of periastron [rad]

    Returns
    -------

      i  : inclination            [rad]
    """
    return arccos(b / (a*af_transit(e, w)))


@njit(cache=cache)
def i_from_ba(b, a):
    """Orbital inclination from the impact parameter and scaled semi-major axis.

    Parameters
    ----------

      b  : impact parameter       [-]
      a  : scaled semi-major axis [R_Star]

    Returns
    -------

      i  : inclination            [rad]
    """
    return arccos(b/a)


@njit(cache=cache)
def d_from_pkaiews(p, k, a, i, e, w, tr_sign, kind=14):
    """Transit duration (T14 or T23) from p, k, a, i, e, w, and the transit sign.

    Calculates the transit duration (T14) from the orbital period, planet-star radius ratio, scaled semi-major axis,
    orbital inclination, eccentricity, argument of periastron, and the sign of the transit (transit:1, eclipse: -1).

     Parameters
     ----------

       p  : orbital period         [d]
       k  : radius ratio           [R_Star]
       a  : scaled semi-major axis [R_star]
       i  : orbital inclination    [rad]
       e  : eccentricity           [-]
       w  : argument of periastron [rad]
       tr_sign : transit sign, 1 for a transit, -1 for an eclipse
       kind: either 14 for full transit duration or 23 for total transit duration

     Returns
     -------

       d  : transit duration T14  [d]
     """
    b  = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = sqrt(1.-e**2)/(1.+tr_sign*e*sin(w))
    ds = 1. if kind == 14 else -1.
    return p/pi  * arcsin(sqrt((1.+ds*k)**2-b**2)/(a*sin(i))) * ae


@njit(cache=cache)
def p_from_dkaiews(d, k, a, i, e, w, tr_sign):
    """Orbital period from d, k, a, i, e, w, and the transit sign.

    Calculates the orbital period from the transit duration (T14), planet-star radius ratio, scaled semi-major axis,
    orbital inclination, eccentricity, argument of periastron, and the sign of the transit (transit:1, eclipse: -1).

     Parameters
     ----------

       d  : transit duration T14   [d]
       k  : radius ratio           [R_Star]
       a  : scaled semi-major axis [R_star]
       i  : orbital inclination    [rad]
       e  : eccentricity           [-]
       w  : argument of periastron [rad]
       tr_sign : transit sign, 1 for a transit, -1 for an eclipse


    Returns
    -------

      p  : orbital period         [d]
    """
    b  = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = sqrt(1.-e**2)/(1.+tr_sign*e*sin(w))
    return (d*pi) / (arcsin(sqrt((1.+k)**2-b**2)/(a*sin(i))) * ae)


@njit
def as_from_dkp(d, p, k):
    """Assumes b=0"""
    return sqrt((1.0+k)**2) / sin(pi*d/p)