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
from numpy import zeros_like, cos, sin, floor, sqrt, zeros

from .orbits_py import ta_newton_s, ta_newton_v, i_from_baew, eclipse_phase


@njit(fastmath=True)
def vajs_from_paiew(p, a, i, e, w):
    """Planet velocity, acceleration, jerk, and snap at mid-transit in [R_star / day]"""

    # Time step for central finite difference
    # ---------------------------------------
    # I've tried to choose a value that is small enough to
    # work with ultra-short-period orbits and large enough
    # not to cause floating point problems with the fourth
    # derivative (anything much smaller starts hitting the
    # double precision limit.)
    dt = 2e-2

    ae = a*(1. - e**2)
    ci = cos(i)

    # Calculation of X and Y positions
    # --------------------------------
    # These could just as well be calculated with a single
    # loop with X and Y as arrays, but I've decided to
    # manually unroll it because it seems to give a small
    # speed advantage with numba.

    f0 = ta_newton_s(-3*dt, 0.0, p, e, w)
    f1 = ta_newton_s(-2*dt, 0.0, p, e, w)
    f2 = ta_newton_s(  -dt, 0.0, p, e, w)
    f3 = ta_newton_s(  0.0, 0.0, p, e, w)
    f4 = ta_newton_s(   dt, 0.0, p, e, w)
    f5 = ta_newton_s( 2*dt, 0.0, p, e, w)
    f6 = ta_newton_s( 3*dt, 0.0, p, e, w)

    r0 = ae/(1. + e*cos(f0))
    r1 = ae/(1. + e*cos(f1))
    r2 = ae/(1. + e*cos(f2))
    r3 = ae/(1. + e*cos(f3))
    r4 = ae/(1. + e*cos(f4))
    r5 = ae/(1. + e*cos(f5))
    r6 = ae/(1. + e*cos(f6))

    x0 = -r0*cos(w + f0)
    x1 = -r1*cos(w + f1)
    x2 = -r2*cos(w + f2)
    x3 = -r3*cos(w + f3)
    x4 = -r4*cos(w + f4)
    x5 = -r5*cos(w + f5)
    x6 = -r6*cos(w + f6)

    y0 = -r0*sin(w + f0)*ci
    y1 = -r1*sin(w + f1)*ci
    y2 = -r2*sin(w + f2)*ci
    y3 = -r3*sin(w + f3)*ci
    y4 = -r4*sin(w + f4)*ci
    y5 = -r5*sin(w + f5)*ci
    y6 = -r6*sin(w + f6)*ci

    # First time derivative of position: velocity
    # -------------------------------------------
    a, b, c = 1/60, 9/60, 45/60
    vx = (a*(x6 - x0) + b*(x1 - x5) + c*(x4 - x2))/dt
    vy = (a*(y6 - y0) + b*(y1 - y5) + c*(y4 - y2))/dt

    # Second time derivative of position: acceleration
    # ------------------------------------------------
    a, b, c, d = 1/90, 3/20, 3/2, 49/18
    ax = (a*(x0 + x6) - b*(x1 + x5) + c*(x2 + x4) - d*x3)/dt**2
    ay = (a*(y0 + y6) - b*(y1 + y5) + c*(y2 + y4) - d*y3)/dt**2

    # Third time derivative of position: jerk
    # ---------------------------------------
    a, b, c = 1/8, 1, 13/8
    jx = (a*(x0 - x6) + b*(x5 - x1) + c*(x2 - x4))/dt**3
    jy = (a*(y0 - y6) + b*(y5 - y1) + c*(y2 - y4))/dt**3

    # Fourth time derivative of position: snap
    # ----------------------------------------
    a, b, c, d = 1/6, 2, 13/2, 28/3
    sx = (-a*(x0 + x6) + b*(x1 + x5) - c*(x2 + x4) + d*x3)/dt**4
    sy = (-a*(y0 + y6) + b*(y1 + y5) - c*(y2 + y4) + d*y3)/dt**4

    return y3, vx, vy, ax, ay, jx, jy, sx, sy


@njit(fastmath=True)
def vajs_from_paiew_eclipse(p, a, i, e, w):
    """Planet velocity, acceleration, jerk, and snap at mid-eclipse in [R_star / day]"""

    # Time step for central finite difference
    # ---------------------------------------
    # I've tried to choose a value that is small enough to
    # work with ultra-short-period orbits and large enough
    # not to cause floating point problems with the fourth
    # derivative (anything much smaller starts hitting the
    # double precision limit.)
    dt = 2e-2

    ae = a * (1. - e ** 2)
    ci = cos(i)

    # Calculation of X and Y positions
    # --------------------------------
    # These could just as well be calculated with a single
    # loop with X and Y as arrays, but I've decided to
    # manually unroll it because it seems to give a small
    # speed advantage with numba.

    te = eclipse_phase(p, i, e, w)

    f0 = ta_newton_s(te - 3 * dt, 0.0, p, e, w)
    f1 = ta_newton_s(te - 2 * dt, 0.0, p, e, w)
    f2 = ta_newton_s(te - dt, 0.0, p, e, w)
    f3 = ta_newton_s(te, 0.0, p, e, w)
    f4 = ta_newton_s(te + dt, 0.0, p, e, w)
    f5 = ta_newton_s(te + 2 * dt, 0.0, p, e, w)
    f6 = ta_newton_s(te + 3 * dt, 0.0, p, e, w)

    r0 = ae / (1. + e * cos(f0))
    r1 = ae / (1. + e * cos(f1))
    r2 = ae / (1. + e * cos(f2))
    r3 = ae / (1. + e * cos(f3))
    r4 = ae / (1. + e * cos(f4))
    r5 = ae / (1. + e * cos(f5))
    r6 = ae / (1. + e * cos(f6))

    x0 = -r0 * cos(w + f0)
    x1 = -r1 * cos(w + f1)
    x2 = -r2 * cos(w + f2)
    x3 = -r3 * cos(w + f3)
    x4 = -r4 * cos(w + f4)
    x5 = -r5 * cos(w + f5)
    x6 = -r6 * cos(w + f6)

    y0 = -r0 * sin(w + f0) * ci
    y1 = -r1 * sin(w + f1) * ci
    y2 = -r2 * sin(w + f2) * ci
    y3 = -r3 * sin(w + f3) * ci
    y4 = -r4 * sin(w + f4) * ci
    y5 = -r5 * sin(w + f5) * ci
    y6 = -r6 * sin(w + f6) * ci

    # First time derivative of position: velocity
    # -------------------------------------------
    a, b, c = 1 / 60, 9 / 60, 45 / 60
    vx = (a * (x6 - x0) + b * (x1 - x5) + c * (x4 - x2)) / dt
    vy = (a * (y6 - y0) + b * (y1 - y5) + c * (y4 - y2)) / dt

    # Second time derivative of position: acceleration
    # ------------------------------------------------
    a, b, c, d = 1 / 90, 3 / 20, 3 / 2, 49 / 18
    ax = (a * (x0 + x6) - b * (x1 + x5) + c * (x2 + x4) - d * x3) / dt ** 2
    ay = (a * (y0 + y6) - b * (y1 + y5) + c * (y2 + y4) - d * y3) / dt ** 2

    # Third time derivative of position: jerk
    # ---------------------------------------
    a, b, c = 1 / 8, 1, 13 / 8
    jx = (a * (x0 - x6) + b * (x5 - x1) + c * (x2 - x4)) / dt ** 3
    jy = (a * (y0 - y6) + b * (y5 - y1) + c * (y2 - y4)) / dt ** 3

    # Fourth time derivative of position: snap
    # ----------------------------------------
    a, b, c, d = 1 / 6, 2, 13 / 2, 28 / 3
    sx = (-a * (x0 + x6) + b * (x1 + x5) - c * (x2 + x4) + d * x3) / dt ** 4
    sy = (-a * (y0 + y6) + b * (y1 + y5) - c * (y2 + y4) + d * y3) / dt ** 4

    return te, y3, vx, vy, ax, ay, jx, jy, sx, sy


@njit
def vajs_from_paiew_v(p, a, i, e, w):
    npv = p.size
    vajs = zeros((npv, 9))
    for j in range(npv):
        vajs[j] = vajs_from_paiew(p[j], a[j], i[j], e[j], w[j])
    return vajs


@njit(fastmath=True)
def z_taylor_s(tc, t0, p, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    """Normalized planet-star center distance using Taylor series expansion.

    Parameters
    ----------
    tc
    t0
    p
    b
    vx
    vy
    ax
    ay
    jx
    jy

    Returns
    -------

    """
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    px =      vx * t + 0.5 * ax * t2 + jx * t3 / 6.0 + sx * t4 / 24.
    py = y0 + vy * t + 0.5 * ay * t2 + jy * t3 / 6.0 + sy * t4 / 24.
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def z_taylor_st(t, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    """Normalized planet-star center distance using Taylor series expansion.

    Parameters
    ----------
    tc
    t0
    p
    b
    vx
    vy
    ax
    ay
    jx
    jy

    Returns
    -------

    """
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    px =      vx * t + 0.5 * ax * t2 + jx * t3 / 6.0 + sx * t4 / 24.
    py = y0 + vy * t + 0.5 * ay * t2 + jy * t3 / 6.0 + sy * t4 / 24.
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def z_taylor_v(times, t0, p, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    z = zeros_like(times)
    npt = times.size
    for i in range(npt):
        epoch = floor((times[i] - t0 + 0.5 * p) / p)
        t = times[i] - (t0 + epoch * p)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        px =      vx * t + 0.5 * ax * t2 + jx * t3 / 6.0 + sx * t4 / 24.
        py = y0 + vy * t + 0.5 * ay * t2 + jy * t3 / 6.0 + sy * t4 / 24.
        z[i] = sqrt(px ** 2 + py ** 2)
    return z


@njit(fastmath=True)
def xy_newton_v(times, t0, p, a, b, e, w):
    """Planet velocity and acceleration at mid-transit in [R_star / day]"""
    i = i_from_baew(b, a, e, w)
    f = ta_newton_v(times, t0, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    return x, y


@njit(fastmath=True)
def xy_taylor_v(times, t0, p, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    npt = times.size
    px = zeros(npt)
    py = zeros(npt)
    for i in range(npt):
        epoch = floor((times[i] - t0 + 0.5 * p) / p)
        t = times[i] - (t0 + epoch * p)
        px[i] =      vx * t + 0.5 * ax * t ** 2 + jx * t ** 3 / 6. + sx * t**4 / 24.
        py[i] = y0 + vy * t + 0.5 * ay * t ** 2 + jy * t ** 3 / 6. + sy * t**4 / 24.
    return px, py


@njit
def find_contact_point(k: float, point: int, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    if point == 1 or point == 2 or point == 12:
        s = -1.0
    else:
        s = 1.0

    if point == 1 or point == 4:
        zt = 1.0 + k
    elif point == 2 or point == 3:
        zt = 1.0 - k
    else:
        zt = 1.0

    t0 = 0.0
    t2 = s*2.0/vx
    t1 = 0.5*t2

    z0 = z_taylor_s(t0, 0.0, 1.0, y0, vx, vy, ax, ay, jx, jy, sx, sy) - zt
    z1 = z_taylor_s(t1, 0.0, 1.0, y0, vx, vy, ax, ay, jx, jy, sx, sy) - zt

    i = 0
    while abs(t2 - t0) > 1e-6 and i < 100:
        if z0*z1 < 0.0:
            t1, t2 = 0.5*(t0 + t1), t1
            z1, z2 = z_taylor_s(t1, 0.0, 1.0, y0, vx, vy, ax, ay, jx, jy, sx, sy) - zt, z1
        else:
            t0, t1 = t1, 0.5*(t1 + t2)
            z0, z1 = z1, z_taylor_s(t1, 0.0, 1.0, y0, vx, vy, ax, ay, jx, jy, sx, sy) - zt
        i += 1
    return t1


@njit
def find_z_min(tc, p, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    r = 0.61803399
    c = 1.0 - r
    x0, x3 = tc - 0.01, tc + 0.01

    x1 = tc
    x2 = tc + c*(x3 - tc)

    f1 = z_taylor_s(x1, tc, p, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    f2 = z_taylor_s(x2, tc, p, y0, vx, vy, ax, ay, jx, jy, sx, sy)

    i = 0
    while abs(x3 - x0) > 1e-7 and i < 100:
        if f2 < f1:
            x0, x1, x2 = x1, x2, r*x2 + c*x3
            f1, f2 = f2, z_taylor_s(x2, tc, p, y0, vx, vy, ax, ay, jx, jy, sx, sy)
        else:
            x3, x2, x1 = x2, x1, r*x1 + c*x0
            f2, f1 = f1, z_taylor_s(x1, tc, p, y0, vx, vy, ax, ay, jx, jy, sx, sy)
        i += 1

    if f1 < f2:
        return x1, f1
    else:
        return x2, f2


def t14(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    t1 = find_contact_point(k, 1, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    t4 = find_contact_point(k, 4, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    return t4-t1


def t23(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    t2 = find_contact_point(k, 2, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    t3 = find_contact_point(k, 3, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    return t3-t2


def t12(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    t1 = find_contact_point(k, 1, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    t2 = find_contact_point(k, 2, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    return t2-t1


def t34(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    t3 = find_contact_point(k, 3, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    t4 = find_contact_point(k, 4, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    return t4-t3


def t1(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    return find_contact_point(k, 1, y0, vx, vy, ax, ay, jx, jy, sx, sy)


def t4(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    return find_contact_point(k, 4, y0, vx, vy, ax, ay, jx, jy, sx, sy)


def bounding_box(k: float, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    t1 = find_contact_point(k, 1, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    t4 = find_contact_point(k, 4, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    return t1, t4