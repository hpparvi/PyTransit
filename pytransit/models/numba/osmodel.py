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
from scipy.constants import G, k, h, c
from numpy import exp, pi, sqrt, zeros, sin, cos, nan, inf, linspace, meshgrid, floor, isfinite, fmax, isnan, nanmean, \
    arange, zeros_like

from .rrmodel import circle_circle_intersection_area
from ...orbits.taylor_z import vajs_from_paiew, find_contact_point

d2sec = 24.*60.*60.


@njit
def planck(l, t):
    return 2.*h*c**2/l**5/(exp(h*c/(l*k*t)) - 1.)


@njit
def stellar_oblateness(w, rho):
    return 3.*w*w/(8.*G*pi*rho)


@njit
def z_s(x, y, r, f, sphi, cphi):
    x2, y2, r2 = x*x, y*y, r*r
    if x2 + y2 > r2:
        return -1.0
    else:
        sphi2, cphi2, f2 = sphi**2, cphi**2, f**2

        a = 1./(1. - f)**2
        d = (4.*y2*sphi2*cphi2*(a - 1.)**2- 4.*(cphi2 + a*sphi2)*(x2 + y2*(a*cphi2 + sphi2) - r2))
        if d >= 0.0:
            return (-2.*y*cphi*sphi*(a - 1.) + sqrt(d))/(2.*(cphi2 + a*sphi2))
        else:
            return -1.0


@njit
def z_v(xs, ys, r, f, sphi, cphi):
    npt = xs.size
    z = zeros(npt)

    sphi2, cphi2, f2, r2 = sphi**2, cphi**2, f**2, r**2
    a = 1./(1. - f)**2

    for i in range(npt):
        x2, y2 = xs[i]**2, ys[i]**2
        if x2 + y2 > r2:
            z[i] = -1.0
        else:
            d = (4.*y2*sphi2*cphi2*(a - 1.)**2 - 4.*(cphi2 + a*sphi2)*(x2 + y2*(a*cphi2 + sphi2) - r2))
            if d >= 0.0:
                z[i] = (-2.*ys[i]*cphi*sphi*(a - 1.) + sqrt(d))/(2.*(cphi2 + a*sphi2))
            else:
                z[i] = -1.0
    return z


@njit
def luminosity_s(x, y, mstar, rstar, ostar, tpole, gpole, f, sphi, cphi, beta, ldc, wavelength):
    dg = zeros(3)
    dc = zeros(3)

    z = z_s(x, y, rstar, f, sphi, cphi)
    if z < 0.0:
        return nan
    else:
        mu = z/sqrt((x**2 + y**2 + z**2))

        dg[0] = x
        dg[1] = y*cphi + z*sphi
        dg[2] = -y*sphi + z*cphi

        dc[0] = dg[0]
        dc[2] = dg[2]

        # Direction vector lengths
        # ------------------------
        lg2 = dg[0]**2 + dg[1]**2 + dg[2]**2
        lg = sqrt(lg2)
        lc = sqrt(dc[0]**2 + dc[2]**2)

        # Normalize the direction vectors
        # -------------------------------
        dg /= lg
        dc /= lc

        gg = - G*mstar/lg2
        gc = ostar*ostar*lc

        dgg = gg*dg + gc*dc
        g = sqrt((dgg**2).sum())
        t = tpole*g**beta/gpole**beta

        return planck(wavelength, t)*(1. - ldc[0]*(1. - mu) - ldc[1]*(1. - mu)**2)


@njit
def luminosity_v(xs, ys, mstar, rstar, ostar, tpole, gpole, f, sphi, cphi, beta, ldc, wavelength):
    npt = xs.size
    l = zeros(npt)
    dg = zeros(3)
    dc = zeros(3)

    for i in range(npt):
        x, y = xs[i], ys[i]
        z = z_s(x, y, rstar, f, sphi, cphi)
        if z < 0.0:
            l[i] = nan
        else:
            mu = z/sqrt((x**2 + y**2 + z**2))

            dg[0] = x
            dg[1] = y*cphi + z*sphi
            dg[2] = -y*sphi + z*cphi

            dc[0] = dg[0]
            dc[2] = dg[2]

            # Direction vector lengths
            # ------------------------
            lg2 = dg[0]**2 + dg[1]**2 + dg[2]**2
            lg = sqrt(lg2)
            lc = sqrt(dc[0]**2 + dc[2]**2)

            # Normalize the direction vectors
            # -------------------------------
            dg /= lg
            dc /= lc

            gg = - G*mstar/lg2
            gc = ostar*ostar*lc

            dgg = gg*dg + gc*dc
            g = sqrt((dgg**2).sum())
            t = tpole*g**beta/gpole**beta

            l[i] = planck(wavelength, t)*(1. - ldc[0]*(1. - mu) - ldc[1]*(1. - mu)**2)
    return l


def create_star_xy(res: int = 64):
    st = linspace(-1., 1., res)
    x, y = meshgrid(st, st)
    return st, x.ravel(), y.ravel()


def create_planet_xy(res: int = 6):
    dd = 2/(res + 1)
    dt = dd*arange(1, res + 1) - 1
    xs, ys = meshgrid(dt, dt)
    m = sqrt(xs**2 + ys**2) <= 1.
    return xs[m], ys[m]


@njit
def create_star_luminosity(res, x, y, mstar, rstar, ostar, tpole, gpole, f, feff, sphi, cphi, beta, ldc, wavelength):
    l = luminosity_v(x*rstar, y*rstar, mstar, rstar, ostar, tpole, gpole, f, sphi, cphi, beta, ldc, wavelength)
    return l.reshape((res, res))


@njit(cache=False, fastmath=False)
def mean_luminosity(xc, yc, k, xs, ys, feff, lt, xt, yt):
    ns = xs.size
    rf = 1.0/(1.0 - feff)

    lsum = 0.0
    weight = 0.0
    for i in range(ns):
        x = xc + xs[i]*k
        y = (yc + ys[i]*k)*rf

        if x**2 + y**2 > 1.0:
            continue

        dx = xt[1] - xt[0]
        dy = yt[1] - yt[0]

        ix = int(floor((x - xt[0])/dx))
        ax1 = (x - xt[ix])/dx
        ax2 = 1.0 - ax1

        iy = int(floor((y - yt[0])/dy))
        ay1 = (y - yt[iy])/dy
        ay2 = 1.0 - ay1

        l = (  lt[iy,     ix    ]*ay2*ax2
             + lt[iy + 1, ix    ]*ay1*ax2
             + lt[iy,     ix + 1]*ay2*ax1
             + lt[iy + 1, ix + 1]*ay1*ax1)

        if isfinite(l):
            lsum += l
            weight += 1.

    if weight > 0.:
        return lsum/weight
    else:
        return nan


@njit
def mean_luminosity_under_planet(x, y, mstar, rstar, ostar, tpole, gpole, f, feff, sphi, cphi, beta, ldc, wavelength):
    l = luminosity_v(x*rstar, y*rstar, mstar, rstar, ostar, tpole, gpole, f, sphi, cphi, beta, ldc, wavelength)
    return nanmean(l)


@njit
def calculate_luminosity_interpolation_table(res, k, xp, yp, sa, ca, y0, vx, vy, ax, ay, jx, jy, sx, sy,
                                             mstar, rstar, ostar, tpole, gpole, f, feff, sphi, cphi, beta, ldc,
                                             wavelength):
    t1 = find_contact_point(k, 1, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    t4 = find_contact_point(k, 4, y0, vx, vy, ax, ay, jx, jy, sx, sy)
    times = linspace(t1, t4, res)

    lt = zeros_like(times)
    for i in range(lt.size):
        x, y = xy_taylor_st(times[i], sa, ca, y0, vx, vy, ax, ay, jx, jy, sx, sy)
        xs = x + k * xp
        ys = y + k * yp
        lt[i] = mean_luminosity_under_planet(xs, ys, mstar, rstar, ostar, tpole, gpole, f, feff, sphi, cphi, beta, ldc,
                                             wavelength)

    i = 0
    while isnan(lt[i]):
        i += 1

    ifill = lt[i]
    while i >= 0:
        lt[i] = ifill
        i -= 1

    i = lt.size - 1
    while isnan(lt[i]):
        i -= 1

    ifill = lt[i]
    while i < lt.size:
        lt[i] = ifill
        i += 1

    return times, lt


@njit(fastmath=True)
def xy_taylor_st(t, sa, ca, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    px =      vx*t + 0.5*ax*t2 + jx*t3/6.0 + sx*t4/24.
    py = y0 + vy*t + 0.5*ay*t2 + jy*t3/6.0 + sy*t4/24.

    x = ca*px - sa*py
    y = ca*py + sa*px

    return x, y


@njit(fastmath=True)
def xy_taylor_vt(ts, a, y0, vx, vy, ax, ay, jx, jy, sx, sy):
    npt = ts.size
    x, y = zeros(npt), zeros(npt)
    ca, sa = cos(a), sin(a)

    for i in range(npt):
        t = ts[i]
        t2 = t*t
        t3 = t2*t
        t4 = t3*t
        px =      vx*t + 0.5*ax*t2 + jx*t3/6.0 + sx*t4/24.
        py = y0 + vy*t + 0.5*ay*t2 + jy*t3/6.0 + sy*t4/24.

        x[i] = ca*px - sa*py
        y[i] = ca*py + sa*px

    return x, y


@njit
def oblate_model_s(t, k, t0, p, a, aa, i, e, w, ldc,
                   mstar, rstar, ostar, tpole, gpole,
                   f, feff, sphi, cphi, beta, wavelength,
                   tres, ts, xs, ys, xp, yp,
                   lcids, pbids, nsamples, exptimes, npb):
    y0, vx, vy, ax, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, i, e, w)

    sa, ca = sin(aa), cos(aa)
    half_window_width = fmax(0.125, (2.0 + k[0])/vx)

    npt = t.size
    flux = zeros(npt)

    tp, lp = calculate_luminosity_interpolation_table(tres, k[0], xp, yp, sa, ca,
                                                      y0, vx, vy, ax, ay, jx, jy, sx, sy,
                                                      mstar, rstar, ostar, tpole, gpole, f, feff,
                                                      sphi, cphi, beta, ldc, wavelength)
    dtp = tp[1] - tp[0]

    ls = create_star_luminosity(ts.size, xs, ys, mstar, rstar, ostar, tpole, gpole,
                                f, feff, sphi, cphi, beta, ldc, wavelength)

    astar = pi * (1. - feff)      # Area of an ellipse = pi * a * b, where a = 1 and b = (1 - feff)
    istar = astar * nanmean(ls)

    for j in range(npt):
        epoch = floor((t[j] - t0 + 0.5*p)/p)
        tc = t[j] - (t0 + epoch*p)
        if abs(tc) > half_window_width:
            flux[j] = 1.0
        else:
            ilc = lcids[j]
            ipb = pbids[ilc]

            if k.size == 1:
                _k = k[0]
            else:
                _k = k[ipb]

            if isnan(_k) or isnan(a):
                flux[j] = inf
            else:
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)

                    to = tc + time_offset
                    it = int(floor((to - tp[0]) / dtp))
                    if it < 0:
                        it = 0
                        at = 0.0
                    elif it > lp.size - 2:
                        it = lp.size - 2
                        at = 1.0
                    else:
                        at = (to - tp[it]) / dtp
                    ml = (1.0 - at) * lp[it] + at * lp[it + 1]

                    x, y = xy_taylor_st(to, sa, ca, y0, vx, vy, ax, ay, jx, jy, sx, sy)

                    b = sqrt(x**2 + (y / (1. - feff))**2)
                    ia = circle_circle_intersection_area(1., _k, b)
                    flux[j] += (istar - ml * ia) / istar

                flux[j] /= nsamples[ilc]
    return flux


def map_osm(rstar, rho, rperiod, tpole, phi):
    omega = 2*pi/(rperiod*d2sec)  # Stellar rotation rate [rad/s]
    rho = 1e3*rho  # Stellar density       [kg/m^3]

    f = stellar_oblateness(omega, rho)  # Stellar oblateness
    feff = 1 - sqrt((1 - f)**2*cos(phi)**2 + sin(phi)**2)  # Projected stellar oblateness
    mstar = rho*4*pi/3*rstar**2*rstar*(1 - f)  # Stellar mass [kg]
    gpole = G*mstar/(rstar*(1 - f))**2  # Surface gravity at the pole  [m/s^2]
    return mstar, omega, gpole, f, feff