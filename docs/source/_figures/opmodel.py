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

"""Plotting helpers for the oblate planet feature page.

The page is about how small the oblateness signal is, how much of it a spherical planet can absorb,
and how accurate the model must be to see the rest. The figures therefore compare an oblate planet
with the best-fitting spherical one, and the model accuracy settings with a direct numerical
integration of the stellar disk.
"""

import warnings
from math import cos, sin, sqrt

from matplotlib.patches import Circle, Ellipse
from matplotlib.pyplot import subplots, setp
from numpy import abs as npabs, arccos, arcsin, array, degrees, linspace, pi
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import least_squares

from meepmeep.backends.numba.point2d import pos_c, solve2d

from pytransit import OPModel, RoadRunnerModel

__all__ = ['plot_oblateness_signal', 'plot_accuracy_levels']

# The example transit used throughout the page: a Saturn-like planet on a ten-day orbit.
P, A = 10.0, 15.0
LDC = (0.4, 0.25)


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _sphere_fit(time, flux, k0, ldc):
    """The spherical planet that best reproduces a light curve, with the limb darkening fixed.

    The radius ratio, scaled semi-major axis, impact parameter, and transit centre are free. The
    fit starts from several impact parameters, since an elongated planet can mimic a spherical one
    on a quite different chord.
    """
    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)

    def residuals(x):
        k, a, b, t0 = x
        return tm.evaluate(k, ldc, t0, P, a, arccos(b / a)) - flux

    bounds = ([0.5 * k0, 5.0, 0.0, -0.01], [2 * k0, 40.0, 0.95, 0.01])
    best = None
    for bs in (0.05, 0.3, 0.5, 0.7, 0.85):
        fit = least_squares(residuals, [k0, A, bs, 0.0], bounds=bounds, x_scale='jac',
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if best is None or fit.cost < best.cost:
            best = fit
    return tm.evaluate(best.x[0], ldc, best.x[3], P, best.x[1], arccos(best.x[2] / best.x[1]))


def _draw_geometry(ax, b, alpha, k=0.3, f=0.3):
    """Sketch the planet crossing the stellar limb, with its shape exaggerated so that it is visible."""
    ax.add_patch(Circle((0, 0), 1.0, fc='0.94', ec='0.5', lw=0.8))
    xyc = solve2d(0.0, P, A, arccos(b / A), 0.0, 0.0)
    xs = array([pos_c(t, xyc) for t in linspace(-0.4, 0.4, 200)])
    ax.plot(xs[:, 0], xs[:, 1], c='0.6', lw=0.8, ls='--')
    yc = pos_c(0.0, xyc)[1]
    for x in (-sqrt(1 - yc ** 2), 0.0, sqrt(1 - yc ** 2)):
        ax.add_patch(Ellipse((x, yc), 2 * k, 2 * k * (1 - f), angle=degrees(alpha), fc='k', ec='none'))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_oblateness_signal(time, fluxes, k, f, b, ldc=LDC):
    """Draw the oblate planet's light curve against an equal-area and a best-fitting spherical planet.

    Parameters
    ----------
    time : ndarray
        The mid-exposure times the fluxes were computed for.
    fluxes : dict
        Mapping from the projected obliquity in radians to the oblate planet's flux.
    k, f, b : float
        The radius ratio, flattening, and impact parameter the fluxes were computed with.
    ldc : tuple, optional
        The quadratic limb darkening coefficients the fluxes were computed with.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ldc = list(ldc)
    rr = RoadRunnerModel('quadratic')
    rr.set_data(time)
    keq = k * sqrt(1.0 - f)
    equal_area = rr.evaluate(keq, ldc, 0.0, P, A, arccos(b / A))

    n = len(fluxes)
    fig, axs = subplots(2, n, figsize=(9, 5), height_ratios=(2, 3), sharey='row')
    for (alpha, flux), at, ab in zip(fluxes.items(), axs[0], axs[1]):
        _draw_geometry(at, b, alpha)
        at.set_title(f'$\\alpha$ = {degrees(alpha):.0f}°')
        fitted = _sphere_fit(time, flux, keq, ldc)
        ab.plot(24 * time, 1e6 * (flux - equal_area), c='0.6', ls='--', label='Equal-area sphere')
        ab.plot(24 * time, 1e6 * (flux - fitted), c='k', label='Best-fitting sphere')
        ab.axhline(0.0, c='0.8', lw=0.5, zorder=-1)
        ab.autoscale(axis='x', tight=True)
        setp(ab, xlabel='Time - $t_c$ [h]')
        _despine(ab)
    setp(axs[1, 0], ylabel='Oblate - spherical [ppm]')
    axs[1, 0].legend(fontsize='small', frameon=False, loc='lower left')
    fig.tight_layout()
    return fig


def _intensity(x, y, u1, u2):
    r2 = x * x + y * y
    if r2 >= 1.0:
        return 0.0
    mu = sqrt(1.0 - r2)
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


def _reference_flux(cx, cy, k, f, alpha, u1, u2):
    """Oblate planet transit flux from nested adaptive quadrature over the exact overlap region.

    The same reference the test suite validates the model against (``tests/test_opmodel.py``).
    """
    bb = 1.0 - f
    ca, sa = cos(alpha), sin(alpha)
    hh = k * sqrt(sa * sa + bb * bb * ca * ca)

    def scanline(y):
        if abs(y) >= 1.0:
            return 0.0
        yp = (y - cy) / k
        d = bb * bb * ca * ca + sa * sa - yp * yp
        if d <= 0.0:
            return 0.0
        sd = sqrt(d) / bb
        u = yp * ca * sa * (1.0 / (bb * bb) - 1.0)
        v = sa * sa / (bb * bb) + ca * ca
        w = sqrt(1.0 - y * y)
        lo = max(cx + k * (u - sd) / v, -w)
        hi = min(cx + k * (u + sd) / v, w)
        if hi <= lo:
            return 0.0
        return quad(_intensity, lo, hi, args=(y, u1, u2), epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    ylo, yhi = max(cy - hh, -1.0), min(cy + hh, 1.0)
    blocked = quad(scanline, ylo, yhi, epsabs=1e-12, epsrel=1e-11, limit=500)[0] if yhi > ylo else 0.0
    return 1.0 - blocked / (pi * (1.0 - u1 / 3.0 - u2 / 6.0))


def plot_accuracy_levels(k=0.1, f=0.1, alpha=pi / 4, bs=(0.5, 1.0), ldc=LDC):
    """Compare the model's accuracy settings against direct numerical integration of the stellar disk.

    Returns
    -------
    matplotlib.figure.Figure
    """
    settings = {'Default': {},
                '``exact_ld``': dict(exact_ld=True),
                '``exact_ld`` and ``exact_areas``': dict(exact_ld=True, exact_areas=True)}
    fig, axs = subplots(1, len(bs), figsize=(9, 3.6), sharey=True)
    for b, ax in zip(bs, axs):
        # From mid-transit to slightly past the last contact, which is where all the error lives.
        half_duration = P / (2 * pi) * arcsin(sqrt((1 + k) ** 2 - b ** 2) / A)
        time = linspace(0.0, 1.05 * half_duration, 150)
        tm = OPModel('quadratic')
        tm.set_data(time)

        inc = arccos(b / A)
        xyc = solve2d(0.0, P, A, inc, 0.0, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', IntegrationWarning)
            reference = array([_reference_flux(*pos_c(t, xyc), k, f, alpha, *ldc) for t in time])
        for label, kwargs in settings.items():
            flux = tm.evaluate(k, f, alpha, list(ldc), 0.0, P, A, inc, **kwargs)
            ax.plot(24 * time, 1e6 * npabs(flux - reference), label=label.replace('``', ''))
        ax.axhline(1.0, ls='--', c='0.75', zorder=-1)
        setp(ax, xlabel='Time - $t_c$ [h]', yscale='log', ylim=(1e-3, 1e2),
             title=f'Impact parameter {b:g}' + (' (grazing)' if b + k >= 1.0 else ''))
        ax.autoscale(axis='x', tight=True)
        _despine(ax)
    setp(axs[0], ylabel='Model error [ppm]')
    axs[0].legend(fontsize='small', frameon=False, loc='upper left')
    fig.tight_layout()
    return fig
