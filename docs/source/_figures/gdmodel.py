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

"""Plotting helpers for the gravity-darkened star feature page.

The page is about what a gravity-darkened star does to a transit that an ordinary limb-darkened
star cannot. The figure therefore shows the stellar disk with the planet's path beside the
residuals left after fitting a standard transit model.
"""

from math import cos, sin

from matplotlib.pyplot import subplots, setp
from numpy import arccos, atleast_2d, degrees, nanmax
from scipy.optimize import least_squares

from meepmeep.backends.numba.point2d import solve2d

from pytransit import RoadRunnerModel
from pytransit.models.numba.gdmodel import create_star_luminosity, create_star_xy, map_osm, xy_taylor_vt

__all__ = ['plot_gravity_darkening_signal']


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _standard_fit(time, flux, p, a0):
    """The light curve of the best-fitting spherical, limb-darkened star.

    The radius ratio, scaled semi-major axis, impact parameter, transit centre, and quadratic limb
    darkening coefficients are all free, so whatever remains is what gravity darkening adds.
    """
    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)

    def residuals(x):
        k, a, b, t0, u1, u2 = x
        return tm.evaluate(k, [u1, u2], t0, p, a, arccos(b / a)) - flux

    bounds = ([0.03, 2.0, 0.0, -0.01, -1.0, -1.0], [0.3, 20.0, 0.95, 0.01, 2.0, 2.0])
    best = None
    for bs in (0.05, 0.3, 0.5, 0.7):
        fit = least_squares(residuals, [0.1, a0, bs, 0.0, 0.3, 0.2], bounds=bounds, x_scale='jac',
                            xtol=1e-14, ftol=1e-14)
        if best is None or fit.cost < best.cost:
            best = fit
    return flux + best.fun          # `fun` is the model minus the data


def _draw_star(ax, tm, time, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l, res=300):
    """Draw the gravity-darkened stellar disk and the planet's path, as `evaluate` sees them."""
    mstar, ostar, gpole, f, feff = map_osm(tm.rstar, rho, rperiod, phi)
    st, sx, sy = create_star_xy(res)
    image = create_star_luminosity(res, sx, sy, mstar, tm.rstar, ostar, tpole, gpole, f, sin(phi),
                                   cos(phi), beta, atleast_2d(ldc), tm._flux_table, tm._teff0,
                                   tm._dteff, tm.accurate_mu)[0]
    ax.imshow(image / nanmax(image), extent=(-1, 1, -1, 1), origin='lower', cmap='inferno',
              vmin=0.4, vmax=1.0)
    px, py = xy_taylor_vt(time - t0, l, solve2d(0.0, p, a, i, 0.0, 0.0))
    ax.plot(px, py, c='w', lw=1, ls='--')
    # An arrowhead where the path leaves the star shows the direction of motion.
    j = max(n for n in range(px.size) if px[n] ** 2 + py[n] ** 2 < 1.0)
    ax.annotate('', xy=(px[j], py[j]), xytext=(px[j - 15], py[j - 15]),
                arrowprops=dict(arrowstyle='-|>', color='w', lw=1))
    setp(ax, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    ax.set_aspect('equal')
    ax.axis('off')


def plot_gravity_darkening_signal(time, fluxes, models, rho, rperiod, tpole, phi, beta, ldc,
                                  t0, p, a, i):
    """Draw the stellar disk and the residuals of a standard transit model fit for several orbits.

    Parameters
    ----------
    time : ndarray
        The mid-exposure times the fluxes were computed for.
    fluxes : dict
        Mapping from ``(band, l)`` to the modelled flux, where `band` names a passband and `l` is
        the projected spin-orbit angle in radians.
    models : dict
        Mapping from the passband names to the `GravityDarkenedModel` instances used. The stellar
        disk is drawn in the first passband.
    rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i
        The parameters the fluxes were computed with.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bands = list(models)
    ls = sorted({l for _, l in fluxes})
    colours = {'blue': 'C0', 'red': 'C3'}

    fig, axs = subplots(2, len(ls), figsize=(9, 5.4), height_ratios=(2, 3), sharey='row')
    for l, at, ab in zip(ls, axs[0], axs[1]):
        _draw_star(at, models[bands[0]], time, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l)
        at.set_title(f'$\\lambda$ = {degrees(l):.0f}°')
        for band in bands:
            flux = fluxes[band, l]
            residual = flux - _standard_fit(time, flux, p, a)
            ab.plot(24 * time, 1e6 * residual, c=colours.get(band), label=band)
        ab.axhline(0.0, c='0.8', lw=0.5, zorder=-1)
        ab.autoscale(axis='x', tight=True)
        setp(ab, xlabel='Time - $t_c$ [h]')
        _despine(ab)
    setp(axs[1, 0], ylabel='Residual of a standard fit [ppm]')
    axs[1, 0].legend(fontsize='small', frameon=False, loc='lower left')
    fig.tight_layout()
    return fig
