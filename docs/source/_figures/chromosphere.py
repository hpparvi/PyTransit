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

"""Plotting helpers for the chromosphere feature page.

The page is about a limb-brightened stellar disk, so the figures show its intensity profile beside
the transit it produces, and how far the best limb darkening laws fall from reproducing it.
"""

from matplotlib.pyplot import subplots, setp
from numpy import arccos, linspace, sqrt
from scipy.optimize import least_squares

from pytransit import RoadRunnerModel

__all__ = ['plot_profiles_and_transits', 'plot_limb_darkening_fits']

P, A = 4.0, 13.0


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_profiles_and_transits(time, fluxes, ldc):
    """Draw the shell, uniform, and quadratic intensity profiles beside the transits they produce.

    The profiles are normalised to the same disk-integrated flux, so that they compare the
    distribution of the light rather than its total.

    Parameters
    ----------
    time : ndarray
        The mid-exposure times the fluxes were computed for.
    fluxes : dict
        The fluxes for the keys 'shell', 'uniform', and 'quadratic'.
    ldc : list
        The quadratic limb darkening coefficients the quadratic flux was computed with.

    Returns
    -------
    matplotlib.figure.Figure
    """
    r = linspace(0.0, 0.995, 500)
    mu = sqrt(1.0 - r ** 2)
    u1, u2 = ldc
    profiles = {'shell': 0.5 / mu,
                'uniform': 1.0 + 0.0 * mu,
                'quadratic': ((1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2)
                              / (1.0 - u1 / 3.0 - u2 / 6.0))}
    labels = {'shell': 'Optically thin shell', 'uniform': 'Uniform disk',
              'quadratic': 'Limb-darkened photosphere'}
    styles = {'shell': dict(c='k'), 'uniform': dict(c='0.6', ls=':'),
              'quadratic': dict(c='C0', ls='--')}

    fig, (al, ar) = subplots(1, 2, figsize=(9, 3.6))
    for key in ('shell', 'uniform', 'quadratic'):
        al.plot(r, profiles[key], label=labels[key], **styles[key])
        ar.plot(24 * time, fluxes[key], label=labels[key], **styles[key])
    setp(al, xlabel='Distance from disk centre $z$ [$R_\\star$]', ylabel='Relative intensity',
         ylim=(0, 3),
         title='Intensity profile')
    setp(ar, xlabel='Time - $t_c$ [h]', ylabel='Normalised flux', title='Resulting transit')
    ar.autoscale(axis='x', tight=True)
    al.legend(fontsize='small', frameon=False, loc='upper left')
    _despine(al)
    _despine(ar)
    fig.tight_layout()
    return fig


def _best_fit(time, flux, law, ldc0):
    """The best fit of a limb-darkened transit to `flux`, with every parameter free.

    The limb darkening coefficients have wide bounds, so a law is free to brighten towards the limb
    as far as its functional form allows.
    """
    tm = RoadRunnerModel(law)
    tm.set_data(time)
    n = len(ldc0)

    def residuals(x):
        k, a, b, t0 = x[:4]
        return tm.evaluate(k, x[4:], t0, P, a, arccos(b / a)) - flux

    bounds = ([0.03, 3.0, 0.0, -0.01] + [-50.0] * n, [0.3, 40.0, 0.95, 0.01] + [50.0] * n)
    best = None
    for bs in (0.05, 0.3, 0.5):
        fit = least_squares(residuals, [0.1, A, bs, 0.0] + list(ldc0), bounds=bounds, x_scale='jac',
                            xtol=1e-14, ftol=1e-14)
        if best is None or fit.cost < best.cost:
            best = fit
    return best.fun


def plot_limb_darkening_fits(time, flux):
    """Draw the residuals left when limb darkening laws are fitted to a limb-brightened transit.

    Returns
    -------
    matplotlib.figure.Figure
    """
    laws = {'quadratic': [-0.3, 0.0], 'power-2': [-0.5, 1.0], 'nonlinear': [0.0, 0.0, 0.0, -0.5]}
    fig, ax = subplots(figsize=(9, 3.4))
    for law, ldc0 in laws.items():
        residual = -_best_fit(time, flux, law, ldc0)            # data minus model
        ax.plot(24 * time, 1e6 * residual, label=f"'{law}'")
    ax.axhline(0.0, c='0.8', lw=0.5, zorder=-1)
    setp(ax, xlabel='Time - $t_c$ [h]', ylabel='Shell - best fit [ppm]',
         title='Residuals of the best-fitting limb-darkened transits')
    ax.autoscale(axis='x', tight=True)
    ax.legend(fontsize='small', frameon=False, loc='lower center', ncols=3)
    _despine(ax)
    fig.tight_layout()
    return fig
