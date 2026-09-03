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

"""Plotting helpers for the RoadRunner feature page.

The page is about the separation between the stellar intensity profile and the transit geometry,
so every figure shows the profile beside the light curve it produces. Keeping the plotting here
lets the page show only the PyTransit calls.
"""

from matplotlib.pyplot import subplots, setp
from numpy import abs as npabs, array, asarray, linspace, sqrt

from pytransit import QuadraticModel, RoadRunnerModel

__all__ = ['plot_laws', 'plot_profile_and_transit', 'plot_accuracy']

# The example transit used throughout the page.
WINDOW = 3.5 / 24
TIME = linspace(-0.5 * WINDOW, 0.5 * WINDOW, 1500)
K, T0, P, A, INC = 0.1, 0.0, 4.0, 13.0, 0.49 * 3.141592653589793


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _profile(fn, pv, n=1000):
    """Sample an intensity profile over the stellar disk, from centre to limb."""
    z = linspace(0.0, 1.0, n)
    return z, fn(sqrt(1.0 - z ** 2), asarray(pv, float))


def plot_laws(laws, fluxes, time=TIME):
    """Draw several built-in limb darkening profiles beside the transits they produce.

    Parameters
    ----------
    laws : dict
        Mapping from a built-in law name to its coefficients.
    fluxes : dict
        Mapping from the same names to the modelled fluxes.
    time : ndarray, optional
        The mid-exposure times the fluxes were computed for.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (al, ar) = subplots(1, 2, figsize=(9, 3.6))
    for name, ldc in laws.items():
        z, profile = _profile(RoadRunnerModel.ldmodels[name][0], ldc)
        al.plot(z, profile, label=name)
        ar.plot(24 * time, fluxes[name], label=name)

    setp(al, xlabel='Distance from disk centre $z$ [$R_\\star$]', ylabel='Stellar intensity',
         title='Intensity profile')
    setp(ar, xlabel='Time - $t_c$ [h]', ylabel='Normalised flux', title='Resulting transit')
    ar.autoscale(axis='x', tight=True)
    al.legend(fontsize='small', frameon=False)
    _despine(al)
    _despine(ar)
    fig.tight_layout()
    return fig


def plot_profile_and_transit(profile, pv, flux, time=TIME, title=''):
    """Draw one intensity profile beside the transit it produces.

    Parameters
    ----------
    profile : callable
        The intensity profile, ``f(mu, pv)``.
    pv : array-like
        The profile's coefficients.
    flux : ndarray
        The modelled flux.
    time : ndarray, optional
        The mid-exposure times the flux was computed for.
    title : str, optional
        Title for the profile panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    z, values = _profile(profile, pv)

    fig, (al, ar) = subplots(1, 2, figsize=(9, 3.6))
    al.plot(z, values, 'k')
    ar.plot(24 * time, flux, 'k')

    setp(al, xlabel='Distance from disk centre $z$ [$R_\\star$]', ylabel='Stellar intensity',
         title=title or 'Intensity profile')
    setp(ar, xlabel='Time - $t_c$ [h]', ylabel='Normalised flux', title='Resulting transit')
    ar.autoscale(axis='x', tight=True)
    _despine(al)
    _despine(ar)
    fig.tight_layout()
    return fig


def plot_accuracy(ks=(0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22), ldc=(0.3, 0.1)):
    """Compare RoadRunner against the analytic Mandel & Agol model as a function of radius ratio.

    :class:`~pytransit.models.ma_quadratic.QuadraticModel` is the exact analytic solution for the
    quadratic law, so the difference is RoadRunner's discretisation error alone.

    Returns
    -------
    matplotlib.figure.Figure
    """
    settings = {'Defaults (nq = 8, ng = 100)': {},
                'nq = 12, ng = 200': dict(nq=12, ng=200)}

    reference = QuadraticModel()
    reference.set_data(TIME)

    fig, ax = subplots(figsize=(9, 4))
    for label, kwargs in settings.items():
        tm = RoadRunnerModel('quadratic', **kwargs)
        tm.set_data(TIME)
        errors = [1e6 * npabs(tm.evaluate(k, list(ldc), T0, P, A, INC)
                              - reference.evaluate(k, list(ldc), T0, P, A, INC)).max()
                  for k in ks]
        ax.plot(ks, errors, 'o-', label=label)

    ax.axhline(1.0, ls='--', c='0.75', zorder=-1)
    ax.text(ks[0], 1.05, '1 ppm', color='0.5', fontsize='small', va='bottom')
    setp(ax, xlabel='Radius ratio $k$', ylabel='Maximum deviation [ppm]', yscale='log',
         title='Deviation from the analytic Mandel & Agol model')
    ax.legend(fontsize='small', frameon=False)
    _despine(ax)
    fig.tight_layout()
    return fig
