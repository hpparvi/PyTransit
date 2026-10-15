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

from matplotlib.patches import Circle
from matplotlib.pyplot import subplots, setp
from numpy import abs as npabs, array, asarray, cos, empty, linspace, minimum, sin, sqrt, zeros

from pytransit import QuadraticModel, RoadRunnerModel
from pytransit.models.roadrunner.common import (circle_circle_intersection_area, cubic_coefficients,
                                                g_nodes, ldm_lookup, ldm_nodes, ldm_table,
                                                planet_angular_extent, profile_grid, quadrature_rules,
                                                split_cubic_coefficients, split_point)

__all__ = ['plot_laws', 'plot_profile_and_transit', 'plot_accuracy', 'plot_quadrature_nodes',
           'plot_mean_intensity_table']

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


def _quadratic_profile_table(ldc=(0.3, 0.1)):
    """The quadratic law tabulated on the fixed profile grid, as the model tabulates it."""
    mu, t0, dt = profile_grid()
    return t0, dt, RoadRunnerModel.ldmodels['quadratic'][0](mu, array(ldc, float))


def plot_quadrature_nodes(k=0.3, bs=(0.45, 0.12, 0.9), nq=8):
    """Draw the quadrature nodes the model uses for three planet positions.

    Each node is a ring of constant distance z from the disk centre, so it is drawn as the arc of
    that ring lying under the planet, with its width proportional to the node's weight in the
    mean intensity. The three positions cover the three ways the model sets up the integral: the
    planet fully on the disk, the planet covering the disk centre, and the planet crossing the limb.
    Each panel is centred on the planet, and the cross marks the disk centre.

    Returns
    -------
    matplotlib.figure.Figure
    """
    titles = ('Planet on the disk', 'Planet over the disk centre', 'Planet crossing the limb')
    rules = quadrature_rules(nq)
    half_width = 1.4 * k

    fig, axs = subplots(1, 3, figsize=(9, 3.4))
    for ax, b, title in zip(axs, bs, titles):
        mu = empty((1, 2 * nq))
        wf = zeros((1, 2 * nq))
        ldm_nodes(k, array([b / (1.0 + k)]), rules, mu, wf)
        wmax = wf.max()

        ax.add_patch(Circle((0, 0), 1.0, fc='0.94', ec='0.6', lw=1))
        ax.add_patch(Circle((b, 0), k, fc='none', ec='k', lw=1, ls='--'))
        for m, w in zip(mu[0], wf[0]):
            if w == 0.0:
                continue
            z = sqrt(1.0 - m * m)
            half = 0.5 * planet_angular_extent(z, b, k)
            phi = linspace(-half, half, 200)
            ax.plot(z * cos(phi), z * sin(phi), c='C0', lw=0.5 + 3.0 * w / wmax,
                    solid_capstyle='butt')

        ax.plot(0, 0, '+', c='0.4', ms=8)
        ax.set_title(title, fontsize='medium')
        ax.set_aspect('equal')
        ax.set_xlim(b - half_width, b + half_width)
        ax.set_ylim(-half_width, half_width)
        ax.axis('off')
    fig.tight_layout()
    return fig


def _uniform_cubic_lookup(g, coef):
    """Evaluate a cubic table on a uniform g grid from 0 to 1 (the unsplit alternative)."""
    n = coef.shape[0] + 1
    x = g * (n - 1)
    i = minimum(x.astype(int), n - 2)
    a = x - i
    c = coef[i]
    return c[:, 0] + a * (c[:, 1] + a * (c[:, 2] + a * c[:, 3]))


def plot_mean_intensity_table(k=0.2, ng=100, nq=8, ldc=(0.3, 0.1)):
    """Draw the mean intensity table against the grazing parameter, and its interpolation error.

    The table and the reference curve use the same quadrature, so the lower panel shows the error
    of the table interpolation alone: the split table beside a single uniform table of the same
    size, whose cubic has to bridge the kink at the limb contact. The error is shown in flux, the
    mean intensity error times the overlap area over the disk-integrated intensity, because a
    large mean intensity error where the planet barely overlaps the star costs almost nothing.
    The dotted line marks the planet's edge crossing the disk centre, a weaker kink that the table
    does not split at.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t0, dt, ldp = _quadratic_profile_table(ldc)
    rules = quadrature_rules(nq)

    def table(gs):
        mu = empty((gs.size, 2 * nq))
        wf = zeros((gs.size, 2 * nq))
        ldm_nodes(k, gs, rules, mu, wf)
        out = empty(gs.size)
        ldm_table(mu, wf, t0, dt, ldp, out)
        return out

    gc = split_point(k)
    gf = linspace(0.0, 0.999, 2000)
    reference = table(gf)

    gs, n1 = g_nodes(k, ng)
    tab = table(gs)
    coef = empty((ng - 2, 4))
    split_cubic_coefficients(tab, n1, coef)
    split = array([ldm_lookup(g, gc, n1, coef) for g in gf])

    gu = linspace(0.0, 1.0, ng)
    uniform = _uniform_cubic_lookup(gf, cubic_coefficients(table(gu)))

    ldi = RoadRunnerModel.ldmodels['quadratic'][1](array(ldc, float))
    scale = 1e6 * array([circle_circle_intersection_area(1.0, k, g * (1.0 + k)) for g in gf]) / ldi

    fig, (at, ab) = subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=(3, 2))
    at.plot(gf, reference, 'k', lw=1)
    at.plot(gs, tab, 'o', c='C0', ms=3, label=f'Table nodes (ng = {ng})')
    ab.plot(gf, scale * npabs(split - reference), c='C0', zorder=2, label='Split at the limb contact')
    ab.plot(gf, scale * npabs(uniform - reference), c='C1', zorder=1, label='Single uniform table')

    gz = k / (1.0 + k)
    for ax in (at, ab):
        ax.axvline(gc, c='0.6', ls='--', lw=1, zorder=-1)
        ax.axvline(gz, c='0.6', ls=':', lw=1, zorder=-1)
        _despine(ax)
    at.text(gc, reference.max(), '  limb contact', color='0.4', fontsize='small', va='top')
    at.text(gz, 0.5 * (reference.min() + reference.max()), '  edge over centre', color='0.4',
            fontsize='small', va='center')
    setp(at, ylabel='Mean intensity under planet', title=f'Mean intensity table, k = {k}')
    setp(ab, xlabel='Grazing parameter $g = b\\,/\\,(1 + k)$', ylabel='Flux error [ppm]',
         yscale='log', xlim=(0, 1))
    at.legend(fontsize='small', frameon=False, loc='lower left')
    ab.legend(fontsize='small', frameon=False, loc='upper left')
    fig.tight_layout()
    return fig
