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

"""Plotting helpers for the transmission spectroscopy pages.

The pages are about many wavelength bins sharing one transit geometry, so the figures show a whole
spectroscopic dataset at once and measure what sharing the geometry costs in accuracy. Keeping the
plotting and the toy spectrum here lets the pages show only the PyTransit calls.

`plot_shared_work` illustrates how the model works: the map from a limb darkening profile to a
mean intensity table that all the bins share, the tables it gives for a few bins, and how the
covered area is scaled from the mean planet to a bin's own radius ratio.
"""

from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Polygon
from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import MaxNLocator
from meepmeep.backends.numba.point2d import sep_c, solve2d
from numpy import (abs as npabs, arctan2, array, column_stack, concatenate, cos, empty, exp, geomspace, linspace,
                   pi, sin, stack, tile, where, zeros)

from pytransit import RoadRunnerModel, TSModel
from pytransit.models.limb_darkening import evaluate_ld, evaluate_ldi
from pytransit.models.roadrunner.common import (g_nodes, ldm_lookup, ldm_nodes, ldm_table_banded, ldm_table_map,
                                                split_cubic_coefficients, split_point)

__all__ = ['toy_spectrum', 'plot_spectroscopic_transit', 'plot_accuracy', 'plot_shared_work']

# The example transit used throughout the page.
WINDOW = 3.5 / 24
TIME = linspace(-0.5 * WINDOW, 0.5 * WINDOW, 1500)
T0, P, A, INC = 0.0, 4.0, 13.0, 0.49 * pi


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _gaussian(x, centre, width):
    return exp(-0.5 * ((x - centre) / width) ** 2)


def toy_spectrum(npb, k0=0.1):
    """A toy near-infrared transmission spectrum with wavelength-dependent limb darkening.

    The radius ratio has a weak scattering slope and a few absorption bands loosely placed at the
    water and carbon dioxide features, each changing the radius ratio by one to two percent. The
    quadratic limb darkening weakens towards the red. None of it is meant to be physical; it only
    gives the bins something to differ by.

    Parameters
    ----------
    npb : int
        Number of wavelength bins between 0.6 and 5.0 micron.
    k0 : float, optional
        The continuum radius ratio.

    Returns
    -------
    wl : ndarray
        Bin centres in micron, shape ``(npb,)``.
    k : ndarray
        Radius ratios, shape ``(npb,)``.
    ldc : ndarray
        Quadratic limb darkening coefficients, shape ``(npb, 2)``.
    """
    wl = linspace(0.6, 5.0, npb)
    k = k0 * (1.0 + 0.004 * (0.6 / wl) ** 4
              + 0.010 * _gaussian(wl, 1.40, 0.10)
              + 0.012 * _gaussian(wl, 1.90, 0.12)
              + 0.015 * _gaussian(wl, 2.70, 0.15)
              + 0.020 * _gaussian(wl, 4.30, 0.12))
    ldc = stack([0.15 + 0.35 * (0.6 / wl) ** 0.9, 0.12 + 0.10 * (0.6 / wl)], axis=1)
    return wl, k, ldc


def plot_spectroscopic_transit(wl, k, flux, time=TIME):
    """Draw the transmission spectrum beside the spectroscopic light curves it produces.

    The spectrum is drawn as the radius ratio with its mean over the bins marked, since the model
    shares its geometry through that mean. The light curves are shown relative to their mean over
    wavelength, since the transit itself is the same in every bin and would otherwise hide the
    differences between them.

    Parameters
    ----------
    wl : ndarray
        Bin centres in micron.
    k : ndarray
        Radius ratio per bin.
    flux : ndarray
        Modelled flux with shape ``(npb, npt)``.
    time : ndarray, optional
        The mid-exposure times the flux was computed for.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (al, ar) = subplots(1, 2, figsize=(9, 3.6), width_ratios=(2, 3))
    al.plot(wl, k, 'k')
    al.axhline(k.mean(), ls='--', c='0.6', zorder=-1)
    al.text(wl[-1], k.mean(), '$\\bar{k}$', color='0.4', ha='right', va='bottom')

    differential = 1e6 * (flux - flux.mean(0))
    vmax = npabs(differential).max()
    im = ar.pcolormesh(24 * time, wl, differential, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='nearest', rasterized=True)
    fig.colorbar(im, ax=ar, label='Flux - mean flux [ppm]')

    setp(al, xlabel='Wavelength [$\\mu$m]', ylabel='Radius ratio $k$',
         title='Transmission spectrum')
    setp(ar, xlabel='Time - $t_c$ [h]', ylabel='Wavelength [$\\mu$m]',
         title='Light curves relative to their mean')
    al.autoscale(axis='x', tight=True)
    al.yaxis.set_major_locator(MaxNLocator(5))
    _despine(al)
    fig.tight_layout()
    return fig


def _bin_errors(kmean, deviation, inc, ldc, time):
    """Flux error of the two bins at ``kmean * (1 -/+ deviation)``, against per-bin RoadRunner.

    The third bin sits at the mean so the spread is symmetric and the model's mean radius ratio is
    ``kmean`` exactly.
    """
    k = kmean * array([1.0 - deviation, 1.0, 1.0 + deviation])
    ts = TSModel()
    ts.set_data(time)
    flux = ts.evaluate(k, tile(ldc, (3, 1)), T0, P, A, inc)[0]

    rr = RoadRunnerModel('quadratic')
    rr.set_data(time)
    reference = array([rr.evaluate(kk, ldc, T0, P, A, inc) for kk in k])
    return 1e6 * (flux - reference)[[0, 2]]


def plot_accuracy(kmean=0.1, deviation=0.01, ldc=(0.4, 0.2),
                  ks=(0.05, 0.1, 0.2), deviations=geomspace(0.002, 0.2, 12)):
    """Measure the error of sharing the geometry across bins with different radius ratios.

    The reference is :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` evaluated bin by
    bin with the same discretisation, so the difference is the error of the shared geometry alone,
    not the quadrature error both models have in common.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ldc = list(ldc)
    fig, (al, ar) = subplots(1, 2, figsize=(9, 3.8))

    # Where along the transit the error lives
    # ---------------------------------------
    xyc = solve2d(0.0, P, A, INC, 0.0, 0.0)
    z = array([sep_c(t, xyc) for t in TIME])
    errors = _bin_errors(kmean, deviation, INC, ldc, TIME)
    al.plot(24 * TIME, errors[0], label=f'$k = \\bar{{k}}\\,(1 - {deviation:g})$')
    al.plot(24 * TIME, errors[1], label=f'$k = \\bar{{k}}\\,(1 + {deviation:g})$')
    al.fill_between(24 * TIME, 0, 1, where=(z > 1.0 - kmean) & (z < 1.0 + kmean), color='0.92',
                    transform=al.get_xaxis_transform(), zorder=-1, label='Ingress and egress')
    al.axhline(0.0, c='0.6', lw=0.5, zorder=-1)
    setp(al, xlabel='Time - $t_c$ [h]', ylabel='Flux error [ppm]',
         title=f'$\\bar{{k}} = {kmean:g}$, bins at $\\pm${100 * deviation:g}%')
    al.autoscale(axis='x', tight=True)
    al.legend(fontsize='small', frameon=False, loc='upper center')

    # How the error scales with the spread
    # ------------------------------------
    time = linspace(0.0, 0.5 * WINDOW, 1000)
    for kk in ks:
        ar.plot(100 * deviations, [npabs(_bin_errors(kk, d, INC, ldc, time)).max() for d in deviations],
                'o-', ms=3, label=f'$\\bar{{k}} = {kk:g}$')
    ar.axhline(1.0, ls='--', c='0.75', zorder=-1)
    ar.text(100 * deviations[-1], 1.1, '1 ppm', color='0.5', fontsize='small', va='bottom', ha='right')
    setp(ar, xlabel='Deviation from the mean radius ratio [%]', ylabel='Maximum flux error [ppm]',
         xscale='log', yscale='log', title='Error against the radius ratio spread')
    ar.legend(fontsize='small', frameon=False)

    _despine(al)
    _despine(ar)
    fig.tight_layout()
    return fig


def plot_shared_work(npb=7, kmean=0.1):
    """Illustrate the work the wavelength bins share and the work each bin does on its own.

    The left panel shows the map from a limb darkening profile to a mean intensity table, the
    middle panel the tables the map gives for a few bins of the toy spectrum, and the right panel
    how the covered area changes with the radius ratio while the planet crosses the limb. The map
    and the tables are computed with the model's own code; the right panel is a sketch with the
    planet and the change in its radius drawn much larger than in a real spectrum.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tm = TSModel('quadratic')
    ng, nmu, nq = tm.ng, tm.mu.size, tm._rules.shape[2]
    wl, _, ldc = toy_spectrum(npb)

    # The shared map, built exactly as in the model.
    mu, wf = empty((ng, 2 * nq)), zeros((ng, 2 * nq))
    gs, n1 = g_nodes(kmean, ng)
    ldm_nodes(kmean, gs, tm._rules, mu, wf)
    wt, lo, hi = ldm_table_map(mu, wf, tm._t0, tm._dt, nmu)

    # One mean intensity table per bin, from the same map.
    ldp = evaluate_ld(tm.ldmodel, tm.mu, ldc.reshape((1, npb, 2)))[0]
    istar = evaluate_ldi(tm.ldmmean, ldc.reshape((1, npb, 2)))[0]
    tables = empty((npb, ng))
    for ipb in range(npb):
        ldm_table_banded(wt, lo, hi, ldp[ipb], pi / istar[ipb], tables[ipb])

    cmap = colormaps['plasma']
    colours = cmap(Normalize(wl[0], wl[0] + (wl[-1] - wl[0]) / 0.88)(wl))

    fig, axs = subplots(1, 3, figsize=(9, 3.4))

    ax = axs[0]
    order = gs.argsort()
    # Only the pattern matters here: which profile nodes each table node reads.
    ax.imshow(where(wt[order] != 0.0, 0.75, 0.0), aspect='auto', origin='lower', cmap='Blues', vmin=0.0, vmax=1.0,
              extent=(-0.5, nmu - 0.5, -0.5, ng - 0.5), interpolation='nearest')
    ax.set_title('One shared map', fontsize='medium')
    setp(ax, xlabel='Profile node, limb to centre', ylabel='Table node, disk centre to limb')

    ax = axs[1]
    # The tables are drawn as the model reads them, with a cubic through the nodes.
    gc, coef, g = split_point(kmean), empty((ng - 3, 4)), linspace(0.0, 1.0, 400)
    for ipb in range(npb):
        split_cubic_coefficients(tables[ipb], n1, coef)
        ax.plot(g, [ldm_lookup(x, gc, n1, coef) for x in g], c=colours[ipb], lw=1.5)
    ax.text(0.02, tables[0].max() + 0.03, f'{wl[0]:.1f} µm', fontsize='small', color='0.15')
    ax.text(0.02, tables[-1].max() - 0.03, f'{wl[-1]:.1f} µm', fontsize='small', color='0.15', va='top')
    ax.set_ylim(top=tables[0].max() + 0.12)
    ax.set_title('One table per bin', fontsize='medium')
    setp(ax, xlabel='Grazing parameter $b / (1 + k)$', ylabel='Mean intensity under the planet')
    _despine(ax)

    # The covered area across the limb: a larger planet gains a crescent along its arc inside the star.
    ax = axs[2]
    k0, k1, z = 0.30, 0.37, 0.92
    ax.add_patch(Circle((0, 0), 1.0, fc='0.94', ec='0.5', lw=0.8, zorder=0))
    t = linspace(0, 2 * pi, 2001)

    def arc_inside(r):
        # The part of a planet's edge that lies inside the star, a single arc around t = pi.
        m = (z + r * cos(t)) ** 2 + (r * sin(t)) ** 2 < 1.0
        return column_stack([z + r * cos(t[m]), r * sin(t[m])])

    def limb(p0, p1):
        # The stellar limb from the point p0 to the point p1.
        a = linspace(arctan2(p0[1], p0[0]), arctan2(p1[1], p1[0]), 30)
        return column_stack([cos(a), sin(a)])

    arc0, arc1 = arc_inside(k0), arc_inside(k1)
    crescent = concatenate([arc1, limb(arc1[-1], arc0[-1]), arc0[::-1], limb(arc0[0], arc1[0])])
    ax.add_patch(Polygon(crescent, fc='#eb6834', ec='none', alpha=0.85, zorder=1))
    ax.add_patch(Polygon(concatenate([arc0, limb(arc0[-1], arc0[0])]), fc='0.55', ec='none', zorder=1))
    ax.plot(z + k0 * cos(t), k0 * sin(t), c='k', lw=1, zorder=3)
    ax.plot(z + k1 * cos(t), k1 * sin(t), c='k', lw=1, ls='--', zorder=3)
    ax.plot(arc0[:, 0], arc0[:, 1], c='#2a78d6', lw=2.5, zorder=4)
    ax.text(z - k1 - 0.05, 0.0, 'Arc inside\nthe star', color='0.15', fontsize='small', ha='right', va='center')
    ax.set_title('Area across the limb', fontsize='medium')
    setp(ax, xlim=(z - 0.75, z + 0.45), ylim=(-0.6, 0.6))
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    return fig
