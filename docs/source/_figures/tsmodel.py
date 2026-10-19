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

"""Plotting helpers for the transmission spectroscopy feature page.

The page is about many wavelength bins sharing one transit geometry, so the figures show a whole
spectroscopic dataset at once and measure what sharing the geometry costs in accuracy. Keeping the
plotting and the toy spectrum here lets the page show only the PyTransit calls.
"""

from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import MaxNLocator
from meepmeep.backends.numba.point2d import sep_c, solve2d
from numpy import abs as npabs, array, exp, geomspace, linspace, pi, stack, tile

from pytransit import RoadRunnerModel, TSModel

__all__ = ['toy_spectrum', 'plot_spectroscopic_transit', 'plot_accuracy']

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
