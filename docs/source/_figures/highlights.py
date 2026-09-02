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

"""Plotting helpers for the highlights page.

The point of that page is how little PyTransit code each feature takes, so the plotting is
factored out here and the page shows only the model calls. The helpers read the light curve,
passband and epoch structure from the model itself rather than being told about it, which is
also a fair demonstration of what `set_data` records.
"""

from matplotlib.pyplot import subplots, setp
from numpy import atleast_2d, ndarray

__all__ = ['plot_transit', 'plot_light_curves']

_YTICKS = (0.99, 0.995, 1.0)


def _despine(ax, left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)


def plot_transit(times, flux, title='', step=False, reference=None):
    """Plot a single light curve against time from mid-transit in hours.

    Parameters
    ----------
    times : ndarray
        Mid-exposure times in days, relative to the transit centre.
    flux : ndarray
        Modelled flux.
    title : str, optional
        Panel title.
    step : bool, optional
        Draw the model as a step function with markers, which is how binned long-cadence data
        actually looks.
    reference : tuple, optional
        A ``(times, flux)`` pair drawn behind in grey, for comparing against an unbinned model.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = subplots(figsize=(9, 4))
    if reference is not None:
        ax.plot(24 * reference[0], reference[1], c='0.75', zorder=-1)
    if step:
        ax.step(24 * times, flux, 'o-', where='mid', c='k', ms=4)
    else:
        ax.plot(24 * times, flux, 'k')
    ax.autoscale(axis='x', tight=True)
    setp(ax, xlabel='Time - $t_c$ [h]', ylabel='Normalised flux', title=title, yticks=_YTICKS)
    _despine(ax)
    fig.tight_layout()
    return fig


def plot_light_curves(model, times, flux, show_epochs=False):
    """Plot one panel per light curve, reading the structure from the model.

    The panels share a y axis so the depths can be compared by eye. A light curve the model
    supersamples is drawn as a step function, so the effect of `nsamples` is visible directly.

    Parameters
    ----------
    model : TransitModel
        The model the flux was computed with, already set up with `set_data`.
    times : ndarray
        The mid-exposure times the model was given.
    flux : ndarray
        Modelled flux, either 1D for a single parameter set or 2D for a population.
    show_epochs : bool, optional
        Add the epoch index to each panel title and mark the shared zero epoch, for TTV examples.

    Returns
    -------
    matplotlib.figure.Figure
    """
    flux = atleast_2d(flux)
    nlc = model.nlc

    fig, axs = subplots(1, nlc, figsize=(9, 3.6), sharey=True, squeeze=False)
    axs = axs[0]
    population = flux.shape[0] > 1

    for j, ax in enumerate(axs):
        mask = model.lcids == j
        t, f = 24 * times[mask], flux[:, mask]
        style = dict(alpha=0.2) if population else {}

        if model.nsamples[j] > 1:
            ax.step(t, f.T, '-k', where='mid', **(style or dict(marker='o', ms=4)))
        else:
            ax.plot(t, f.T, 'k', **style)

        ax.autoscale(axis='x', tight=True)
        title = f'LC {j}, PB {model.pbids[j]}'
        if show_epochs:
            title += f', EP {model.epids[j]}'
            ax.axvline(0.0, ls='--', c='0.75', zorder=-1)
        ax.set_title(title, fontsize='medium')

    setp(axs, xlabel='Time - $t_0$ [h]' if show_epochs else 'Time - $t_c$ [h]', yticks=_YTICKS)
    axs[0].set_ylabel('Normalised flux')
    _despine(axs[0])
    for ax in axs[1:]:
        _despine(ax, left=True)
        ax.tick_params(left=False)
    fig.tight_layout()
    return fig
