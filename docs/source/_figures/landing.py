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

"""The showcase figure of the documentation landing page.

One figure, three models: the upper row shows what each model is given and the lower row the light
curves it returns. Everything is computed with PyTransit at build time; nothing is a sketch.

- **RoadRunner** takes any radially symmetric stellar intensity profile, so the same planet is
  shown transiting three stars: two with built-in limb darkening laws and one with a profile
  written as a plain function.
- **The transmission spectroscopy model** takes one radius ratio and one set of limb darkening
  coefficients per wavelength bin and returns all the light curves from a single call.
- **The gravity-darkened star model** takes a rapidly rotating star, whose poles are brighter than
  its equator, and returns the asymmetric transit of a planet on a misaligned orbit.
"""

from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.pyplot import figure, setp
from numba import njit
from numpy import arccos, array, column_stack, linspace, meshgrid, nan, pi, sqrt, stack, where

from pytransit import GravityDarkenedModel, RoadRunnerModel, TSModel
from pytransit.contamination import BoxcarFilter

from .gdmodel import _draw_star
from .tsmodel import toy_spectrum

__all__ = ['plot_showcase']

# The three series colours of the RoadRunner panel, and the ink used for text and axes.
_SERIES = ('#2a78d6', '#eb6834', '#1baf7a')
_INK, _MUTED = '0.15', '0.45'

# Wavelength is drawn with one sequential colour map, cut before its palest end so that the
# reddest bins stay visible on a white page.
_WL_CMAP = colormaps['plasma']
_WL_RANGE = 0.88


@njit
def limb_brightened(mu, pv):
    """A stellar disk that brightens towards its limb, as an optically thin emission line does."""
    return 1.0 + pv[0] * (1.0 - mu) ** 2


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize='small')
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_MUTED)


def _heading(fig, ax, title, subtitle):
    """A column heading above the light curve axes `ax`: the model's name and what sets it apart."""
    box = ax.get_position()
    x = 0.5 * (box.x0 + box.x1)
    fig.text(x, 0.955, title, ha='center', va='center', fontsize='large', fontweight='bold', color=_INK)
    fig.text(x, 0.905, subtitle, ha='center', va='center', fontsize='small', color=_MUTED)


def _draw_disk(ax, profile, pv, res=200):
    """Draw a stellar disk with the given intensity profile."""
    x, y = meshgrid(linspace(-1, 1, res), linspace(-1, 1, res))
    z2 = x ** 2 + y ** 2
    mu = sqrt(where(z2 < 1.0, 1.0 - z2, 0.0))
    image = profile(mu.ravel(), array(pv, float)).reshape(mu.shape)
    image = where(z2 < 1.0, image, nan)
    ax.imshow(image, extent=(-1, 1, -1, 1), origin='lower', cmap='inferno', vmin=0.0, vmax=1.25 * image[res // 2, res // 2])
    setp(ax, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax.set_aspect('equal')
    ax.axis('off')


def _roadrunner(fig, top, bottom):
    """Three intensity profiles, one planet: RoadRunner works with any of them."""
    time = linspace(-0.085, 0.085, 600)
    k, t0, p, a, i = 0.1, 0.0, 4.0, 13.0, 0.5 * pi
    laws = {'Quadratic': ('quadratic', [0.2, 0.1]),
            'Power-2': ('power-2', [0.85, 0.7]),
            'Your own function': (limb_brightened, [1.5])}

    disks = top.subgridspec(1, 3, wspace=0.05)
    ax = fig.add_subplot(bottom)
    for j, (label, (law, pv)) in enumerate(laws.items()):
        tm = RoadRunnerModel(law)
        tm.set_data(time)
        ax.plot(24 * time, tm.evaluate(k, pv, t0, p, a, i), c=_SERIES[j], lw=2, label=label)

        ad = fig.add_subplot(disks[j])
        profile = RoadRunnerModel.ldmodels[law][0] if isinstance(law, str) else law
        _draw_disk(ad, profile, pv)
        # The label under each disk carries the colour of its light curve as a short rule, so
        # the text itself stays in ink.
        ad.plot([-0.45, 0.45], [-1.28, -1.28], c=_SERIES[j], lw=2.5, clip_on=False, solid_capstyle='round')
        ad.text(0.0, -1.42, label, ha='center', va='top', fontsize='small', color=_INK)
    _heading(fig, ax, 'RoadRunner', 'Any stellar intensity profile')
    return ax


def _spectroscopy(fig, top, bottom, npb=200):
    """Two hundred wavelength bins from one call of the transmission spectroscopy model."""
    time = linspace(-0.085, 0.085, 240)       # Coarse enough to keep the 200 curves a small SVG
    t0, p, a, i = 0.0, 4.0, 13.0, 0.5 * pi
    wl, k, ldc = toy_spectrum(npb)

    tm = TSModel('quadratic')
    tm.set_data(time)
    flux = tm.evaluate(k, ldc, t0, p, a, i)[0]

    norm = Normalize(wl[0], wl[0] + (wl[-1] - wl[0]) / _WL_RANGE)
    colours = _WL_CMAP(norm(wl))

    # The input: the planet's size at each wavelength, drawn in the colours of its light curves.
    # The spectrum's axis is inset from the left so that its label clears the neighbouring panel.
    at = fig.add_subplot(top.subgridspec(1, 2, width_ratios=(0.17, 1.0), wspace=0.0)[1])
    depth = 1e2 * k ** 2
    points = column_stack([wl, depth])
    at.add_collection(LineCollection(stack([points[:-1], points[1:]], axis=1), colors=colours[:-1], lw=2.5,
                                     capstyle='round'))
    setp(at, xlim=(wl[0], wl[-1]), ylim=(depth.min() - 0.006, depth.max() + 0.006), xlabel='Wavelength [µm]',
         ylabel='Transit depth [%]')
    at.xaxis.label.set_color(_MUTED)
    at.yaxis.label.set_color(_MUTED)
    at.xaxis.label.set_size('small')
    at.yaxis.label.set_size('small')
    _despine(at)

    ax = fig.add_subplot(bottom)
    _heading(fig, ax, 'Transmission spectroscopy', f'{npb} wavelength bins from one call')
    # Drawn from red to blue so that the deeper blue curves end up on top at mid-transit.
    for j in range(npb - 1, -1, -1):
        ax.plot(24 * time, flux[j], c=colours[j], lw=0.6)
    return ax


def _gravity_darkening(fig, top, bottom):
    """A planet on a misaligned orbit crossing a fast-rotating, gravity-darkened star."""
    time = linspace(-0.13, 0.13, 400)
    rho, rperiod, tpole, phi, beta = 0.48, 0.47, 8500.0, 0.5, 0.22
    k, ldc, t0, p, a = 0.1, [0.3, 0.2], 0.0, 3.0, 6.0
    i = arccos(0.3 / a)
    l = pi / 3

    tm = GravityDarkenedModel(filters=BoxcarFilter('blue', 400, 500), rstar=1.8, model='blackbody',
                              tmin=5000, tmax=9500, sres=160, pres=40, tres=240)
    tm.set_data(time)

    at = fig.add_subplot(top)
    _draw_star(at, tm, time, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l)

    ax = fig.add_subplot(bottom)
    _heading(fig, ax, 'Gravity-darkened star', 'Spin-orbit angles from photometry')
    ax.plot(24 * time, tm.evaluate(k, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, 0.0), c='0.65', lw=2,
            label='Aligned orbit')
    ax.plot(24 * time, tm.evaluate(k, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l), c=_SERIES[1], lw=2,
            label='Misaligned by 60°')
    return ax


def plot_showcase():
    """Draw the landing page figure: three models, what each is given and what it returns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.25), left=0.07, right=0.99, top=0.84, bottom=0.10,
                          wspace=0.10, hspace=0.42)

    axs = (_roadrunner(fig, gs[0, 0], gs[1, 0]),
           _spectroscopy(fig, gs[0, 1], gs[1, 1]),
           _gravity_darkening(fig, gs[0, 2], gs[1, 2]))

    for ax in axs:
        ax.autoscale(axis='x', tight=True)
        ax.set_xlabel('Time from mid-transit [h]', color=_MUTED, fontsize='small')
        _despine(ax)
        ax.set_ylim(0.9866, 1.0006)
    axs[0].set_ylabel('Normalised flux', color=_MUTED, fontsize='small')
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    # The first panel is labelled under its disks and the second by the spectrum above it.
    axs[2].legend(fontsize='small', frameon=False, loc='center', bbox_to_anchor=(0.5, 0.72), labelcolor=_INK,
                  handlelength=1.4)
    return fig
