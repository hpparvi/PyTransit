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

"""Diagrams of the index arrays `TransitModel.set_data` takes.

The example dataset is the one used throughout :doc:`../guide/data_setup`: 25 exposures split
into three light curves, observed in two passbands at two cadences. Every array is drawn as a
row of boxes, and the per-light-curve arrays are coloured to match the exposures they govern,
so the reader can see which entry applies where.
"""

from matplotlib.patches import Rectangle
from matplotlib.pyplot import subplots, setp
from numpy import arange, asarray, zeros

__all__ = ['draw_dataset', 'TIMES', 'LCIDS', 'PBIDS', 'EPIDS', 'NSAMPLES', 'EXPTIMES']

# The example dataset. Light curve 0 is short cadence, 1 and 2 are long cadence; light curves
# 0 and 2 share a passband.
TIMES = arange(25)
LCIDS = zeros(TIMES.size, 'int')
LCIDS[10:15] = 1
LCIDS[15:] = 2

PBIDS = [0, 1, 0]
EPIDS = [0, 1, 2]
NSAMPLES = [1, 10, 10]
EXPTIMES = [0.0, 0.02, 0.02]

_BOX = 0.4              # Box side length in data units.
_SLOT = 3.0             # Horizontal slot reserved for each per-light-curve array.
_X0 = 1.0               # Left edge of the first box, leaving room for the row labels.


def _draw_array(ax, a, x0=0.0, y0=0.0, fc='w', label='', cids=None, time=False):
    """Draw one array as a row of labelled boxes starting at (`x0`, `y0`).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    a : array-like
        The array to draw. One box per element.
    x0, y0 : float
        Position of the row's lower left corner.
    fc : str
        Box face colour, used when `cids` is not given.
    label : str
        Row label, drawn to the left of the first box.
    cids : array-like, optional
        Colour cycle index per box. Used to colour each entry by the light curve it belongs to.
    time : bool
        Label the boxes ``t_i`` rather than with the array values.
    """
    a = asarray(a)
    for i in range(a.size):
        colour = f'C{cids[i]}' if cids is not None else fc
        ax.add_patch(Rectangle((_BOX * i + x0, y0), _BOX, _BOX, fill=True, fc=colour, ec='k'))
        value = f'{a[i]:g}'
        text = f'$t_{{{a[i]}}}$' if time else value
        # Shrink long values, such as the exposure times, so that they stay inside their box.
        # The measure is the rendered value, not the math-mode source that wraps it.
        size = min(10.0, 22.0 / max(len(value), 2))
        ax.text(_BOX * (i + 0.5) + x0, 0.5 * _BOX + y0, text, va='center', ha='center',
                fontsize=size)
    ax.text(x0 - 0.3 * _BOX, 0.5 * _BOX + y0, label, ha='right', va='center')


def draw_dataset(lcids=False, pbids=False, epids=False, sampling=False, figsize=None):
    """Draw the example dataset, showing only the arrays asked for.

    The mid-exposure times are always drawn. Each further flag adds the array it names, so a
    page can build the picture up one argument at a time.

    Parameters
    ----------
    lcids : bool
        Draw the per-exposure light curve indices.
    pbids : bool
        Draw the per-light-curve passband indices.
    epids : bool
        Draw the per-light-curve epoch indices.
    sampling : bool
        Draw the per-light-curve supersampling rates and exposure times.
    figsize : tuple, optional
        Figure size in inches. Defaults to a size that suits the number of rows drawn.

    Returns
    -------
    matplotlib.figure.Figure
    """
    per_lc = ([('pbids', PBIDS)] if pbids else []) \
             + ([('epids', EPIDS)] if epids else []) \
             + ([('nsamples', NSAMPLES), ('exptimes', EXPTIMES)] if sampling else [])

    has_third_row = bool(per_lc)
    ymin = -1.6 if has_third_row else (-0.7 if lcids else -0.1)

    if figsize is None:
        figsize = (10.0, 2.0 if has_third_row else (1.4 if lcids else 1.0))

    fig, ax = subplots(figsize=figsize)
    _draw_array(ax, TIMES, _X0, 0.0, label='times', time=True)
    if lcids:
        _draw_array(ax, LCIDS, _X0, -0.6, label='lcids', cids=LCIDS)
    for slot, (label, values) in enumerate(per_lc):
        _draw_array(ax, values, _X0 + _SLOT * slot, -1.5, label=label, cids=arange(len(values)))

    setp(ax, xlim=(0.0, 11.4), ylim=(ymin, 0.5), xticks=[], yticks=[])
    ax.set_aspect(1)
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    return fig
