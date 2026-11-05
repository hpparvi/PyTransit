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

"""Miscellaneous utility functions."""

import numpy as np
from scipy.constants import c, k, h

def fold(time, period, origo=0.0, shift=0.0, normalize=True, clip_range=None):
    """Fold a time array over a period.

    Unlike :func:`pytransit.orbits.fold`, which returns phases centred on zero, this function
    returns phases between 0 and 1 (or between 0 and the period), with zero phase at `origo`.

    Parameters
    ----------
    time : ndarray
        Times to fold.
    period : float
        Folding period.
    origo : float
        Time of zero phase.
    shift : float
        Phase shift added before the folding, in units of the period.
    normalize : bool
        Return the phase normalised to the range 0-1 if True, and in the units of the period
        otherwise.
    clip_range : tuple of two floats, optional
        Phase range to keep, in the same units as the returned phase.

    Returns
    -------
    ndarray or tuple
        The folded phases. If `clip_range` is given, a tuple ``(phase, mask)`` where `phase`
        contains only the phases inside the range and `mask` tells which elements of `time`
        were kept.
    """
    tf = ((time - origo)/period + shift) % 1.

    if not normalize:
        tf *= period
        
    if clip_range is not None:
        mask = np.logical_and(clip_range[0]<tf, tf<clip_range[1])
        tf = tf[mask], mask
    return tf


def planck(T, wl):
    """Spectral radiance of a black body per unit wavelength.

    Parameters
    ----------
    T : float or ndarray
        Black body temperature [K].
    wl : float or ndarray
        Wavelength [m].

    Returns
    -------
    float or ndarray
        Spectral radiance [W m^-3 sr^-1].
    """
    return 2*h*c**2/wl**5 / (np.exp(h*c/(wl*k*T))-1)


def contamination_bb(c1, T, wl1, wl2):
    """Scale a contamination estimate from a reference wavelength to a target wavelength.

    Scales the contamination by the ratio of the contaminant's black body radiances at the two
    wavelengths.

    Parameters
    ----------
    c1 : float or ndarray
        Contamination at the reference wavelength, as a fraction of the total flux.
    T : float or ndarray
        Temperature of the contaminating object [K].
    wl1 : float
        Reference wavelength [m].
    wl2 : float
        Target wavelength [m].

    Returns
    -------
    float or ndarray
        Contamination at the target wavelength, as a fraction of the total flux.

    Notes
    -----
    The scaling ignores the host star's spectrum and treats the contamination fraction as
    proportional to the contaminant's flux, so it is valid only for a small contamination and a
    host star whose spectrum is flat between the two wavelengths. Use
    :mod:`pytransit.contamination` for a physically consistent contamination model.
    """
    B1   = planck(T, wl1)
    B2   = planck(T, wl2)

    return  c1*(B2/B1)
