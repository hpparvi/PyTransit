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

"""Binning time series into fixed-duration bins."""

from math import ceil, floor

from numba import njit
from numpy import full, nan, zeros, sqrt, int64, asarray, ndarray


@njit
def _downsample(time, vals, inttime, tmin, tmax):
    """Bin the columns of `vals` into bins of duration `inttime` starting from `tmin`.

    The bin index is computed directly from each point's time, so the data need not be sorted,
    and no point can land outside the output arrays. The bins are half-open,
    ``[tmin + i inttime, tmin + (i+1) inttime)``, except the last one, which also includes `tmax`.
    """
    npt, ncol = vals.shape
    nbins = max(1, int(ceil((tmax - tmin) / inttime)))

    bids = full(npt, -1, int64)
    counts = zeros(nbins, int64)
    bt, bv, be = zeros(nbins), zeros((nbins, ncol)), zeros((nbins, ncol))

    for i in range(npt):
        if tmin <= time[i] <= tmax:
            ib = min(int(floor((time[i] - tmin) / inttime)), nbins - 1)
            bids[i] = ib
            counts[ib] += 1
            bt[ib] += time[i]
            for j in range(ncol):
                bv[ib, j] += vals[i, j]

    for ib in range(nbins):
        if counts[ib] > 0:
            bt[ib] /= counts[ib]
            for j in range(ncol):
                bv[ib, j] /= counts[ib]
        else:
            bt[ib] = nan
            bv[ib, :] = nan

    # The scatter is computed around the bin means in a second pass, which avoids the cancellation
    # of the one-pass sum-of-squares formula for values with a small relative scatter, such as fluxes.
    for i in range(npt):
        ib = bids[i]
        if ib >= 0:
            for j in range(ncol):
                be[ib, j] += (vals[i, j] - bv[ib, j])**2

    for ib in range(nbins):
        n = counts[ib]
        if n > 2:
            for j in range(ncol):
                be[ib, j] = sqrt(be[ib, j] / n) / sqrt(n)
        else:
            be[ib, :] = nan

    return bt, bv, be


def _validate(time, vals, inttime, tmin, tmax):
    time = asarray(time, dtype='d')
    vals = asarray(vals, dtype='d')
    if time.ndim != 1:
        raise ValueError("'time' must be a 1D array.")
    if vals.shape[0] != time.size:
        raise ValueError(f"'vals' must have the same length as 'time' ({time.size}), got {vals.shape[0]}.")
    if not inttime > 0.0:
        raise ValueError(f"'inttime' must be positive, got {inttime}.")
    tmin = time.min() if tmin is None else float(tmin)
    tmax = time.max() if tmax is None else float(tmax)
    if tmax < tmin:
        raise ValueError(f"'tmax' ({tmax}) must not be smaller than 'tmin' ({tmin}).")
    return time, vals, float(inttime), tmin, tmax


def downsample_time_1d(time, vals, inttime: float = 1., tmin=None, tmax=None,
                       is_sorted: bool = True) -> tuple[ndarray, ndarray, ndarray]:
    """Bin a time series into bins of a fixed duration.

    The bins are ``[tmin + i inttime, tmin + (i+1) inttime)``, with the last bin also including
    `tmax`. The last bin is shorter than `inttime` if the time range is not a multiple of it.

    Parameters
    ----------
    time : array_like
        1D array of times.
    vals : array_like
        1D array of values with the same length as `time`.
    inttime : float
        Bin duration, in the same units as `time`.
    tmin, tmax : float, optional
        Start and end of the binned time range. Default to the minimum and maximum of `time`.
        The points outside the range are ignored.
    is_sorted : bool
        Not used. The data do not need to be sorted.

    Returns
    -------
    bt : ndarray
        Mean time of the points in each bin.
    bv : ndarray
        Mean value of the points in each bin.
    be : ndarray
        Standard error of the mean value in each bin, NaN for bins with fewer than three points.

    Notes
    -----
    All the returned arrays are NaN for the empty bins.
    """
    time, vals, inttime, tmin, tmax = _validate(time, vals, inttime, tmin, tmax)
    if vals.ndim != 1:
        raise ValueError("'vals' must be a 1D array, use downsample_time_2d for 2D arrays.")
    bt, bv, be = _downsample(time, vals.reshape((-1, 1)), inttime, tmin, tmax)
    return bt, bv[:, 0], be[:, 0]


def downsample_time_2d(time, vals, inttime: float = 1., tmin=None, tmax=None,
                       is_sorted: bool = True) -> tuple[ndarray, ndarray, ndarray]:
    """Bin a multi-column time series into bins of a fixed duration.

    The bins are ``[tmin + i inttime, tmin + (i+1) inttime)``, with the last bin also including
    `tmax`. The last bin is shorter than `inttime` if the time range is not a multiple of it.

    Parameters
    ----------
    time : array_like
        1D array of times.
    vals : array_like
        2D array of values with a shape ``(npt, ncol)``, where `npt` is the length of `time`. A
        1D array is treated as a single column.
    inttime : float
        Bin duration, in the same units as `time`.
    tmin, tmax : float, optional
        Start and end of the binned time range. Default to the minimum and maximum of `time`.
        The points outside the range are ignored.
    is_sorted : bool
        Not used. The data do not need to be sorted.

    Returns
    -------
    bt : ndarray
        Mean time of the points in each bin.
    bv : ndarray
        Mean of each column in each bin as a ``(nbins, ncol)`` array.
    be : ndarray
        Standard error of the mean of each column in each bin as a ``(nbins, ncol)`` array, NaN
        for bins with fewer than three points.

    Notes
    -----
    All the returned arrays are NaN for the empty bins.
    """
    time, vals, inttime, tmin, tmax = _validate(time, vals, inttime, tmin, tmax)
    if vals.ndim == 1:
        vals = vals.reshape((-1, 1))
    elif vals.ndim != 2:
        raise ValueError("'vals' must be a 1D or 2D array.")
    return _downsample(time, vals, inttime, tmin, tmax)
