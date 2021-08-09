#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2021  Hannu Parviainen
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

from numba import njit
from numpy import argsort, ceil, full, nan, zeros, sqrt, atleast_2d


@njit
def downsample_time_1d(time, vals, inttime=1., tmin=None, tmax=None, is_sorted=True):
    tmin = time.min() if tmin is None else tmin
    tmax = time.max() if tmax is None else tmax
    duration = tmax - tmin
    npt = time.size

    if not is_sorted:
        sids = argsort(time)
        time = time[sids]
        vals = vals[sids]

    nbins = int(ceil(duration / inttime))
    bt, bv, be = full(nbins, nan), zeros(nbins), zeros(nbins)

    ibin, tstart, istart, iend = 0, tmin, 0, 0
    while tstart <= tmax:
        while iend < npt and time[iend] <= tstart + inttime:
            iend += 1

        if iend > istart:
            bt[ibin] = time[istart:iend].mean()
            bv[ibin] = vals[istart:iend].mean()
            if iend - istart > 2:
                be[ibin] = vals[istart:iend].std() / sqrt(iend - istart)
            else:
                be[ibin] = nan

        istart = iend
        ibin += 1
        tstart += inttime
        if iend == npt:
            break
    return bt, bv, be


@njit
def downsample_time_2d(time, vals, inttime=1., tmin=None, tmax=None, is_sorted=True):
    vals = atleast_2d(vals)
    tmin = time.min() if tmin is None else tmin
    tmax = time.max() if tmax is None else tmax
    duration = tmax - tmin
    npt = time.size

    if not is_sorted:
        sids = argsort(time)
        time = time[sids]
        vals = vals[sids]

    nbins = int(ceil(duration / inttime))
    bt, bv, be = full(nbins, nan), zeros((nbins, vals.shape[1])), zeros((nbins, vals.shape[1]))

    ibin, tstart, istart, iend = 0, tmin, 0, 0
    while tstart <= tmax:
        while iend < npt and time[iend] <= tstart + inttime:
            iend += 1

        if iend > istart:
            nv = iend - istart
            bt[ibin] = time[istart:iend].mean()
            for j in range(vals.shape[1]):
                bv[ibin, j] = vals[istart:iend, j].mean()
            if nv > 2:
                for j in range(vals.shape[1]):
                    be[ibin, j] = vals[istart:iend, j].std() / sqrt(iend - istart)
            else:
                be[ibin, :] = nan

        istart = iend
        ibin += 1
        tstart += inttime
        if iend == npt:
            break
    return bt, bv, be
