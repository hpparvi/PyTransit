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

"""Containers for radial velocity data and metadata.

`RVData` holds the radial velocities from a single instrument or observing campaign, and
`RVDataGroup` collects several of them. Adding datasets together gives a group, exactly as
with the light curve containers::

    rv1 + rv2                    # -> RVDataGroup with two datasets
    sum([rv1, rv2, rv3])         # -> RVDataGroup with three

The group exposes the per-dataset quantities in the form `RVLPF` wants::

    lpf = RVLPF('name', nplanets=2, times=rvd.times, rvs=rvd.rvs, rves=rvd.errors,
                rvis=rvd.rvis, is_transiting=[True, False])

Radial velocities are in **m/s** and times in **days**, matching `pytransit.lpf.rvlpf`.

Unlike `LCData` these containers have no `pids` field: a radial velocity signal is
the sum over every planet in the system, so there is no per-dataset subset of planets to
name. `is_transiting` is a property of a planet rather than of a dataset and stays an
argument of `RVLPF`.
"""

from collections.abc import Sequence
from typing import Optional, Union

from numpy import ndarray, array, diff, nanstd, sqrt, nan, isfinite

from .base import (_Data, _DataGroup, _validate_time, _validate_series, _validate_error,
                   _validate_covariates)

__all__ = ['RVData', 'RVDataGroup']


class RVData(_Data):
    """Radial velocities and metadata for a single instrument or campaign.

    Parameters
    ----------
    time
        Mid-measurement times in days, converted to a 1D float64 array. Must all be finite.
    rv
        Radial velocities in m/s, converted to a 1D float64 array of the same length as
        `time`.
    error
        Radial velocity uncertainties in m/s, a 1D float64 array of the same length as
        `time`. Required: unlike photometry, an RV measurement essentially always comes
        with an uncertainty, and `RVLPF` needs one per point.
    covariates
        Covariate matrix with shape ``(npt, ncov)`` holding activity indicators, airmass,
        or similar. A 1D array of length `npt` is treated as a single covariate, and `None`
        gives an empty ``(npt, 0)`` matrix.
    instrument
        Instrument name. `RVLPF` interpolates it into the names of the systemic velocity
        and jitter parameters, `rv_shift_{instrument}` and `rv_err_{instrument}`, so it
        must be unique within a group.

    Notes
    -----
    The arrays are stored without copying when they already are float64 ndarrays, so the
    caller and the object may share memory.
    """

    def __init__(self,
                 time: Union[Sequence, ndarray],
                 rv: Union[Sequence, ndarray],
                 error: Union[Sequence, ndarray],
                 covariates: Optional[Union[Sequence, ndarray]] = None,
                 instrument: str = '') -> None:

        self.time = _validate_time(time)
        npt = self.time.size

        self.rv = _validate_series(rv, npt, 'rv')

        if error is None:
            raise ValueError("'error' is required for RV data; radial velocities without "
                             "uncertainties cannot be modelled by RVLPF.")
        self.error = _validate_error(error, npt)

        self.covariates = _validate_covariates(covariates, npt)
        self.instrument = str(instrument)

    @property
    def size(self) -> int:
        """Number of datapoints."""
        return self.time.size

    @property
    def npt(self) -> int:
        """Number of datapoints."""
        return self.time.size

    @property
    def ncov(self) -> int:
        """Number of covariates."""
        return self.covariates.shape[1]

    @property
    def noise(self) -> float:
        """Point-to-point scatter estimate, ``nanstd(diff(rv)) / sqrt(2)``.

        A read-only diagnostic. Unlike `LCData` this is never a constructor
        argument, since RV uncertainties are always given, and it is not used by any model:
        `RVLPF` fits a separate jitter term per instrument on top of `error`.
        """
        if self.time.size < 2:
            return nan
        d = diff(self.rv)
        if not isfinite(d).any():
            return nan
        return float(nanstd(d) / sqrt(2))

    @property
    def has_error(self) -> bool:
        """Always True: RV uncertainties are required."""
        return True

    def __repr__(self) -> str:
        return (f"RVData(npt={self.size}, instrument={self.instrument!r}, ncov={self.ncov})")


class RVDataGroup(_DataGroup):
    """A container of `RVData` objects.

    The datasets keep their insertion order, which is the order `RVLPF` uses when it builds
    its per-dataset systemic velocity and jitter parameters. Nothing is sorted implicitly;
    use `sorted_by` if you need a different order.

    Parameters
    ----------
    data
        An `RVData`, a sequence of them, or another `RVDataGroup`. Nested groups are
        flattened.
    """

    _item_type = RVData

    @property
    def rvs(self) -> list:
        """List of 1D radial velocity arrays in m/s."""
        return [d.rv for d in self.data]

    @property
    def noises(self) -> ndarray:
        """Array of per-dataset point-to-point scatter estimates."""
        return array([d.noise for d in self.data], float)

    @property
    def rvis(self) -> list:
        """Instrument labels, one per dataset, ready for `RVLPF`.

        Returned as a `list` rather than an array on purpose: `RVLPF.__init__` tests
        ``if rvis:``, which raises on an ndarray.

        Raises
        ------
        ValueError
            If two datasets share an instrument label. `RVLPF` names one systemic velocity
            and one jitter parameter after each label, so duplicates would collide.
        """
        labels = self.instruments
        dupes = {lbl: [i for i, x in enumerate(labels) if x == lbl]
                 for lbl in set(labels) if labels.count(lbl) > 1}
        if dupes:
            raise ValueError(f"Instrument labels {sorted(dupes)} are used by more than one "
                             f"dataset ({dupes}); RVLPF creates one offset and one jitter "
                             f"parameter per label, so each dataset needs a distinct one. "
                             f"Merge the datasets or relabel them.")
        return labels

    def __repr__(self) -> str:
        return (f"RVDataGroup with {self.size} datasets, {int(self.npts.sum())} points, "
                f"instruments {self.instruments}")


RVData._group_type = RVDataGroup
