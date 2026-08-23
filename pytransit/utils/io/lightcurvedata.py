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

"""Containers for light curve data and metadata.

`LightCurveData` holds a single light curve together with the metadata describing it, and
`LightCurveDataGroup` collects several of them. Adding light curves together gives a group::

    lc1 + lc2                    # -> LightCurveDataGroup with two light curves
    lc1 + lc2 + lc3              # -> LightCurveDataGroup with three
    sum([lc1, lc2, lc3])         # -> the same

The group exposes the per-light-curve quantities as lists and arrays ready to be handed to
a `BaseLPF`::

    lpf = BaseLPF('name', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                  pbids=lcs.pbids, covariates=lcs.covariates, wnids=lcs.wnids,
                  nsamples=lcs.nsamples, exptimes=lcs.exptimes)
"""

import warnings

from collections.abc import Sequence
from typing import Callable, Optional, Union

from numpy import (ndarray, asarray, zeros, full, array, diff, nanstd, sqrt, nan, isfinite,
                   argsort, integer, floating, bool_)

__all__ = ['LightCurveData', 'LightCurveDataGroup']


def _as_float_array(x, name: str) -> ndarray:
    """Convert `x` into a float64 array, re-raising conversion failures informatively."""
    try:
        return asarray(x, 'd')
    except (TypeError, ValueError) as e:
        raise ValueError(f"Could not convert '{name}' into a float array: {e}") from e


def _as_int(v, name: str) -> int:
    """Convert `v` into an int, rejecting non-integral values."""
    if isinstance(v, (bool, bool_)):
        raise ValueError(f"'{name}' must be an integer, got a boolean.")
    try:
        iv = int(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"'{name}' must be an integer, got {v!r}.") from e
    if iv != v:
        raise ValueError(f"'{name}' must be an integer, got {v!r}.")
    return iv


def _as_float(v, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"'{name}' must be a float, got {v!r}.") from e


class LightCurveData:
    """Data and metadata for a single light curve.

    Parameters
    ----------
    time
        Mid-exposure times, converted to a 1D float64 array. Must all be finite.
    flux
        Fluxes, converted to a 1D float64 array of the same length as `time`. Normally
        normalised so that the out-of-transit level is close to unity.
    error
        Per-point flux uncertainties, converted to a 1D float64 array of the same length as
        `time`. Defaults to `None`, in which case `error` falls back to the white noise
        estimate repeated.
    covariates
        Covariate matrix with shape ``(npt, ncov)``. A 1D array of length `npt` is treated
        as a single covariate, and `None` gives an empty ``(npt, 0)`` matrix.
    passband
        Passband name, or a sequence of names for a light curve combining several. Stored
        as a tuple of strings.
    noise
        White noise estimate. Defaults to the point-to-point estimate
        ``nanstd(diff(flux)) / sqrt(2)``, the same estimator `BaseLPF` uses, or to NaN for
        a light curve with fewer than two points.
    instrument
        Instrument name.
    sector
        Sector, night, or similar observation id. Defaults to -1 for "unspecified".
    segment
        Segment id within the sector.
    exptime
        Exposure time **in days**.
    nsamples
        Number of supersamples used to integrate the model over the exposure.

    Notes
    -----
    The arrays are stored without copying when they already are float64 ndarrays, so the
    caller and the object may share memory.
    """

    def __init__(self,
                 time: Union[Sequence, ndarray],
                 flux: Union[Sequence, ndarray],
                 error: Optional[Union[Sequence, ndarray]] = None,
                 covariates: Optional[Union[Sequence, ndarray]] = None,
                 passband: Union[str, Sequence[str]] = 'white',
                 noise: Optional[float] = None,
                 instrument: str = '',
                 sector: int = -1,
                 segment: int = 0,
                 exptime: float = 0.0,
                 nsamples: int = 1) -> None:

        self.time = _as_float_array(time, 'time')
        if self.time.ndim != 1:
            raise ValueError(f"'time' must be a 1D array, got a {self.time.ndim}D one.")
        if not isfinite(self.time).all():
            raise ValueError("'time' contains non-finite values.")

        self.flux = _as_float_array(flux, 'flux')
        if self.flux.ndim != 1:
            raise ValueError(f"'flux' must be a 1D array, got a {self.flux.ndim}D one.")
        if self.flux.size != self.time.size:
            raise ValueError(f"'flux' has {self.flux.size} points but 'time' has {self.time.size}.")
        if not isfinite(self.flux).all():
            warnings.warn("'flux' contains non-finite values.")

        if error is None:
            self._error = None
        else:
            e = _as_float_array(error, 'error')
            if e.ndim != 1:
                raise ValueError(f"'error' must be a 1D array, got a {e.ndim}D one.")
            if e.size != self.time.size:
                raise ValueError(f"'error' has {e.size} points but 'time' has {self.time.size}.")
            finite = isfinite(e)
            if not finite.all():
                warnings.warn("'error' contains non-finite values.")
            if (e[finite] <= 0.0).any():
                raise ValueError("'error' contains non-positive values.")
            self._error = e

        npt = self.time.size
        if covariates is None:
            self.covariates = zeros((npt, 0))
        else:
            cv = _as_float_array(covariates, 'covariates')
            if cv.ndim == 1:
                cv = cv.reshape((-1, 1))
            elif cv.ndim != 2:
                raise ValueError(f"'covariates' must be a 1D or 2D array, got a {cv.ndim}D one.")
            if cv.shape[0] != npt:
                raise ValueError(f"'covariates' has {cv.shape[0]} rows but 'time' has {npt} points.")
            self.covariates = cv

        if isinstance(passband, str):
            self.passband = (passband,)
        else:
            try:
                self.passband = tuple(str(pb) for pb in passband)
            except TypeError as e:
                raise ValueError(f"'passband' must be a string or a sequence of strings, "
                                 f"got {passband!r}.") from e
        if len(self.passband) == 0:
            raise ValueError("'passband' cannot be empty.")

        if noise is None:
            self.noise = self._estimate_noise()
        else:
            self.noise = _as_float(noise, 'noise')
            if not isfinite(self.noise) or self.noise <= 0.0:
                raise ValueError(f"'noise' must be a finite positive number, got {noise!r}.")

        self.instrument = str(instrument)
        self.sector = _as_int(sector, 'sector')
        self.segment = _as_int(segment, 'segment')

        self.exptime = _as_float(exptime, 'exptime')
        if not isfinite(self.exptime) or self.exptime < 0.0:
            raise ValueError(f"'exptime' must be a finite non-negative number, got {exptime!r}.")

        self.nsamples = _as_int(nsamples, 'nsamples')
        if self.nsamples < 1:
            raise ValueError(f"'nsamples' must be at least one, got {nsamples!r}.")
        if self.nsamples > 1 and self.exptime == 0.0:
            warnings.warn("Supersampling has no effect with a zero exposure time.")

    def _estimate_noise(self) -> float:
        """Point-to-point white noise estimate, matching `BaseLPF`'s own."""
        if self.time.size < 2:
            return nan
        d = diff(self.flux)
        if not isfinite(d).any():
            return nan
        return float(nanstd(d) / sqrt(2))

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
    def error(self) -> ndarray:
        """Per-point flux uncertainty.

        The array given at construction, or the white noise estimate repeated if none was
        given. `noise` is estimated from the point-to-point flux scatter and is not derived
        from `error`, so the two are independent descriptions of the uncertainty.
        """
        return self._error if self._error is not None else full(self.size, self.noise)

    @property
    def has_error(self) -> bool:
        """True if an explicit uncertainty array was given."""
        return self._error is not None

    def __add__(self, other) -> 'LightCurveDataGroup':
        if isinstance(other, LightCurveData):
            return LightCurveDataGroup([self, other])
        elif isinstance(other, LightCurveDataGroup):
            return LightCurveDataGroup([self] + other.data)
        return NotImplemented

    def __radd__(self, other):
        if other == 0 or other is None:
            return LightCurveDataGroup([self])
        return NotImplemented

    def __repr__(self) -> str:
        return (f"LightCurveData(npt={self.size}, passband={self.passband}, "
                f"instrument={self.instrument!r}, sector={self.sector}, segment={self.segment}, "
                f"ncov={self.ncov})")


class LightCurveDataGroup:
    """A container of `LightCurveData` objects.

    The light curves keep their insertion order, which defines both the light curve index
    (`lcid`) an LPF will use and the order of `passband_names`. Nothing is sorted
    implicitly; use `sorted_by` if you need a different order.

    Parameters
    ----------
    data
        A `LightCurveData`, a sequence of them, or another `LightCurveDataGroup`. Nested
        groups are flattened.
    """

    def __init__(self, data: Union['LightCurveData', Sequence, 'LightCurveDataGroup'] = ()) -> None:
        self.data: list = []
        if isinstance(data, (LightCurveData, LightCurveDataGroup)):
            data = [data]
        for d in data:
            self._add_data(d)

    def _add_data(self, d) -> None:
        if isinstance(d, LightCurveDataGroup):
            for x in d.data:
                self._add_data(x)
            return
        if not isinstance(d, LightCurveData):
            raise TypeError(f"A LightCurveDataGroup holds LightCurveData objects, "
                            f"got {type(d).__name__}.")
        if any(d is x for x in self.data):
            raise ValueError("The same LightCurveData instance cannot be added to a group twice.")
        self.data.append(d)

    # Bulk data
    # ---------
    @property
    def times(self) -> list:
        """List of 1D time arrays."""
        return [d.time for d in self.data]

    @property
    def fluxes(self) -> list:
        """List of 1D flux arrays."""
        return [d.flux for d in self.data]

    @property
    def covariates(self) -> list:
        """List of 2D covariate matrices."""
        return [d.covariates for d in self.data]

    @property
    def errors(self) -> list:
        """List of 1D flux uncertainty arrays."""
        return [d.error for d in self.data]

    # Metadata
    # --------
    @property
    def noises(self) -> ndarray:
        """Array of white noise estimates."""
        return array([d.noise for d in self.data], float)

    @property
    def passbands(self) -> list:
        """List of per-light-curve passband name tuples."""
        return [d.passband for d in self.data]

    @property
    def instruments(self) -> list:
        """List of instrument names."""
        return [d.instrument for d in self.data]

    @property
    def sectors(self) -> ndarray:
        """Array of sector ids."""
        return array([d.sector for d in self.data], int)

    @property
    def segments(self) -> ndarray:
        """Array of segment ids."""
        return array([d.segment for d in self.data], int)

    @property
    def exptimes(self) -> ndarray:
        """Array of exposure times in days."""
        return array([d.exptime for d in self.data], float)

    @property
    def nsamples(self) -> ndarray:
        """Array of supersampling factors."""
        return array([d.nsamples for d in self.data], int)

    # Sizes
    # -----
    @property
    def size(self) -> int:
        """Number of light curves."""
        return len(self.data)

    @property
    def npts(self) -> ndarray:
        """Array of per-light-curve datapoint counts."""
        return array([d.size for d in self.data], int)

    @property
    def ncovs(self) -> ndarray:
        """Array of per-light-curve covariate counts."""
        return array([d.ncov for d in self.data], int)

    @property
    def has_errors(self) -> bool:
        """True if every light curve was given an explicit uncertainty array."""
        return all(d.has_error for d in self.data) if self.data else False

    @property
    def has_covariates(self) -> bool:
        """True if any light curve has at least one covariate."""
        return bool((self.ncovs > 0).any())

    @property
    def tmin(self) -> float:
        """Earliest time in the group."""
        return min((d.time.min() for d in self.data if d.size), default=nan)

    @property
    def tmax(self) -> float:
        """Latest time in the group."""
        return max((d.time.max() for d in self.data if d.size), default=nan)

    # Derived ids
    # -----------
    def _group_ids(self, *fields) -> ndarray:
        """Map tuples of per-light-curve metadata into 0-based contiguous group ids.

        The ids are numbered by first appearance, so they are contiguous by construction,
        which `BaseLPF._init_data` asserts for the noise ids.
        """
        keys = list(zip(*fields))
        if not keys:
            return zeros(0, int)
        cats = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        return array([cats[k] for k in keys], int)

    @property
    def passband_names(self) -> list:
        """Unique passband names, ordered by first appearance.

        Suitable as the `passbands` argument of a `BaseLPF`, which expects the passbands to
        be ordered from blue to red. The order follows insertion order rather than being
        sorted, so arrange the light curves accordingly.
        """
        return list(dict.fromkeys(pb for d in self.data for pb in d.passband))

    @property
    def pbids(self) -> ndarray:
        """Passband index per light curve, indexing into `passband_names`.

        Raises
        ------
        ValueError
            If any light curve has more than one passband. An LPF passband carries a single
            radius ratio and limb darkening pair, so a light curve combining several
            passbands has no single correct index.
        """
        bad = [i for i, d in enumerate(self.data) if len(d.passband) != 1]
        if bad:
            raise ValueError(f"Light curves {bad} have more than one passband; 'pbids' requires "
                             f"exactly one passband per light curve.")
        names = self.passband_names
        return array([names.index(d.passband[0]) for d in self.data], int)

    @property
    def wnids(self) -> ndarray:
        """Noise block index per light curve.

        Light curves sharing an instrument, sector, and segment share a white noise
        parameter. With the default metadata every light curve falls into block 0, matching
        `BaseLPF`'s own default, and the blocks separate as soon as the metadata is filled
        in. Use `_group_ids` directly for a different grouping.
        """
        return self._group_ids(self.instruments, self.sectors, self.segments)

    @property
    def ins(self) -> list:
        """Instrument names, under the name `LinearModelBaseline` looks for."""
        return self.instruments

    @property
    def piis(self) -> ndarray:
        """Running index of each light curve within its instrument.

        Together with `ins`, this gives `LinearModelBaseline` the per-light-curve tokens it
        uses to name its baseline parameters::

            lpf.ins, lpf.piis = lcs.ins, lcs.piis
        """
        counts: dict = {}
        out = []
        for ins in self.instruments:
            out.append(counts.get(ins, 0))
            counts[ins] = out[-1] + 1
        return array(out, int)

    # Selection
    # ---------
    def select(self, instrument=None, passband=None, sector=None, segment=None) -> 'LightCurveDataGroup':
        """Return a new group with the light curves matching all the given criteria."""
        def matches(d):
            return ((instrument is None or d.instrument == instrument)
                    and (passband is None or passband in d.passband)
                    and (sector is None or d.sector == sector)
                    and (segment is None or d.segment == segment))
        return LightCurveDataGroup([d for d in self.data if matches(d)])

    def sorted_by(self, key: Union[str, Callable] = 'time') -> 'LightCurveDataGroup':
        """Return a new group ordered by `key`.

        Parameters
        ----------
        key
            One of 'time', 'instrument', 'passband', 'sector', or 'segment', or a callable
            taking a `LightCurveData` and returning a sort key.
        """
        keys = {'time': lambda d: d.time.min() if d.size else nan,
                'instrument': lambda d: d.instrument,
                'passband': lambda d: d.passband,
                'sector': lambda d: d.sector,
                'segment': lambda d: d.segment}
        if callable(key):
            f = key
        elif key in keys:
            f = keys[key]
        else:
            raise ValueError(f"Unknown sort key {key!r}, choose from {sorted(keys)} or pass a callable.")
        return LightCurveDataGroup(sorted(self.data, key=f))

    # Container protocol
    # ------------------
    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        if isinstance(index, (int, integer)):
            return self.data[index]
        if isinstance(index, slice):
            return LightCurveDataGroup(self.data[index])
        ix = asarray(index)
        if ix.dtype == bool:
            if ix.size != self.size:
                raise IndexError(f"Boolean index has {ix.size} entries but the group has {self.size} "
                                 f"light curves.")
            return LightCurveDataGroup([d for d, m in zip(self.data, ix) if m])
        if ix.dtype.kind in 'iu':
            return LightCurveDataGroup([self.data[i] for i in ix])
        raise TypeError(f"Cannot index a LightCurveDataGroup with {type(index).__name__}.")

    def __add__(self, other) -> 'LightCurveDataGroup':
        if isinstance(other, LightCurveData):
            return LightCurveDataGroup(self.data + [other])
        elif isinstance(other, LightCurveDataGroup):
            return LightCurveDataGroup(self.data + other.data)
        return NotImplemented

    def __radd__(self, other):
        if other == 0 or other is None:
            return LightCurveDataGroup(self.data)
        return NotImplemented

    def __repr__(self) -> str:
        return (f"LightCurveDataGroup with {self.size} light curves, {int(self.npts.sum())} points, "
                f"passbands {self.passband_names}")
