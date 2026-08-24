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

`LCData` holds a single light curve together with the metadata describing it, and
`LCDataGroup` collects several of them. Adding light curves together gives a group::

    lc1 + lc2                    # -> LCDataGroup with two light curves
    lc1 + lc2 + lc3              # -> LCDataGroup with three
    sum([lc1, lc2, lc3])         # -> the same

The group exposes the per-light-curve quantities as lists and arrays ready to be handed to
a `BaseLPF`::

    lpf = BaseLPF('name', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                  pbids=lcs.pbids, covariates=lcs.covariates, wnids=lcs.wnids,
                  nsamples=lcs.nsamples, exptimes=lcs.exptimes)
"""

import warnings

from collections.abc import Sequence
from typing import Optional, Union

from numpy import ndarray, full, array, diff, nanstd, sqrt, nan, isfinite

from .base import (_Data, _DataGroup, _as_float, _as_int, _validate_time, _validate_series,
                   _validate_error, _validate_covariates, _validate_names, _validate_pids)

__all__ = ['LCData', 'LCDataGroup']


class LCData(_Data):
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
    pids
        Indices of the planets transiting in this light curve, for modelling multiplanet
        systems. An integer is accepted for a single planet. Stored as a tuple of ints.
        Defaults to `None` meaning "unspecified", which is distinct from an empty sequence
        meaning "no planet transits here".
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
                 pids: Optional[Union[int, Sequence[int]]] = None,
                 noise: Optional[float] = None,
                 instrument: str = '',
                 sector: int = -1,
                 segment: int = 0,
                 exptime: float = 0.0,
                 nsamples: int = 1) -> None:

        self.time = _validate_time(time)
        npt = self.time.size

        self.flux = _validate_series(flux, npt, 'flux')
        self._error = _validate_error(error, npt)
        self.covariates = _validate_covariates(covariates, npt)
        self.passband = _validate_names(passband, 'passband')
        self.pids = _validate_pids(pids)

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

    @property
    def has_pids(self) -> bool:
        """True if the transiting planets were specified."""
        return self.pids is not None

    def __repr__(self) -> str:
        return (f"LCData(npt={self.size}, passband={self.passband}, "
                f"pids={self.pids}, instrument={self.instrument!r}, sector={self.sector}, "
                f"segment={self.segment}, ncov={self.ncov})")


class LCDataGroup(_DataGroup):
    """A container of `LCData` objects.

    The light curves keep their insertion order, which defines both the light curve index
    (`lcid`) an LPF will use and the order of `passband_names`. Nothing is sorted
    implicitly; use `sorted_by` if you need a different order.

    Parameters
    ----------
    data
        A `LCData`, a sequence of them, or another `LCDataGroup`. Nested
        groups are flattened.
    """

    _item_type = LCData

    # Bulk data
    # ---------
    @property
    def fluxes(self) -> list:
        """List of 1D flux arrays."""
        return [d.flux for d in self.data]

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
    def pids(self) -> list:
        """List of per-light-curve transiting-planet index tuples.

        An entry is `None` for a light curve whose transiting planets were not specified.
        """
        return [d.pids for d in self.data]

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

    @property
    def has_pids(self) -> bool:
        """True if every light curve names its transiting planets."""
        return all(d.has_pids for d in self.data) if self.data else False

    @property
    def n_planets(self) -> int:
        """Number of planets implied by the planet indices.

        One past the largest index given by any light curve, or zero if none of them
        specify their planets.
        """
        ids = [p for d in self.data if d.pids is not None for p in d.pids]
        return max(ids) + 1 if ids else 0

    # Derived ids
    # -----------
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
    def piis(self) -> ndarray:
        """Running index of each light curve within its instrument.

        Together with `ins`, this gives `LinearModelBaseline` the per-light-curve tokens it
        uses to name its baseline parameters::

            lpf.ins, lpf.piis = lcs.ins, lcs.piis
        """
        return super().piis

    def __repr__(self) -> str:
        return (f"LCDataGroup with {self.size} light curves, {int(self.npts.sum())} points, "
                f"passbands {self.passband_names}")


LCData._group_type = LCDataGroup
