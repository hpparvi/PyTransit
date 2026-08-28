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

from matplotlib.pyplot import subplots, setp
from numpy import (ndarray, full, array, diff, nanstd, sqrt, nan, isfinite, ceil, floor, s_,
                   median, zeros, atleast_1d, ones, hstack, where, ptp)
from numpy.linalg import lstsq
from scipy.signal import medfilt

from .base import (_Data, _DataGroup, _as_float, _as_int, _validate_time, _validate_series,
                   _validate_error, _validate_covariates, _validate_names, _validate_pids)

__all__ = ['LCData', 'LCDataGroup']


# Running median and outliers
# ---------------------------
def _as_odd_width(width) -> int:
    """Validate a running median width: a positive odd integer, as `medfilt` requires."""
    w = _as_int(width, 'width')
    if w < 1 or w % 2 == 0:
        raise ValueError(f"'width' must be a positive odd integer, got {width!r}.")
    return w


def _running_median(flux: ndarray, width: int) -> ndarray:
    """Running median of `flux` with the zero-padded edges repaired.

    `scipy.signal.medfilt` pads with zeros, which biases the median of the first and the last
    `width // 2` points towards the low end of their window. Those points are given the median of
    the first and the last full window instead.
    """
    m = medfilt(flux, width)
    h = width // 2
    if h and flux.size > width:
        m[:h] = median(flux[:width])
        m[-h:] = median(flux[-width:])
    return m


def _mad_sigma(residuals: ndarray) -> float:
    """Robust scatter estimate, 1.4826 times the median absolute deviation from the median."""
    return float(1.4826 * median(abs(residuals - median(residuals))))


# Covariates
# ----------
def _standardise(covariates: ndarray) -> ndarray:
    """Standardise the covariate columns to zero mean and unit standard deviation.

    Mixing a column on an arbitrary scale with an intercept column of ones makes the design matrix
    badly conditioned, and the least-squares solution numerically poor. Constant columns carry no
    information and are only centred, since scaling them would divide by zero. This is the same
    standardisation `LSTSQBaseline.normalize_covariates` applies.
    """
    cs = covariates.std(0)
    return (covariates - covariates.mean(0)) / where(cs > 0.0, cs, 1.0)


# Default line styles for the overlays `LCDataGroup.plot` draws on top of the flux. The zorder is
# above the data points, which matplotlib draws at zorder 2.
_MEDIAN_STYLE = dict(c='k', lw=1, alpha=1.0, zorder=10)
_LINEAR_MODEL_STYLE = dict(c='C1', lw=1, alpha=1.0, zorder=10)


def _line_style(defaults: dict, overrides: Optional[dict]) -> dict:
    """Merge user-given line properties over the defaults."""
    return {**defaults, **(overrides or {})}


def _normalised_time(time: ndarray) -> ndarray:
    """Time mapped linearly onto -1 ... 1. A light curve with no time span maps onto zeros."""
    if time.size == 0:
        return time.copy()
    span = ptp(time)
    return zeros(time.size) if span == 0.0 else 2.0 * (time - time.min()) / span - 1.0


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

    A light curve also carries a boolean `marked` flag, `False` at construction, that
    `LCDataGroup.mark_for_removal` sets and `LCDataGroup.remove_marked` acts on. It records the
    state of an interactive session rather than a property of the data, so it is not a constructor
    argument and does not affect anything a model sees.
    """

    # A class-level default so that a light curve pickled before this attribute existed unpickles
    # as unmarked instead of raising an AttributeError.
    marked: bool = False

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

        self._noise_given = noise is not None
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

        self.marked = False

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

    # Outliers
    # --------
    def running_median(self, width: int = 15) -> ndarray:
        """Running median of the flux, `width` points wide.

        Parameters
        ----------
        width
            Width of the median filter in points. Must be a positive odd integer, and is clipped to
            the length of the light curve when it is longer.
        """
        w = _as_odd_width(width)
        if w > self.size:
            w = max(1, self.size - 1 + self.size % 2)
        return _running_median(self.flux, w)

    def outlier_mask(self, nsigma: float = 3.0, width: int = 15) -> ndarray:
        """Boolean mask flagging the flux points further than `nsigma` from the running median.

        The scatter is a robust MAD estimate of the residuals from the running median, so that a
        few strong outliers cannot inflate the very threshold meant to catch them. A light curve
        with no scatter at all flags nothing.

        Parameters
        ----------
        nsigma
            Clipping threshold in units of the robust residual scatter.
        width
            Width of the median filter in points. Must be a positive odd integer.
        """
        if self.size == 0:
            return zeros(0, bool)
        r = self.flux - self.running_median(width)
        s = _mad_sigma(r)
        if not isfinite(s) or s == 0.0:
            return zeros(self.size, bool)
        return abs(r) > nsigma * s

    def remove_outliers(self, nsigma: float = 3.0, width: int = 15) -> int:
        """Remove the flux points further than `nsigma` from the running median, in place.

        The times, fluxes, covariates, and uncertainties are all filtered together, and the noise
        estimate is refreshed unless it was given explicitly. Returns the number of points removed.

        Parameters
        ----------
        nsigma
            Clipping threshold in units of the robust residual scatter.
        width
            Width of the median filter in points. Must be a positive odd integer.
        """
        m = ~self.outlier_mask(nsigma, width)
        if m.all():
            return 0

        self.time = self.time[m]
        self.flux = self.flux[m]
        self.covariates = self.covariates[m]
        if self._error is not None:
            self._error = self._error[m]
        if not self._noise_given:
            self.noise = self._estimate_noise()
        return int((~m).sum())

    # Covariates
    # ----------
    def add_time_covariates(self, order: int = 1) -> None:
        """Add the normalised time and its powers as covariates, in place.

        The time is mapped linearly onto -1 ... 1 and its powers 1 ... `order` are appended to the
        existing covariates, so that a linear-in-covariates baseline can absorb a polynomial trend
        in time. The powers start from one because the intercept is added by the baseline model
        itself, and a constant column would only make the design matrix singular.

        Parameters
        ----------
        order
            Highest power of the normalised time to add. Must be a positive integer.
        """
        o = _as_int(order, 'order')
        if o < 1:
            raise ValueError(f"'order' must be a positive integer, got {order!r}.")
        tn = _normalised_time(self.time)
        self.covariates = hstack([self.covariates] + [(tn ** i)[:, None] for i in range(1, o + 1)])

    def linear_model(self) -> ndarray:
        """Least-squares linear model of the flux in terms of the covariates.

        The covariate columns are standardised and an intercept is added, and the flux is fitted
        against them by linear least squares. The result is the part of the flux variability the
        covariates can explain, which is what `LCDataGroup.plot` overlays with `show_linear_model`.
        A light curve with no covariates gives its mean flux.
        """
        x = ones((self.size, 1)) if self.ncov == 0 else hstack([ones((self.size, 1)),
                                                                _standardise(self.covariates)])
        return x @ lstsq(x, self.flux, rcond=None)[0]

    def __repr__(self) -> str:
        marked = ', marked' if self.marked else ''
        return (f"LCData(npt={self.size}, passband={self.passband}, "
                f"pids={self.pids}, instrument={self.instrument!r}, sector={self.sector}, "
                f"segment={self.segment}, ncov={self.ncov}{marked})")


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

    @property
    def lcslices(self) -> list:
        """Slices splitting a concatenated array back into per-light-curve arrays.

        The slices follow the order of the light curves in the group, so an array created by
        concatenating any of the per-light-curve quantities can be split back into a list::

            timea = concatenate(lcs.times)
            times = [timea[sl] for sl in lcs.lcslices]

        These are the slices `BaseLPF` stores as `lcslices` for the same data.
        """
        slices, start = [], 0
        for npt in self.npts:
            slices.append(s_[start:start + int(npt)])
            start += int(npt)
        return slices

    # Marking and removal
    # -------------------
    @property
    def marked(self) -> ndarray:
        """Boolean array telling which light curves are marked for removal."""
        return array([d.marked for d in self.data], bool)

    @property
    def n_marked(self) -> int:
        """Number of light curves marked for removal."""
        return int(sum(d.marked for d in self.data))

    def mark_for_removal(self, indices) -> None:
        """Mark the given light curves for removal, modifying the group in place.

        Marking only sets a flag: the marked light curves stay in the group and keep appearing in
        every bulk property until `remove_marked` is called, so nothing a model sees changes until
        then. `plot` draws the marked light curves on a light gray background, so the workflow is
        to plot the group, mark the bad light curves by the index shown in the upper left corner of
        their panels, plot again to check, and then remove them.

        Parameters
        ----------
        indices
            Index of a light curve, a sequence, set, or array of indices, a boolean mask with one
            entry per light curve, or a slice. Negative indices count from the end.

        Notes
        -----
        Marking is additive, so several calls accumulate, and `unmark` clears the marks again. The
        flag lives in the `LCData` objects themselves, which the groups share by reference, so
        marking a light curve through a `select` result or a slice marks it in the group it came
        from as well. The marked light curves can be selected with `select(marked=True)`.

        Examples
        --------
        ::

            lcs.plot()                       # Eyeball the light curves
            lcs.mark_for_removal([2, 7, 9])  # Mark the bad ones by their panel index
            lcs.plot()                       # The marked ones now have a gray background
            lcs.remove_marked()
        """
        for i in self._resolve_indices(indices):
            self.data[i].marked = True

    def unmark(self, indices=None) -> None:
        """Clear the removal marks of the given light curves, modifying the group in place.

        Parameters
        ----------
        indices
            The light curves to unmark, in any of the forms `mark_for_removal` accepts. Defaults to
            `None` for all of them.
        """
        for i in range(self.size) if indices is None else self._resolve_indices(indices):
            self.data[i].marked = False

    def remove_marked(self) -> None:
        """Remove the light curves marked for removal from the group, in place.

        The group shrinks and every index derived from the order, such as `pbids`, `wnids`, `piis`,
        and `lcslices`, is renumbered the next time it is asked for, which is why the removal should
        happen before the group is handed to a `BaseLPF`. Removing every light curve leaves a legal
        empty group.

        Unlike marking, removal is not shared: the light curves are dropped from this group only,
        and any other group holding them keeps them, still marked, until its own `remove_marked`.
        """
        self.data = [d for d in self.data if not d.marked]

    def remove_outliers(self, nsigma: float = 3.0, width: int = 15) -> int:
        """Remove the outlying flux points from every light curve, in place.

        Each light curve is clipped separately against its own running median, see
        `LCData.remove_outliers`. Returns the total number of points removed. The light curves
        themselves are always kept, even if one loses every point; use `remove_marked` to drop
        whole light curves.

        Parameters
        ----------
        nsigma
            Clipping threshold in units of the robust residual scatter.
        width
            Width of the median filter in points. Must be a positive odd integer.
        """
        return sum(d.remove_outliers(nsigma, width) for d in self.data)

    # Covariates
    # ----------
    def add_time_covariates(self, order: int = 1) -> None:
        """Add the normalised time and its powers as covariates to every light curve, in place.

        Each light curve's time is mapped onto -1 ... 1 separately, see `LCData.add_time_covariates`.

        Parameters
        ----------
        order
            Highest power of the normalised time to add. Must be a positive integer.
        """
        for d in self.data:
            d.add_time_covariates(order)

    # Plotting
    # --------
    def plot(self, ncols: int = 5, figsize: Optional[tuple] = None,
             passbands: Optional[Union[str, Sequence[str]]] = None,
             instruments: Optional[Union[str, Sequence[str]]] = None,
             sectors: Optional[Union[int, Sequence[int]]] = None,
             pids: Optional[Union[int, Sequence[int]]] = None,
             annotate: bool = True, show_index: bool = True, show_xticks: bool = True,
             show_median: bool = False, median_width: int = 15,
             nsigma: Union[float, Sequence[float]] = 3.0, show_linear_model: bool = False,
             median_kwargs: Optional[dict] = None, linear_model_kwargs: Optional[dict] = None,
             errorbars: bool = False, xoffset: Optional[float] = None,
             ylim: Optional[tuple] = None, alpha: float = 0.5, **kwargs):
        """Plot the light curves in a grid of subplots.

        The panels share their y limits so the transit depths can be compared by eye, but not
        their x limits, since the light curves generally cover different times. Each panel's
        time axis is offset by its own zero point by default, keeping the tick labels short. The
        light curves marked for removal are drawn on a light gray background, `show_median` overlays
        the running median with its n-sigma limits for spotting outlying points, and
        `show_linear_model` overlays the variability the covariates can explain.

        Parameters
        ----------
        ncols
            Number of columns in the subplot grid, clipped to the number of light curves.
        figsize
            Figure size in inches. Defaults to a 13-inch-wide figure 2.5 inches per row tall.
        passbands, instruments, sectors, pids
            Plot only the light curves matching these criteria. Each takes a single value or a
            sequence of accepted values, and `None` means no filtering. A `pids` criterion never
            selects a light curve with unspecified transiting planets.
        annotate
            Show the instrument name and the passband in the upper right corner of each panel.
        show_index
            Show each light curve's index in the group in the upper left corner of its panel. This
            is the index `mark_for_removal` takes, and it is the position in the group `plot` was
            called on even when the panels are filtered.
        show_xticks
            Draw the x axis ticks and labels. Switching them off packs more light curves onto the
            screen when eyeballing the data.
        show_median
            Overlay the running median of the flux and its n-sigma limits, for spotting the points
            `remove_outliers` would clip.
        median_width
            Width of the median filter in points. Must be a positive odd integer.
        nsigma
            Limits to draw around the running median, as a multiple of the robust MAD scatter of
            the residuals from it. Either a single number or a sequence of them, in which case one
            band is drawn per value.
        show_linear_model
            Overlay the least-squares linear model of the flux in terms of the covariates, showing
            how much of the variability the covariates can explain. Light curves without covariates
            are left alone, see `LCData.linear_model`.
        median_kwargs, linear_model_kwargs
            Line properties (`c`, `lw`, `alpha`, `zorder`, and anything else
            `matplotlib.axes.Axes.plot` takes) for the running median and the linear model. They
            override the defaults, which draw both lines on top of the flux points. The n-sigma
            bands take their colour from `median_kwargs`.
        errorbars
            Plot the flux uncertainties as error bars. The uncertainties fall back to the
            estimated point-to-point scatter for the light curves without explicit errors, see
            `LCData.error` and `has_errors`.
        xoffset
            Time subtracted from the plotted times. Defaults to `None` for a per-panel offset,
            and `0.0` plots the times as they are.
        ylim
            Y limits shared by all the panels. Defaults to `None` for automatic limits.
        alpha
            Opacity of the plotted flux.
        **kwargs
            Passed to `matplotlib.axes.Axes.plot` or `matplotlib.axes.Axes.errorbar`.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If the group is empty or nothing matches the given criteria.

        Examples
        --------
        ::

            lcs.plot(ncols=4, passbands=['g', 'r'], instruments='MuSCAT2')
        """
        if ncols < 1:
            raise ValueError(f"'ncols' must be at least one, got {ncols!r}.")

        lcs = self.select(passband=passbands, instrument=instruments, sector=sectors, pids=pids)
        nlc = lcs.size
        if nlc == 0:
            given = {k: v for k, v in (('passbands', passbands), ('instruments', instruments),
                                       ('sectors', sectors), ('pids', pids)) if v is not None}
            raise ValueError(f"No light curves to plot matching {given}." if given else
                             "No light curves to plot: the group is empty.")

        # The position of each light curve in this group, kept through the filter so that the index
        # shown in a panel is the one 'mark_for_removal' takes.
        lcids = {id(d): i for i, d in enumerate(self.data)}

        ncols = min(ncols, nlc)
        nrows = int(ceil(nlc / ncols))
        fig, axs = subplots(nrows, ncols, figsize=figsize or (13, 2.5 * nrows),
                            constrained_layout=True, sharey='all', squeeze=False)

        kwargs.setdefault('marker', '.')
        kwargs.setdefault('ls', '')

        for i, lc in enumerate(lcs):
            ax = axs.flat[i]
            if lc.marked:
                ax.set_facecolor('0.9')

            t0 = (floor(lc.time.min()) if lc.size else 0.0) if xoffset is None else xoffset
            if errorbars:
                ax.errorbar(lc.time - t0, lc.flux, lc.error, alpha=alpha, **kwargs)
            else:
                ax.plot(lc.time - t0, lc.flux, alpha=alpha, **kwargs)

            if show_index:
                ax.text(0.02, 0.95, str(lcids[id(lc)]), ha='left', va='top', size='small',
                        transform=ax.transAxes)

            if annotate:
                label = '+'.join(lc.passband)
                if lc.instrument:
                    label = f"{lc.instrument}\n{label}"
                ax.text(0.98, 0.95, label, ha='right', va='top', size='small', transform=ax.transAxes)

            # Drawn after the flux so that the data stays the panel's first line, and with a zorder
            # above it so that the overlays stay visible over dense points.
            if show_median and lc.size:
                style = _line_style(_MEDIAN_STYLE, median_kwargs)
                m = lc.running_median(median_width)
                sigma = _mad_sigma(lc.flux - m)
                fc = style.get('color', style.get('c', 'k'))
                for n in atleast_1d(nsigma):
                    ax.fill_between(lc.time - t0, m - n * sigma, m + n * sigma, fc=fc, alpha=0.15,
                                    zorder=-100)
                ax.plot(lc.time - t0, m, **style)

            if show_linear_model and lc.size and lc.ncov:
                ax.plot(lc.time - t0, lc.linear_model(),
                        **_line_style(_LINEAR_MODEL_STYLE, linear_model_kwargs))

            if xoffset is None:
                setp(ax, xlabel=f"Time - {t0:.0f} [BJD]")

        if xoffset is not None:
            setp(axs[-1, :], xlabel="Time [BJD]" if xoffset == 0.0 else f"Time - {xoffset:.0f} [BJD]")
        if not show_xticks:
            setp(axs, xticks=[], xlabel='')
        setp(axs[:, 0], ylabel='Normalised flux')
        if ylim is not None:
            setp(axs, ylim=ylim)

        for ax in axs.flat[nlc:]:
            ax.remove()
        return fig

    def __repr__(self) -> str:
        marked = f", {self.n_marked} marked for removal" if self.n_marked else ""
        return (f"LCDataGroup with {self.size} light curves, {int(self.npts.sum())} points, "
                f"passbands {self.passband_names}{marked}")


LCData._group_type = LCDataGroup
