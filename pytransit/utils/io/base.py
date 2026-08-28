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

"""Shared machinery for the observation data containers.

`LCData`/`LCDataGroup` and `RVData`/`RVDataGroup` differ in the metadata
they carry but share their field validation and their container behaviour. The validators
here are plain functions each item class calls in order, so the classes stay flat and
readable, while `_Data` and `_DataGroup` hold the addition algebra and the container
protocol.
"""

import warnings

from collections.abc import Sequence
from typing import Callable, Optional, Union

from numpy import (ndarray, asarray, zeros, array, nan, isfinite, integer, bool_)


# Scalar conversion
# -----------------
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


# Field validation
# ----------------
def _validate_time(time, name: str = 'time') -> ndarray:
    """Return `time` as a finite 1D float64 array."""
    t = _as_float_array(time, name)
    if t.ndim != 1:
        raise ValueError(f"'{name}' must be a 1D array, got a {t.ndim}D one.")
    if not isfinite(t).all():
        raise ValueError(f"'{name}' contains non-finite values.")
    return t


def _validate_series(values, npt: int, name: str) -> ndarray:
    """Return `values` as a 1D float64 array of length `npt`, warning on non-finite values."""
    v = _as_float_array(values, name)
    if v.ndim != 1:
        raise ValueError(f"'{name}' must be a 1D array, got a {v.ndim}D one.")
    if v.size != npt:
        raise ValueError(f"'{name}' has {v.size} points but 'time' has {npt}.")
    if not isfinite(v).all():
        warnings.warn(f"'{name}' contains non-finite values.")
    return v


def _validate_error(error, npt: int, name: str = 'error') -> Optional[ndarray]:
    """Return `error` as a positive 1D float64 array of length `npt`, or None."""
    if error is None:
        return None
    e = _as_float_array(error, name)
    if e.ndim != 1:
        raise ValueError(f"'{name}' must be a 1D array, got a {e.ndim}D one.")
    if e.size != npt:
        raise ValueError(f"'{name}' has {e.size} points but 'time' has {npt}.")
    finite = isfinite(e)
    if not finite.all():
        warnings.warn(f"'{name}' contains non-finite values.")
    if (e[finite] <= 0.0).any():
        raise ValueError(f"'{name}' contains non-positive values.")
    return e


def _validate_covariates(covariates, npt: int, name: str = 'covariates') -> ndarray:
    """Return `covariates` as a 2D float64 array with `npt` rows, empty when None."""
    if covariates is None:
        return zeros((npt, 0))
    cv = _as_float_array(covariates, name)
    if cv.ndim == 1:
        cv = cv.reshape((-1, 1))
    elif cv.ndim != 2:
        raise ValueError(f"'{name}' must be a 1D or 2D array, got a {cv.ndim}D one.")
    if cv.shape[0] != npt:
        raise ValueError(f"'{name}' has {cv.shape[0]} rows but 'time' has {npt} points.")
    return cv


def _validate_names(names, name: str = 'passband') -> tuple:
    """Return `names` as a non-empty tuple of strings."""
    if isinstance(names, str):
        return (names,)
    try:
        out = tuple(str(n) for n in names)
    except TypeError as e:
        raise ValueError(f"'{name}' must be a string or a sequence of strings, "
                         f"got {names!r}.") from e
    if len(out) == 0:
        raise ValueError(f"'{name}' cannot be empty.")
    return out


def _validate_pids(pids, name: str = 'pids') -> Optional[tuple]:
    """Return `pids` as a tuple of unique non-negative ints, or None.

    Only `LCData` uses this: an RV signal always contains every planet in the
    system, so there is no per-dataset subset to name. It lives here to keep the validators
    in one place.
    """
    if pids is None:
        return None
    if isinstance(pids, str):
        raise ValueError(f"'{name}' must be an integer or a sequence of integers, "
                         f"got the string {pids!r}.")
    if isinstance(pids, (int, integer)) and not isinstance(pids, (bool, bool_)):
        pids = [pids]
    try:
        pl = tuple(_as_int(p, name) for p in pids)
    except TypeError as e:
        raise ValueError(f"'{name}' must be an integer or a sequence of integers, "
                         f"got {pids!r}.") from e
    if any(p < 0 for p in pl):
        raise ValueError(f"'{name}' cannot contain negative planet indices, got {pids!r}.")
    if len(set(pl)) != len(pl):
        raise ValueError(f"'{name}' contains duplicate planet indices: {pids!r}.")
    return pl


# Selection
# ---------
def _as_value_set(v) -> tuple:
    """Return `v` as a tuple of accepted values, treating a string or a scalar as one value."""
    if isinstance(v, str) or not isinstance(v, (Sequence, ndarray, set, frozenset)):
        return (v,)
    return tuple(v)


class _Data:
    """Addition algebra shared by the single-dataset containers.

    Subclasses set `_group_type` to their group class once it is defined.
    """

    _group_type = None

    def __add__(self, other):
        gt = self._group_type
        if isinstance(other, gt):
            return gt([self] + other.data)
        if isinstance(other, gt._item_type):
            return gt([self, other])
        return NotImplemented

    def __radd__(self, other):
        if other == 0 or other is None:
            return self._group_type([self])
        return NotImplemented


class _DataGroup:
    """Container protocol and metadata shared by the dataset groups.

    Subclasses set `_item_type` to the class they hold. The datasets keep their insertion
    order, which defines the dataset index a model will use. Nothing is sorted implicitly.
    """

    _item_type = None

    def __init__(self, data=()) -> None:
        self.data: list = []
        if isinstance(data, (self._item_type, _DataGroup)):
            data = [data]
        for d in data:
            self._add_data(d)

    def _add_data(self, d) -> None:
        if isinstance(d, _DataGroup):
            for x in d.data:
                self._add_data(x)
            return
        if not isinstance(d, self._item_type):
            raise TypeError(f"A {type(self).__name__} holds {self._item_type.__name__} objects, "
                            f"got {type(d).__name__}.")
        if any(d is x for x in self.data):
            raise ValueError(f"The same {self._item_type.__name__} instance cannot be added "
                             f"to a group twice.")
        self.data.append(d)

    # Bulk data
    # ---------
    @property
    def times(self) -> list:
        """List of 1D time arrays."""
        return [d.time for d in self.data]

    @property
    def covariates(self) -> list:
        """List of 2D covariate matrices."""
        return [d.covariates for d in self.data]

    @property
    def errors(self) -> list:
        """List of 1D uncertainty arrays."""
        return [d.error for d in self.data]

    # Metadata
    # --------
    @property
    def instruments(self) -> list:
        """List of instrument names."""
        return [d.instrument for d in self.data]

    # Sizes
    # -----
    @property
    def size(self) -> int:
        """Number of datasets."""
        return len(self.data)

    @property
    def npts(self) -> ndarray:
        """Array of per-dataset datapoint counts."""
        return array([d.size for d in self.data], int)

    @property
    def ncovs(self) -> ndarray:
        """Array of per-dataset covariate counts."""
        return array([d.ncov for d in self.data], int)

    @property
    def has_errors(self) -> bool:
        """True if every dataset was given an explicit uncertainty array."""
        return all(d.has_error for d in self.data) if self.data else False

    @property
    def has_covariates(self) -> bool:
        """True if any dataset has at least one covariate."""
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
        """Map tuples of per-dataset metadata into 0-based contiguous group ids.

        The ids are numbered by first appearance, so they are contiguous by construction,
        which `BaseLPF._init_data` asserts for the noise ids.
        """
        keys = list(zip(*fields))
        if not keys:
            return zeros(0, int)
        cats = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        return array([cats[k] for k in keys], int)

    @property
    def wnids(self) -> ndarray:
        """Noise block index per dataset, grouping the datasets by instrument."""
        return self._group_ids(self.instruments)

    @property
    def ins(self) -> list:
        """Instrument names, under the name `LinearModelBaseline` looks for."""
        return self.instruments

    @property
    def piis(self) -> ndarray:
        """Running index of each dataset within its instrument."""
        counts: dict = {}
        out = []
        for ins in self.instruments:
            out.append(counts.get(ins, 0))
            counts[ins] = out[-1] + 1
        return array(out, int)

    # Selection
    # ---------
    def select(self, **criteria) -> '_DataGroup':
        """Return a new group with the datasets matching all the given criteria.

        Each keyword names an attribute of the contained objects. A tuple-valued attribute
        such as `passband` or `pids` matches by membership, everything else by equality. A
        criterion can be a single value or a sequence of accepted values, in which case a
        dataset matches if it matches any of them, so `pids=[0, 2]` selects the datasets
        with a transiting planet 0 or 2, and an empty sequence selects nothing. A dataset
        with unspecified pids (`None`) is never selected by a `pids` criterion. Criteria
        given as `None` are ignored.
        """
        def matches(d) -> bool:
            for k, v in criteria.items():
                if v is None:
                    continue
                if not hasattr(d, k):
                    raise ValueError(f"Unknown selection criterion {k!r} for "
                                     f"{self._item_type.__name__}.")
                a, vs = getattr(d, k), _as_value_set(v)
                if isinstance(a, tuple):
                    if not any(x in a for x in vs):
                        return False
                elif a not in vs:
                    return False
            return True
        return type(self)([d for d in self.data if matches(d)])

    def sorted_by(self, key: Union[str, Callable] = 'time') -> '_DataGroup':
        """Return a new group ordered by `key`.

        Parameters
        ----------
        key
            'time', the name of any attribute of the contained objects, or a callable
            taking one of them and returning a sort key.
        """
        if callable(key):
            f = key
        elif key == 'time':
            f = lambda d: d.time.min() if d.size else nan
        else:
            if not self.data:
                return type(self)()
            if not hasattr(self.data[0], key):
                raise ValueError(f"Unknown sort key {key!r}, choose 'time', an attribute of "
                                 f"{self._item_type.__name__}, or pass a callable.")
            f = lambda d: getattr(d, key)
        return type(self)(sorted(self.data, key=f))

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
            return type(self)(self.data[index])
        ix = asarray(index)
        if ix.dtype == bool:
            if ix.size != self.size:
                raise IndexError(f"Boolean index has {ix.size} entries but the group has "
                                 f"{self.size} datasets.")
            return type(self)([d for d, m in zip(self.data, ix) if m])
        if ix.dtype.kind in 'iu':
            return type(self)([self.data[i] for i in ix])
        raise TypeError(f"Cannot index a {type(self).__name__} with {type(index).__name__}.")

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.data + other.data)
        if isinstance(other, self._item_type):
            return type(self)(self.data + [other])
        return NotImplemented

    def __radd__(self, other):
        if other == 0 or other is None:
            return type(self)(self.data)
        return NotImplemented
