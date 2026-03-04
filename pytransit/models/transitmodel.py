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
import jax

from typing import Literal

from numpy import (ones, ndarray, asarray, zeros, unique, atleast_1d, issubdtype, integer, float64, array, full, isscalar)


class TransitModel:
    """Exoplanet transit light curve model.
    """
    def __init__(self, backend: Literal["numba", "jax"] = "numba", return_grad: bool = False,
                 parallel: bool = False, n_threads: int | None = None, **kwargs):
        self.backend = backend
        self.return_grad = return_grad
        self.n_threads = n_threads
        self.parallel = parallel

        self.times = None
        self.lcids = None
        self.pbids = None
        self.epids = None
        self.nsamples = None
        self.exptimes = None

        self.nlc: int = 0
        self.npt: int = 0
        self.npb: int = 0
        self.ntc: int = 0
        self.nor: int = 0

        self.simple = True
        self._time_id: int | None = None

        self._init_model()

    def _init_model(self):
        raise NotImplementedError

    def set_data(self,
                 times: ndarray | list,
                 lcids: ndarray | list | None = None,
                 pbids: ndarray | list | None = None,
                 epids: ndarray | list | None = None,
                 nsamples: ndarray | list | None  = None,
                 exptimes: ndarray | list | None = None,
                 include_orbit_variations: bool = False) -> None:

        if id(times) == self._time_id and lcids is None and pbids is None and nsamples is None and exptimes is None and epids is None:
            return

        self._time_id = id(times)
        self.times    = array(times, float64)
        self.npt      = self.times.size

        # Light curve indices
        # -------------------
        # The light curve a datapoint belongs to.
        self.lcids    = asarray(lcids) if lcids is not None else zeros(self.npt, 'int')
        self.nlc = unique(self.lcids).size

        if not issubdtype(self.lcids.dtype, integer):
            raise ValueError(f"The light curve indices must be given as integers instead of {self.lcids.dtype}.")

        if self.lcids.size != self.npt:
            raise ValueError(f"Light curve index array size ({self.lcids.size}) should equal to the number of datapoints ({self.npt}).")

        # Passband indices
        # ----------------
        self.pbids = asarray(pbids) if pbids is not None else zeros(self.nlc, 'int')
        self.npb = unique(self.pbids).size

        if not issubdtype(self.pbids.dtype, integer):
            raise ValueError(f"The passband indices must be given as integers instead of {self.pbids.dtype}.")

        if self.pbids.size != self.nlc:
            raise ValueError(f"Passband index array size ({self.pbids.size}) should equal to the number of light curves ({self.nlc}).")

        if not (self.pbids.max() == (self.npb-1) and self.pbids.min() == 0):
            raise ValueError(f"Passband indices (`pbids`) for {self.npb} unique passbands should be given as integers between 0 and {self.npb - 1}.")

        # Epoch indices
        # -------------
        self.epids = asarray(epids) if epids is not None else zeros(self.nlc, 'int')
        self.ntc = unique(self.epids).size

        if include_orbit_variations:
            self.nor = self.ntc
        else:
            self.nor = 1

        if self.epids.size != self.nlc:
            raise ValueError(f"Epoch index array size ({self.epids.size}) should equal to the number of light curves ({self.nlc}).")

        if not (self.epids.max() == (self.ntc-1) and self.epids.min() == 0):
            raise ValueError(f"Epoch indices (`epids`) for {self.ntc} unique epochs should be given as integers between 0 and {self.ntc - 1}.")

        # Supersampling
        # -------------
        # A number of samples and the exposure time for each light curve.
        if nsamples is None:
            self.nsamples = ones(self.nlc, 'int')
        else:
            if isscalar(nsamples):
                self.nsamples = full(self.nlc, nsamples, 'int')
            else:
                self.nsamples = asarray(nsamples, 'int')
                if self.nsamples.min() < 1:
                    raise ValueError(f"The number of samples must be at least 1, but got {self.nsamples.min()}.")
                if self.nsamples.size != self.nlc:
                    raise ValueError(f"Number of samples array size ({self.nsamples.size}) should equal to the number of light curves ({self.nlc}).")

        if exptimes is None:
            self.exptimes = zeros(self.nlc, 'd')
        else:
            if isscalar(exptimes):
                self.exptimes = full(self.nlc, exptimes, 'd')
            else:
                self.exptimes = asarray(exptimes, 'd')
                if self.exptimes.min() < 0.0:
                    raise ValueError(f"The exposure time must be positive, but got {self.exptimes.min()}.")
                if self.exptimes.size != self.nlc:
                    raise ValueError(f"Exposure time array size ({self.exptimes.size}) should equal to the number of light curves ({self.nlc}).")

        if self.npb > 1 or self.ntc > 1:
            self.simple = False

        if self.backend == "jax":
            self.times = jax.device_put(self.times)
            self.lcids = jax.device_put(self.lcids)
            self.epids = jax.device_put(self.epids)
            self.nsamples = jax.device_put(self.nsamples)
            self.exptimes = jax.device_put(self.exptimes)

    def evaluate(self,
                 k: float | ndarray,
                 t0: float | ndarray,
                 p: float | ndarray,
                 a: float | ndarray,
                 i: float | ndarray,
                 e: float | ndarray = 0.0,
                 w: float | ndarray = 0.0,
                 ldp: ndarray | None = None) -> ndarray | tuple[ndarray, ndarray]:
        raise NotImplementedError

