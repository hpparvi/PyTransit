#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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
from typing import Union, Optional

from numpy import ones, ndarray, asarray, zeros, unique, atleast_1d, issubdtype, integer, float64

from ..orbits.orbits_py import ta_ip_calculate_table


class TransitModel(object):
    """Exoplanet transit light curve model 
    """

    def __init__(self) -> None:

        # Declare the basic arrays
        # ------------------------
        self.time: ndarray = None
        self.lcids: ndarray = None
        self.pbids: ndarray = None
        self.nsamples: ndarray = None
        self.exptimes: ndarray = None

        # Interpolation table for eccentric orbits
        # ----------------------------------------
        self._tae, self._es, self._ms = None, None, None
        self.init_orbit_table()

    def set_data(self, time: ndarray, lcids: Optional[ndarray] = None, pbids: Optional[ndarray] = None,
                 nsamples: Optional[ndarray]  = None, exptimes: Optional[ndarray] = None,
                 epids: Optional[ndarray] = None) -> None:
        """Set the data for the transit model.

        Parameters
        ----------
        time : array-like
            Array of mid-exposure times for which the model will be evaluated.
        lcids : array-like, optional
            Array of integer light curve indices. Must have the same size as the time array.
        pbids : array-like, optional
            Array of passband indices, one per light curve. Must satisfy `pbids.size == unique(lcids).size`.
        nsamples : int or array-like, optional
            Number of samples per exposure. Can either be an integer, in which case all the light curves will have the
            same supersampling rate, or an array of integers, in which case each light curve can have a different rate.
        exptimes : float or array-like, optional
            Exposure times, again either for all the modelled data, or one value per light curve.
        epids : array-like, optional
            Epoch indices that can be used to link a light curve to a specific zero epoch and period (for TTV calculations).
        """

        # Time samples
        # ------------
        self.time     = asarray(time, float64)
        self.npt      = self.time.size

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
        # The passband a light curve belongs to.
        self.pbids    = asarray(pbids) if pbids is not None else zeros(self.nlc, 'int')
        self.npb = unique(self.pbids).size

        if not issubdtype(self.pbids.dtype, integer):
            raise ValueError(f"The passband indices must be given as integers instead of {self.pbids.dtype}.")

        if self.pbids.size != self.nlc:
            raise ValueError(f"Passband index array size ({self.pbids.size}) should equal to the number of ligt curves ({self.nlc}).")

        if not (pbids.max() == (self.npb-1) and pbids.min() == 0):
            raise ValueError(f"Passband indices (`pbids`) for {self.npb} unique passbands should be given as integers between 0 and {self.npb - 1}.")

        # Epoch indices
        # -------------
        self.epids = asarray(epids) if epids is not None else zeros(self.nlc, 'int')

        # Supersampling
        # -------------
        # A number of samples and the exposure time for each light curve.
        self.nsamples = atleast_1d(nsamples) if nsamples is not None else ones(self.nlc, 'int')
        self.exptimes = atleast_1d(exptimes) if exptimes is not None else zeros(self.nlc, 'int')

    def init_orbit_table(self, ne: int = 256, nm: int = 512):
        self._tae, self._es, self._ms = ta_ip_calculate_table(ne, nm)

    def __call__(self, *nargs, **kwargs):
        raise NotImplementedError

    # Default evaluation methods
    # --------------------------
    def evaluate(self, k: Union[float, ndarray], ldc: ndarray, t0: Union[float, ndarray], p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Union[float, ndarray] = None, w: Union[float, ndarray] = None,
                 copy: bool = True) -> ndarray:
        raise NotImplementedError

    def evaluate_ps(self, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0., copy: bool = True) -> ndarray:
        raise NotImplementedError

    def evaluate_pv(self, pvp: ndarray, ldc: ndarray, copy: bool = True) -> ndarray:
        raise NotImplementedError

    # Evaluation given an array of times
    # ----------------------------------
    def evaluate_t_ps(self, t: ndarray, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0.) -> ndarray:
        raise NotImplementedError

    def evaluate_t_pv(self, t: ndarray, pvp: ndarray) -> ndarray:
        raise NotImplementedError

    # Evaluation given an array of normalised distances
    # -------------------------------------------------
    def evaluate_z_ps(self, z: ndarray, k: float, ldc: ndarray) -> ndarray:
        raise NotImplementedError

    def evaluate_z_pv(self, z: ndarray, pvp: ndarray) -> ndarray:
        raise NotImplementedError
