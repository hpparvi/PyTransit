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

from numpy import ones, ndarray, asarray, zeros, unique, atleast_1d

from ..orbits.orbits_py import ta_ip_calculate_table


class TransitModel(object):
    """Exoplanet transit light curve model 
    """

    methods = 'pars', 'pv'

    def __init__(self, method: str = 'pars', is_secondary: bool = False) -> None:
        """

        Parameters
        ----------
        method
        is_secondary
        """
        assert method in self.methods, f"Unknown TM evaluation method {method}."
        self.method = method
        self.is_secondary = is_secondary

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

    def set_data(self, time: ndarray,
                 lcids: ndarray = None, pbids: ndarray = None,
                 nsamples: ndarray = None, exptimes: ndarray = None):
        # Time samples
        # ------------
        self.time     = asarray(time)
        self.npt      = self.time.size

        # Light curve indices
        # -------------------
        # The light curve a datapoint belongs to.
        self.lcids    = asarray(lcids) if lcids is not None else zeros(self.npt, 'int')
        assert self.lcids.size == self.npt
        self.nlc = unique(self.lcids).size

        # Passband indices
        # ----------------
        # The passband a light curve belongs to.
        self.pbids    = asarray(pbids) if pbids is not None else zeros(self.nlc, 'int')
        assert self.pbids.size == self.nlc
        self.npb = unique(self.pbids).size

        # Supersampling
        # -------------
        # A number of samples and the exposure time for each light curve.
        self.nsamples = atleast_1d(nsamples) if nsamples is not None else ones(self.nlc, 'int')
        self.exptimes = atleast_1d(exptimes) if exptimes is not None else zeros(self.nlc, 'int')


    def init_orbit_table(self, ne: int = 256, nm: int = 512):
        self._tae, self._es, self._ms = ta_ip_calculate_table(ne, nm)

    def __call__(self, *nargs, **kwargs):
        if self.method == 'pars':
            return self.evaluate_ps(*nargs, **kwargs)
        else:
            return self.evaluate_pv(*nargs, **kwargs)


    # Default evaluation methods
    # --------------------------
    def evaluate_ps(self, k: float, ldc: ndarray, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0., copy: bool = True) -> ndarray:
        raise NotImplementedError

    def evaluate_pv(self, pvp: ndarray, copy: bool = True) -> ndarray:
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
