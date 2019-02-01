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

from numpy import pi, sign, cos, sqrt, sin, array, arccos, inf, round, int, s_

from numba import njit, prange
from pytransit.lpf.lpf import BaseLPF
from pytransit.transitmodel import TransitModel
from pytransit.param.parameter import ParameterSet, PParameter, GParameter
from pytransit.param.parameter import UniformPrior as U, NormalPrior as N, GammaPrior as GM
from pytransit.utils.orbits import as_from_rhop


@njit("f8[:](f8[:], f8, f8, f8, f8[:], i8[:])", cache=False, parallel=False)
def z_circular_ttv(t, p, a, i, tc, tcid):
    cosph = cos(2*pi * (t - tc[tcid]) / p)
    z = sign(cosph) * a * sqrt(1.0 - cosph * cosph * sin(i) ** 2)
    return z


class TTVLPF(BaseLPF):
    """Log posterior function for TTV estimation.

    A log posterior function for TTV estimation. Each light curve represents a single transit, and
    is given a separate free transit centre parameter. The average orbital period and (one) transit
    zero epoch are assumed as known.

    Notes: The number of parameters can grow large with Kepler short-period planets.

    """
    def __init__(self, target: str, zero_epoch: float, period: float, tc_sigma: float, passbands: list,
                 times: list = None, fluxes: list = None, pbids: list = None, tm: TransitModel = None):
        self.zero_epoch = zero_epoch
        self.period = period
        self.tc_sigma = tc_sigma
        super().__init__(target, passbands, times, fluxes, pbids, tm)

    def _init_p_orbit(self):
        """Orbit parameter initialisation for a TTV model.
        """
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 25.0), (0, inf)),
                  GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]

        s = self.tc_sigma
        self.tnumber = round((array([t.mean() for t in self.times]) - self.zero_epoch) / self.period).astype(int)
        tcs = self.period * self.tnumber + self.zero_epoch
        for tc, tn in zip(tcs, self.tnumber):
            porbit.append(GParameter(f'tc_{tn:d}', f'transit_centre_{tn:d}', 'd', N(tc, s), (-inf, inf)))
        self.ps.add_global_block('orbit', porbit)
        self._start_tc = 2
        self._sl_tc = s_[self._start_tc:self._start_tc + self.nlc]

    def _compute_z(self, pv):
        a = as_from_rhop(pv[0], self.period)
        i = arccos(pv[1] / a)
        tc = pv[self._sl_tc]
        return z_circular_ttv(self.timea, self.period, a, i, tc, self.lcida)
