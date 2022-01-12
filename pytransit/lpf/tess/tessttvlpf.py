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

from numpy import inf, repeat, atleast_2d, sqrt, arange

from ..lpf import map_ldc
from ..tesslpf import TESSLPF
from ...orbits import as_from_rhop, i_from_ba
from ...param import GParameter, NormalPrior as NP, UniformPrior as UP


class TESSTTVLPF(TESSLPF):

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]
        self.ps.add_global_block('orbit', porbit)

        ptc = [GParameter(f'tc_{i}', f'transit_center_{i}', '-', NP(0.0, 0.1), (-inf, inf)) for i in range(self.nlc)]
        self.ps.add_global_block('tc', ptc)
        self._pid_tc = repeat(self.ps.blocks[-1].start, self.nlc)
        self._start_tc = self.ps.blocks[-1].start
        self._sl_tc = self.ps.blocks[-1].slice

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv)
        ldc = map_ldc(pv[:, self._sl_ld])
        zero_epoch = pv[:, self._sl_tc] - self._tref
        period = pv[:, 0]
        smaxis = as_from_rhop(pv[:, 1], period)
        inclination = i_from_ba(pv[:, 2], smaxis)
        radius_ratio = sqrt(pv[:, self._sl_k2])
        return self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)

    def _post_initialisation(self):
        super()._post_initialisation()
        self.tm.epids = arange(self.nlc)