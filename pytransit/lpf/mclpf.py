#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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

from numpy import repeat, atleast_2d, sqrt, inf

from ..orbits import as_from_rhop, i_from_ba
from ..param import PParameter, UniformPrior as UP
from .lpf import BaseLPF, map_ldc


class MCLPF(BaseLPF):

    def _init_p_planet(self):
        """Set up the planet parametrization

        Set up the radius ratio parametrization, one radius ratio per passband.
        """
        pk2 = [PParameter(f'k2_{pb}', f'{pb} area ratio', 'A_s', UP(0.07 ** 2, 0.12 ** 2), (0, inf)) for pb in
               self.passbands]
        self.ps.add_passband_block('k2', 1, self.npb, pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice

    def transit_model(self, pv, copy=True):
        """Calculate the transit model

        We need to override the default `BaseLPF.transit_model` method since it assumes passband-independent
        (achromatic) radius ratios. We have one free area ratio parameter per passband, and the mapping
        from the parameter array to transit model parameters needs to be adjusted.
        """
        pv = atleast_2d(pv)
        ldc = map_ldc(pv[:, self._sl_ld])
        radius_ratios = sqrt(pv[:, 4:4 + self.npb])
        zero_epoch = pv[:, 0] - self._tref
        period = pv[:, 1]
        smaxis = as_from_rhop(pv[:, 2], period)
        inclination = i_from_ba(pv[:, 3], smaxis)
        return self.tm.evaluate(radius_ratios, ldc, zero_epoch, period, smaxis, inclination)
