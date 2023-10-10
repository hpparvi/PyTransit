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
from pathlib import Path
from typing import Union, Sequence, Optional

import pandas as pd

from numpy import sqrt, inf, repeat, atleast_2d, array, ndarray

from .. import TransitModel
from .lpf import BaseLPF, map_ldc
from ..orbits.orbits_py import as_from_rhop, i_from_ba, epoch
from ..param.parameter import GParameter
from ..param.parameter import UniformPrior as UP, NormalPrior as NP


class TTVLPF(BaseLPF):
    """Log posterior function for TTV estimation for a single planet.
    """

    def __init__(self, name: str, zero_epoch: float, period: float,
                 passbands: Union[Sequence[str], str],
                 times: Optional[Sequence[ndarray]] = None,
                 fluxes: Optional[Sequence[ndarray]] = None,
                 errors: Optional[Sequence[ndarray]] = None,
                 pbids: Optional[Sequence[int]] = None,
                 covariates: Optional[Sequence[ndarray]] = None,
                 wnids: Optional[Sequence[int]] = None,
                 tm: Optional[TransitModel] = None,
                 nsamples: Union[Sequence[int], int] = 1,
                 exptimes: Union[Sequence[float], float] = 0.0,
                 init_data: bool = True,
                 result_dir: Optional[Path] = None,
                 tref: float = 0.0,
                 lnlikelihood: str = 'wn'):

        self.zero_epoch = zero_epoch
        self.period = period
        super().__init__(name, passbands, times, fluxes, errors, pbids, covariates, wnids, tm, nsamples, exptimes,
                         init_data, result_dir, tref, lnlikelihood)

    def _post_data_init_hook(self):
        super()._post_data_init_hook()
        epochs = epoch(array([t.mean() for t in self.times]), self.zero_epoch, self.period)
        ueps = []
        for ep in epochs:
            if ep not in ueps:
                ueps.append(ep)
        self.epochs = array(ueps)
        self.epids = pd.Categorical(epochs, categories=ueps).codes
        self.neps = self.epochs.size
        self.tm.epids = self.epids

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]
        self.ps.add_global_block('orbit', porbit)

        ptc = [GParameter(f'tc_{i}', f'transit_center_{i}', '-', NP(0.0, 0.1), (-inf, inf)) for i in
               range(self.neps)]
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
