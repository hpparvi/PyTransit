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

"""Contaminated TESS LPF

This module contains the log posterior function (LPF) for a TESS light curve with possible third-light
contamination from an unresolved source inside the photometry aperture.
"""
from pathlib import Path
from typing import Union

from numpy import repeat, inf, where, newaxis, squeeze, atleast_2d, isfinite, concatenate, zeros
from numpy.random import uniform
from uncertainties import ufloat, UFloat
from ldtk import tess

from ..tesslpf import TESSLPF
from ...param import UniformPrior as UP, NormalPrior as NP, PParameter


class CTESSLPF(TESSLPF):
    """Contaminated TESS LPF

    This class implements a log posterior function for a TESS light curve that allows for unknown flux contamination.
    The amount of flux contamination is not constrained.
    """

    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2_true', 'true_area_ratio', 'A_s', UP(0.10 ** 2, 0.75 ** 2), (0.10 ** 2, 0.75 ** 2)),
               PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.10 ** 2, 0.50 ** 2), (0.10 ** 2, 0.50 ** 2))]
        ps.add_passband_block('k2', 1, 2, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        self.add_prior(lambda pv: where(pv[:, 5] < pv[:, 4], 0, -inf))

    def transit_model(self, pv):
        pv = atleast_2d(pv)
        flux = super().transit_model(pv)
        cnt = 1. - pv[:, 5] / pv[:, 4]
        return squeeze(cnt[:, newaxis] + (1. - cnt[:, newaxis]) * flux)

    def create_pv_population(self, npop=50):
        pvp = zeros((0, len(self.ps)))
        npv, i = 0, 0
        while npv < npop and i < 10:
            pvp_trial = self.ps.sample_from_prior(npop)
            pvp_trial[:, 5] = pvp_trial[:, 4]
            cref = uniform(0, 0.99, size=npop)
            pvp_trial[:, 4] = pvp_trial[:, 5] / (1. - cref)
            lnl = self.lnposterior(pvp_trial)
            ids = where(isfinite(lnl))
            pvp = concatenate([pvp, pvp_trial[ids]])
            npv = pvp.shape[0]
            i += 1
        pvp = pvp[:npop]
        return pvp
