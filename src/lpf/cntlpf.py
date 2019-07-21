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

from numpy import repeat, arange, atleast_2d, zeros_like, zeros, sqrt, where, isfinite, concatenate, inf
from numpy.random import uniform
from numba import njit, prange

from .lpf import BaseLPF
from ..param import PParameter, GParameter, UniformPrior as UP
from ..orbits.orbits_py import i_from_ba, as_from_rhop
from ..contamination.contamination import Instrument, SMContamination
from ..contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z

@njit(fastmath=True)
def map_ldc(ldc):
    ldc = atleast_2d(ldc)
    uv = zeros_like(ldc)
    a, b = sqrt(ldc[:, 0::2]), 2. * ldc[:, 1::2]
    uv[:, 0::2] = a * b
    uv[:, 1::2] = a * (1. - b)
    return uv


@njit(fastmath=True)
def map_pv_pclpf(pv):
    pv = atleast_2d(pv)
    pvt = zeros((pv.shape[0], 7))
    pvt[:, 0] = sqrt(pv[:, 5])
    pvt[:, 1:3] = pv[:, 0:2]
    pvt[:, 3] = as_from_rhop(pv[:, 2], pv[:, 1])
    pvt[:, 4] = i_from_ba(pv[:, 3], pvt[:, 3])
    return pvt


@njit
def contaminate(flux, cnt, lcids, pbids):
    npv = flux.shape[0]
    npt = flux.shape[1]
    for ipv in range(npv):
        for ipt in range(npt):
            c = cnt[ipv, pbids[lcids[ipt]]]
            flux[ipv, ipt] = c + (1.-c)*flux[ipv, ipt]
    return flux

class ContaminatedLPF(BaseLPF):
    def _init_p_planet(self):
        ps = self.ps
        pk2 = [PParameter('k2', 'area_ratio', 'A_s', UP(0.01**2, 0.5**2), (0.01**2, 0.5**2))]
        pcn = [PParameter('cnt_{}'.format(pb), 'contamination', '', UP(0., 1.), (0., 1.)) for pb in self.passbands]
        ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        ps.add_passband_block('contamination', 1, self.npb, pcn)
        self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        self._sl_cn = ps.blocks[-1].slice

    def transit_model(self, pv):
        pv = atleast_2d(pv)
        flux = super().transit_model(pv)
        return contaminate(flux, pv[:, self._sl_cn], self.lcids, self.pbids)


class PhysContLPF(BaseLPF):

    def _init_p_planet(self):
        ps = self.ps
        pk2 = [GParameter('k2_true', 'true_area_ratio', 'As', UP(0.01**2, 0.75**2), bounds=(1e-8, inf))]
        pcn = [PParameter('k2_app', 'apparent_area_ratio', 'A_s', UP(0.01 ** 2, 0.25 ** 2), (0.01 ** 2, 0.25 ** 2)),
               GParameter('teff_h', 'host_teff', 'K', UP(2500, 12000), bounds=(2500, 12000)),
               GParameter('teff_c', 'contaminant_teff', 'K', UP(2500, 12000), bounds=(2500, 12000))]
        ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(ps.blocks[-1].start, self.npb)
        self._start_k2 = ps.blocks[-1].start
        self._sl_k2 = ps.blocks[-1].slice
        ps.add_global_block('contamination', pcn)
        self._pid_cn = arange(ps.blocks[-1].start, ps.blocks[-1].stop)
        self._sl_cn = ps.blocks[-1].slice

    def additional_priors(self, pv):
        """Additional priors."""
        pv = atleast_2d(pv)
        return sum([f(pv) for f in self.lnpriors], 0)

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('example', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self.lnpriors.append(lambda pv: where(pv[:, 5] < pv[:, 4], 0, -inf))

    def transit_model(self, pvp):
        pvp = atleast_2d(pvp)
        cnt = zeros((pvp.shape[0], self.npb))
        pvt = map_pv_pclpf(pvp)
        ldc = map_ldc(pvp[:, self._sl_ld])
        flux = self.tm.evaluate_pv(pvt, ldc)
        for i, pv in enumerate(pvp):
            if (2500 < pv[6] < 12000) and (2500 < pv[7] < 12000):
                cnref = 1. - pv[5] / pv[4]
                cnt[i, :] = self.cm.contamination(cnref, pv[6], pv[7])
            else:
                cnt[i, :] = -inf
        return contaminate(flux, cnt, self.lcids, self.pbids)

    def create_pv_population(self, npop=50):
        pvp = zeros((0, len(self.ps)))
        npv, i = 0, 0
        while npv < npop and i < 10:
            pvp_trial = self.ps.sample_from_prior(npop)
            pvp_trial[:, 4] = pvp_trial[:, 5]
            cref = uniform(0, 0.99, size=npop)
            pvp_trial[:, 4] = pvp_trial[:, 5] / (1. - cref)
            lnl = self.lnposterior(pvp_trial)
            ids = where(isfinite(lnl))
            pvp = concatenate([pvp, pvp_trial[ids]])
            npv = pvp.shape[0]
            i += 1
        pvp = pvp[:npop]
        return pvp
