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

from typing import Union, List, Iterable

from numpy import inf, repeat, pi, atleast_2d, arctan2, sqrt, squeeze, ndarray, zeros
from ..models.numba.phasecurves import doppler_boosting, ellipsoidal_variation, emission, lambert_phase_function

from .. import QuadraticModel, EclipseModel
from ..orbits import as_from_rhop, i_from_ba
from ..param import GParameter, PParameter, NormalPrior as NP, UniformPrior as UP
from .lpf import BaseLPF, map_ldc


class PhaseCurveLPF(BaseLPF):
    def _init_data(self, times: Union[List, ndarray], fluxes: Union[List, ndarray], pbids: Union[List, ndarray] = None,
                   covariates: Union[List, ndarray] = None, errors: Union[List, ndarray] = None,
                   wnids: Union[List, ndarray] = None,
                   nsamples: Union[int, ndarray, Iterable] = 1, exptimes: Union[float, ndarray, Iterable] = 0.):

        super()._init_data(times=times, fluxes=fluxes, pbids=pbids, covariates=covariates,
                           errors=errors, wnids=wnids, nsamples=nsamples, exptimes=exptimes)

        pbis = self.pbids[self.lcids]
        self.pbmasks = []
        for i in range(self.npb):
            self.pbmasks.append(pbis == i)

    def _post_initialisation(self):
        self.tm = QuadraticModel(interpolate=False)
        self.em = EclipseModel()
        self.tm.set_data(self.timea - self._tref, self.lcids, self.pbids, self.nsamples, self.exptimes)
        self.em.set_data(self.timea - self._tref, self.lcids, self.pbids, self.nsamples, self.exptimes)

    def _init_p_orbit(self):
        porbit = [
            GParameter('tc', 'zero epoch', 'd', NP(0.0, 0.1), (-inf, inf)),
            GParameter('p', 'p', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact parameter', 'R_s', UP(0.0, 1.0), (0, 1)),
            GParameter('secw', 'sqrt(e) cos(w)', '', UP(-0.5, 0.5), (-1, 1)),
            GParameter('sesw', 'sqrt(e) sin(w)', '', UP(-0.5, 0.5), (-1, 1))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        """Planet parameter initialisation.
        """
        pk2 = [PParameter('k2', 'area ratio', 'A_s', UP(0.01 ** 2, 0.2 ** 2), (0, inf))]
        self.ps.add_global_block('k2', pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, 1)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice
        self._ix_k2 = self._start_k2

        pph = [GParameter(f'oev', f'Ellipsoidal variation offset', '', NP(0, 1e-6), (-inf, inf))]
        for pb in self.passbands:
            pph.extend([GParameter(f'aev_{pb}', f'Ellipsoidal variation amplitude in {pb}', '', UP(0, 1), (0, inf)),
                        GParameter(f'adb_{pb}', f'Doppler boosting amplitude in {pb}', '', UP(0, 1), (0, inf)),
                        GParameter(f'ted_{pb}', f'Day-side flux ratio in {pb}', '', UP(0.0, 0.2), (-inf, inf)),
                        GParameter(f'ten_{pb}', f'Night-side flux ratio in {pb}', '', UP(0.0, 0.1), (-inf, inf)),
                        GParameter(f'teo_{pb}', 'Thermal emission offset', 'rad', UP(-pi, pi), (-inf, inf)),
                        GParameter(f'ag_{pb}', f'Geometric albedo in {pb}', '', UP(0, 1), (0, 1))])
        self.ps.add_global_block('phase', pph)
        self._pid_fr = repeat(self.ps.blocks[-1].start, 1)
        self._start_fr = self.ps.blocks[-1].start
        self._sl_fr = self.ps.blocks[-1].slice

    def map_pv(self, pv):
        pv = atleast_2d(pv)
        t0 = pv[:, 0]
        p = pv[:, 1]
        a = as_from_rhop(pv[:, 2], p)
        inc = i_from_ba(pv[:, 3], a)
        ecc = pv[:, 4] ** 2 + pv[:, 5] ** 2
        omega = arctan2(pv[:, 5], pv[:, 4])
        area_ratio = pv[:, 6]
        k = sqrt(pv[:, 6:7])
        ldc = map_ldc(pv[:, self._sl_ld])
        sle = self._sl_fr
        oev = pv[:, sle][:, 0]
        aev = pv[:, sle][:, 1::6]
        adb = pv[:, sle][:, 2::6]
        dte = pv[:, sle][:, 3::6]
        nte = pv[:, sle][:, 4::6]
        ote = pv[:, sle][:, 5::6]
        ag  = pv[:, sle][:, 6::6]
        return t0, p, a, inc, ecc, omega, area_ratio, k, ldc, oev, aev, adb, dte, nte, ote, ag

    def doppler_boosting(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, oev, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        npv, npt = inc.size, self.ofluxa.size
        fdb = zeros((npv, npt))
        for ipb, pbm in enumerate(self.pbmasks):
            fdb[:, pbm] = doppler_boosting(adb[:, ipb], t0, p, True, self.timea[pbm])
        return squeeze(fdb)

    def ellipsoidal_variations(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, oev, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        npv, npt = inc.size, self.ofluxa.size
        fev = zeros((npv, npt))
        for ipb, pbm in enumerate(self.pbmasks):
            fev[:, pbm] = ellipsoidal_variation(aev[:, ipb], t0, p, oev, True, self.timea[pbm])
        return squeeze(fev)

    def thermal_flux(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, oev, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        npv, npt = inc.size, self.ofluxa.size
        fec = self.em.evaluate(k, t0, p, a, inc, ecc, omega, multiplicative=True)
        ft = zeros((npv, npt))
        for ipb, pbm in enumerate(self.pbmasks):
            ft[:, pbm] = emission(area_ratio, nte[:, ipb], dte[:, ipb], ote[:, ipb], t0, p, True, self.timea[pbm])
        return fec * squeeze(ft)

    def reflected_flux(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, oev, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        npv, npt = inc.size, self.ofluxa.size
        fec = self.em.evaluate(k, t0, p, a, inc, ecc, omega, multiplicative=True)
        fr = zeros((npv, npt))
        for ipb, pbm in enumerate(self.pbmasks):
            fr[:, pbm] = lambert_phase_function(a, area_ratio, ab[:, ipb], t0, p, True, self.timea[pbm])
        return fec * squeeze(fr)

    def transit_model(self, pv, copy=True):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, oev, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        npv, npt = inc.size, self.ofluxa.size

        fec = self.em.evaluate(k, t0 - self._tref, p, a, inc, ecc, omega, multiplicative=True)
        ft = zeros((npv, npt))
        fr = zeros((npv, npt))
        for ipb, pbm in enumerate(self.pbmasks):
            ft[:, pbm] = emission(area_ratio, nte[:, ipb], dte[:, ipb], ote[:, ipb], t0, p, True, self.timea[pbm])
            fr[:, pbm] = lambert_phase_function(a, area_ratio, ab[:, ipb], t0, p, True, self.timea[pbm])
        fplanet = fec * squeeze((ft + fr))

        ftr = self.tm.evaluate(k, ldc, t0 - self._tref, p, a, inc, ecc, omega)
        fev = zeros((npv, npt))
        fdb = zeros((npv, npt))
        for ipb, pbm in enumerate(self.pbmasks):
            fev[:, pbm] = ellipsoidal_variation(aev[:, ipb], t0, p, oev, True, self.timea[pbm])
            fdb[:, pbm] = doppler_boosting(adb[:, ipb], t0, p, True, self.timea[pbm])
        fstar = squeeze(ftr + fev + fdb)
        return fplanet + fstar

    def flux_model(self, pv):
        baseline = self.baseline(pv)
        model_flux = self.transit_model(pv)
        return squeeze(model_flux * atleast_2d(baseline))
