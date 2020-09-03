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

from numpy import asarray, unique, zeros, inf, squeeze, zeros_like, isfinite, log10, diff, sqrt
from astropy.stats import mad_std

try:
    from celerite import GP
    from celerite.terms import Matern32Term
    with_celerite = True
except ImportError:
    with_celerite = False

from ...param import  LParameter, NormalPrior as NP, UniformPrior as UP


class MultiCeleriteLogLikelihood:
    def __init__(self, lpf, name: str = 'gp', lcids=None, fixed_hps=None):
        if not with_celerite:
            raise ImportError("MultiCeleriteLogLikelihood requires celerite.")

        self.name = name
        self.lpf = lpf

        if fixed_hps is None:
            self.free = True
        else:
            self.hps = asarray(fixed_hps)
            self.free = False

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising CeleriteLogLikelihood.')
        self.lcids = lcids if lcids is not None else unique(lpf.lcids)
        self.lcslices = [lpf.lcslices[i] for i in self.lcids]
        self.nlc = len(self.lcslices)

        self.times, self.fluxes, self.wns = [], [], []
        for i in self.lcids:
            self.times.append(lpf.times[i])
            self.fluxes.append(lpf.fluxes[i])
            self.wns.append(diff(lpf.fluxes[i]).std()/sqrt(2))

        self.gps = [GP(Matern32Term(0, 0)) for i in range(self.nlc)]

        if self.free:
            self.init_parameters()
        else:
            self.compute_gp(None, force=True, hps=self.hps)

    def init_parameters(self):
        name = self.name
        pgp = [LParameter(f'{name}_ln_out', f'{name} ln output scale', '', NP(-6.5, 0.5), bounds=(-inf, inf)),
               LParameter(f'{name}_ln_in', f'{name} ln input scale', '', UP(-8, 8), bounds=(-inf, inf))]
        self.lpf.ps.thaw()
        self.lpf.ps.add_global_block(self.name, pgp)
        self.lpf.ps.freeze()
        self.pv_slice = self.lpf.ps.blocks[-1].slice
        self.pv_start = self.lpf.ps.blocks[-1].start
        setattr(self.lpf, f"_sl_{name}", self.pv_slice)
        setattr(self.lpf, f"_start_{name}", self.pv_start)

    def compute_gp(self, pv, force: bool = False, hps=None):
        if self.free or force:
            parameters = pv[self.pv_slice] if hps is None else hps
            for i, gp in enumerate(self.gps):
                gp.set_parameter_vector(parameters)
                gp.compute(self.times[i], yerr=self.wns[i])

    def compute_gp_lnlikelihood(self, pv, model):
        self.compute_gp(pv)
        lnlike = 0.0
        for i, gp in enumerate(self.gps):
            lnlike += gp.log_likelihood(self.fluxes[i] - model[self.lcslices[i]])
        return lnlike

    def predict_baseline(self, pv):
        self.compute_gp(pv)
        residuals = self.lpf.ofluxa - squeeze(self.lpf.transit_model(pv))
        bl = zeros_like(self.lpf.timea)
        for i, gp in enumerate(self.gps):
            sl = self.lcslices[i]
            bl[sl] = gp.predict(residuals[sl], self.times[i], return_cov=False)
        return 1. + bl

    def __call__(self, pvp, model):
        if pvp.ndim == 1:
            lnlike = self.compute_gp_lnlikelihood(pvp, model)
        else:
            lnlike = zeros(pvp.shape[0])
            for ipv, pv in enumerate(pvp):
                if all(isfinite(model[ipv])):
                    lnlike[ipv] = self.compute_gp_lnlikelihood(pv, model[ipv])
                else:
                    lnlike[ipv] = -inf
        return lnlike