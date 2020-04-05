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

from numpy import asarray, unique, zeros, inf, squeeze, zeros_like, isfinite, log10, diff, sqrt
from astropy.stats import mad_std

try:
    from celerite import GP
    from celerite.terms import Matern32Term
    with_celerite = True
except ImportError:
    with_celerite = False

from ...param import  LParameter, NormalPrior as NP, UniformPrior as UP

class CeleriteLogLikelihood:
    def __init__(self, lpf, name: str = 'gp', noise_ids=None, fixed_hps=None):
        if not with_celerite:
            raise ImportError("CeleriteLogLikelihood requires celerite.")

        self.name = name
        self.lpf = lpf

        if fixed_hps is None:
            self.free = True
        else:
            self.hps = asarray(fixed_hps)
            self.free = False

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising CeleriteLogLikelihood.')
        self.noise_ids = noise_ids if noise_ids is not None else unique(lpf.noise_ids)

        self.mask = m = zeros(lpf.lcids.size, bool)
        for lcid,nid in enumerate(lpf.noise_ids):
            if nid in self.noise_ids:
                m[lcid == lpf.lcids] = 1

        if m.sum() == lpf.lcids.size:
            self.times = lpf.timea
            self.fluxes = lpf.ofluxa
        else:
            self.times = lpf.timea[m]
            self.fluxes = lpf.ofluxa[m]

        self.gp = GP(Matern32Term(0, 0))

        if self.free:
            self.init_parameters()
        else:
            self.compute_gp(None, force=True, hps=self.hps)

    def init_parameters(self):
        name = self.name
        wns = log10(mad_std(diff(self.fluxes)) / sqrt(2))
        pgp = [LParameter(f'{name}_ln_out', f'{name} ln output scale', '', NP(-6, 1.5), bounds=(-inf, inf)),
               LParameter(f'{name}_ln_in', f'{name} ln input scale', '', UP(-8, 8), bounds=(-inf, inf)),
               LParameter(f'{name}_log10_wn', f'{name} log10 white noise sigma', '', NP(wns, 0.025), bounds=(-inf, inf))]
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
            self.gp.set_parameter_vector(parameters[:-1])
            self.gp.compute(self.times, yerr=10 ** parameters[-1])

    def compute_gp_lnlikelihood(self, pv, model):
        self.compute_gp(pv)
        return self.gp.log_likelihood(self.fluxes - model[self.mask])

    def predict_baseline(self, pv):
        self.compute_gp(pv)
        residuals = self.fluxes - squeeze(self.lpf.transit_model(pv))[self.mask]
        bl = zeros_like(self.lpf.timea)
        bl[self.mask] = self.gp.predict(residuals, self.times, return_cov=False)
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