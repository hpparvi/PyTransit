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

from numpy import asarray, unique, zeros, inf, squeeze, zeros_like, isfinite, log10, diff, sqrt, arange, log
from scipy.optimize import minimize
from tqdm.auto import tqdm

try:
    from celerite import GP
    from celerite.terms import Matern32Term
    with_celerite = True
except ImportError:
    with_celerite = False

class FrozenMultiCeleriteLogLikelihood:
    def __init__(self, lpf, name: str = 'gp', lcids=None):
        if not with_celerite:
            raise ImportError("CeleriteLogLikelihood requires celerite.")

        self.name = name
        self.lpf = lpf

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising CeleriteLogLikelihood.')
        self.lcids = lcids if lcids is not None else arange(lpf.nlc)
        self.gps = []

        self.learn_gps()

    def learn_gps(self):
        for lcid in tqdm(self.lcids, desc='Learning GPs', leave=False):
            kernel = Matern32Term(log(self.lpf.fluxes[lcid].std()), log(0.1))
            gp = GP(kernel, mean=0.0)
            gp.freeze_parameter('kernel:log_sigma')

            def nll(x):
                gp.set_parameter_vector(x)
                gp.compute(self.lpf.times[lcid], self.lpf.wn[lcid])
                return -gp.log_likelihood(self.lpf.fluxes[lcid] - 1.0)

            res = minimize(nll, [log(1.)])
            gp.set_parameter_vector(res.x)
            gp.compute(self.lpf.times[lcid], self.lpf.wn[lcid])
            self.gps.append(gp)

    def compute_gp_lnlikelihood(self, pv, model):
        lnl = 0.0
        for lcid, gp in zip(self.lcids, self.gps):
            sl = self.lpf.lcslices[lcid]
            lnl += gp.log_likelihood(self.lpf.fluxes[lcid] - model[sl])
        return lnl

    def predict_baseline(self, pv):
        residuals = self.lpf.ofluxa - squeeze(self.lpf.transit_model(pv))
        bl = zeros_like(self.lpf.timea)
        for lcid, gp in zip(self.lcids, self.gps):
            sl = self.lpf.lcslices[lcid]
            bl[sl] = gp.predict(residuals[sl], self.lpf.times[lcid], return_cov=False)
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