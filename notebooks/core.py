import math as mt
import warnings

import seaborn as sb
with warnings.catch_warnings():
    cp = sb.color_palette()

from numpy import log, square, pi, array, inf, full_like, isinf, clip
from numba import njit
from emcee import EnsembleSampler
from tqdm import tqdm
from scipy.constants import G

from pyde import DiffEvol
from pytransit import MandelAgol as MA
from pytransit.param.parameter import (ParameterSet, GParameter, PParameter, LParameter,
                                       NormalPrior as NP,
                                       UniformPrior as UP)

d_h = 24.
d_m = 60 * d_h
d_s = 60 * d_m

@njit("f8(f8[:], f8[:], f8)")
def ll_normal_es(o,m,e):
    """Normal log likelihood for scalar average standard deviation."""
    return -o.size*log(e) -0.5*o.size*log(2*pi) - 0.5*square(o-m).sum()/e**2

@njit
def as_from_rhop(rho, period):
    """Scaled semi-major axis from the stellar density and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]

    Returns
    -------

      as : scaled semi-major axis [R_star]
    """
    return (G/(3.*pi))**(1./3.)*((period*d_s)**2 * 1e3*rho)**(1./3.)


class LPFunction(object):
    """A basic log posterior function class.
    """

    def __init__(self, time, flux, nthreads=1):

        # Set up the transit model
        # ------------------------
        self.tm = MA(interpolate=True, klims=(0.08, 0.13), nthr=nthreads)
        self.nthr = nthreads

        # Initialise data
        # ---------------
        self.time = time.copy() if time is not None else array([])
        self.flux_o = flux.copy() if flux is not None else array([])
        self.npt = self.time.size

        # Set the optimiser and the MCMC sampler
        # --------------------------------------
        self.de = None
        self.sampler = None

        # Set up the parametrisation and priors
        # -------------------------------------
        psystem = [
            GParameter('tc', 'zero_epoch', 'd', NP(1.01, 0.02), (-inf, inf)),
            GParameter('pr', 'period', 'd', NP(2.50, 1e-7), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.90, 2.50), (0.90, 2.5)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.00, 1.00), (0.00, 1.0)),
            GParameter('k2', 'area_ratio', 'A_s', UP(0.08 ** 2, 0.13 ** 2), (1e-8, inf))]

        pld = [
            PParameter('q1', 'q1_coefficient', '', UP(0, 1), bounds=(0, 1)),
            PParameter('q2', 'q2_coefficient', '', UP(0, 1), bounds=(0, 1))]

        pbl = [LParameter('es', 'white_noise', '', UP(1e-6, 1e-2), bounds=(1e-6, 1e-2))]
        per = [LParameter('bl', 'baseline', '', NP(1.00, 0.001), bounds=(0.8, 1.2))]

        self.ps = ParameterSet()
        self.ps.add_global_block('system', psystem)
        self.ps.add_passband_block('ldc', 2, 1, pld)
        self.ps.add_lightcurve_block('baseline', 1, 1, pbl)
        self.ps.add_lightcurve_block('error', 1, 1, per)
        self.ps.freeze()

    def compute_baseline(self, pv):
        """Constant baseline model"""
        return full_like(self.flux_o, pv[8])

    def compute_transit(self, pv):
        """Transit model"""
        _a = as_from_rhop(pv[2], pv[1])  # Scaled semi-major axis from stellar density and orbital period
        _i = mt.acos(pv[3] / _a)  # Inclination from impact parameter and semi-major axis
        _k = mt.sqrt(pv[4])  # Radius ratio from area ratio

        a, b = mt.sqrt(pv[5]), 2 * pv[6]
        _uv = array([a * b, a * (1. - b)])  # Quadratic limb darkening coefficients

        return self.tm.evaluate(self.time, _k, _uv, pv[0], pv[1], _a, _i)

    def compute_lc_model(self, pv):
        """Combined baseline and transit model"""
        return self.compute_baseline(pv) * self.compute_transit(pv)

    def lnprior(self, pv):
        """Log prior"""
        if any(pv < self.ps.lbounds) or any(pv > self.ps.ubounds):
            return -inf
        else:
            return self.ps.lnprior(pv)

    def lnlikelihood(self, pv):
        """Log likelihood"""
        flux_m = self.compute_lc_model(pv)
        return ll_normal_es(self.flux_o, flux_m, pv[7])

    def lnposterior(self, pv):
        """Log posterior"""
        lnprior = self.lnprior(pv)
        if isinf(lnprior):
            return lnprior
        else:
            return lnprior + self.lnlikelihood(pv)

    def create_pv_population(self, npop=50):
        return self.ps.sample_from_prior(npop)

    def optimize(self, niter=200, npop=50, population=None, label='Optimisation'):
        """Global optimisation using Differential evolution"""
        if self.de is None:
            self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:, :] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label):
            pass

    def sample(self, niter=500, thin=5, label='MCMC sampling', reset=False):
        """MCMC sampling using emcee"""
        if self.sampler is None:
            self.sampler = EnsembleSampler(self.de.n_pop, self.de.n_par, self.lnposterior)
            pop0 = self.de.population
        else:
            pop0 = self.sampler.chain[:, -1, :].copy()
        if reset:
            self.sampler.reset()
        for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter, desc=label):
            pass