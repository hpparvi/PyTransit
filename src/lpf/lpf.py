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

import math as mt

import pandas as pd
from astropy.stats import sigma_clip
from matplotlib.pyplot import subplots
from numba import njit, prange
from numpy import (inf, sqrt, ones, zeros_like, concatenate, diff, log, ones_like, all,
                   clip, argsort, any, s_, zeros, arccos, nan, full, pi, sum, repeat, asarray, ndarray, log10,
                   array, atleast_2d, isscalar, atleast_1d, where, isfinite)
from numpy.random import uniform, normal
from scipy.stats import norm
from tqdm.auto import tqdm
from emcee import EnsembleSampler

try:
    from ldtk import LDPSetCreator
    with_ldtk = True
except ImportError:
    with_ldtk = False

from ..models.transitmodel import TransitModel
from .. import QuadraticModel
from ..orbits.orbits_py import duration_eccentric, as_from_rhop, i_from_ba
from ..param.parameter import ParameterSet, PParameter, GParameter, LParameter
from ..param.parameter import UniformPrior as U, NormalPrior as N, GammaPrior as GM
from ..contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from ..utils.de import DiffEvol


@njit(cache=False)
def lnlike_normal(o, m, e):
    return -sum(log(e)) -0.5*p.size*log(2.*pi) - 0.5*sum((o-m)**2/e**2)


@njit("f8(f8[:], f8[:], f8)", cache=False)
def lnlike_normal_s(o, m, e):
    return -o.size*log(e) -0.5*o.size*log(2.*pi) - 0.5*sum((o-m)**2)/e**2


@njit(parallel=True, cache=False, fastmath=True)
def lnlike_normal_v(o, m, e, lcids):
    m = atleast_2d(m)
    npv = m.shape[0]
    npt = o.size
    lnl = zeros(npv)
    for i in prange(npv):
        for j in range(npt):
            k = lcids[j]
            lnl[i] += -log(e[i,k]) - 0.5*log(2*pi) - 0.5*((o[j]-m[i,j])/e[i,k])**2
    return lnl

@njit("f8[:](f8[:], f8[:])", fastmath=False, cache=False)
def unpack_orbit(pv, zpv):
    zpv[:2] = pv[:2]
    zpv[2] = as_from_rhop(pv[2], pv[1])
    if zpv[2] <= 1.:
        zpv[2] = nan
    else:
        zpv[3] = arccos(pv[3] / zpv[2])
    return zpv


@njit(fastmath=True)
def map_pv(pv):
    pv = atleast_2d(pv)
    pvt = zeros((pv.shape[0], 7))
    pvt[:,0]   = sqrt(pv[:,4])
    pvt[:,1:3] = pv[:,0:2]
    pvt[:,  3] = as_from_rhop(pv[:,2], pv[:,1])
    pvt[:,  4] = i_from_ba(pv[:,3], pvt[:,3])
    return pvt


@njit(fastmath=True, cache=False)
def map_ldc(ldc):
    ldc = atleast_2d(ldc)
    uv = zeros_like(ldc)
    a, b = sqrt(ldc[:,0::2]), 2.*ldc[:,1::2]
    uv[:,0::2] = a * b
    uv[:,1::2] = a * (1. - b)
    return uv

class BaseLPF:
    _lpf_name = 'base'

    def __init__(self, name: str, passbands: list, times: list = None, fluxes: list = None, errors: list = None,
                 pbids: list = None, covariates: list = None, tm: TransitModel = None,
                 nsamples: tuple = 1, exptimes: tuple = 0.):
        self.tm = tm or QuadraticModel(klims=(0.01, 0.75), nk=512, nz=512)

        # LPF name
        # --------
        self.name = name

        # Passbands
        # ---------
        # Should be arranged from blue to red
        if isinstance(passbands, (list, tuple, ndarray)):
            self.passbands = passbands
        else:
            self.passbands = [passbands]
        self.npb = npb = len(self.passbands)

        self.nsamples = None
        self.exptimes = None

        # Declare high-level objects
        # --------------------------
        self.ps = None          # Parametrisation
        self.de = None          # Differential evolution optimiser
        self.sampler = None     # MCMC sampler
        self.instrument = None  # Instrument
        self.ldsc = None        # Limb darkening set creator
        self.ldps = None        # Limb darkening profile set
        self.cntm = None        # Contamination model

        # Declare data arrays and variables
        # ---------------------------------
        self.nlc: int = 0                # Number of light curves
        self.times: list = None          # List of time arrays
        self.fluxes: list = None         # List of flux arrays
        self.errors: list = None         # List of flux uncertainties
        self.covariates: list = None     # List of covariates
        self.wn: ndarray = None          # Array of white noise estimates for each light curve
        self.timea: ndarray = None       # Array of concatenated times
        self.mfluxa: ndarray = None      # Array of concatenated model fluxes
        self.ofluxa: ndarray = None      # Array of concatenated observed fluxes
        self.errora: ndarray = None      # Array of concatenated model fluxes

        self.lcids: ndarray = None       # Array of light curve indices for each datapoint
        self.pbids: ndarray = None       # Array of passband indices for each light curve
        self.lcslices: list = None       # List of light curve slices

        # Set up the observation data
        # ---------------------------
        if times is not None and fluxes is not None and pbids is not None:
            self._init_data(times, fluxes, pbids, covariates, errors, nsamples, exptimes)

        # Setup parametrisation
        # =====================
        self._init_parameters()

        # Initialise the additional lnprior list
        # --------------------------------------
        self.lnpriors = []

        # Initialise the temporary arrays
        # -------------------------------
        self._zpv = zeros(6)
        self._tuv = zeros((npb, 2))
        self._zeros = zeros(npb)
        self._ones = ones(npb)

        if times is not None:
            self._bad_fluxes = [ones_like(t) for t in self.times]
        else:
            self._bad_fluxes = None


    def _init_data(self, times, fluxes, pbids, covariates=None, errors=None, nsamples=1, exptimes=0.):

        if isinstance(times, ndarray) and times.dtype == float:
            times = [times]

        if isinstance(fluxes, ndarray) and fluxes.dtype == float:
            fluxes = [fluxes]

        self.nlc = len(times)
        self.times = asarray(times)
        self.fluxes = asarray(fluxes)
        self.pbids = asarray(pbids)
        self.wn = [diff(f).std() / sqrt(2) for f in fluxes]
        self.timea = concatenate(self.times)
        self.ofluxa = concatenate(self.fluxes)
        self.mfluxa = zeros_like(self.ofluxa)
        self.pbids = atleast_1d(pbids).astype('int')
        self.lcids = concatenate([full(t.size, i) for i, t in enumerate(self.times)])

        if isscalar(nsamples):
            self.nsamples = full(self.nlc, nsamples)
            self.exptimes = full(self.nlc, exptimes)
        else:
            assert (len(nsamples) == self.nlc) and (len(exptimes) == self.nlc)
            self.nsamples = asarray(nsamples, 'int')
            self.exptimes = asarray(exptimes)

        self.tm.set_data(self.timea, self.lcids, self.pbids, self.nsamples, self.exptimes)

        if errors is None:
            self.errors = array([full(t.size, nan) for t in self.times])
        else:
            self.errors = asarray(errors)
        self.errora = concatenate(self.errors)

        # Initialise the light curves slices
        # ----------------------------------
        self.lcslices = []
        sstart = 0
        for i in range(self.nlc):
            s = self.times[i].size
            self.lcslices.append(s_[sstart:sstart + s])
            sstart += s

        # Initialise the covariate arrays, if given
        # -----------------------------------------
        if covariates is not None:
            self.covariates = covariates
            for cv in self.covariates:
                cv[:, 1:] = (cv[:, 1:] - cv[:, 1:].mean(0)) / cv[:, 1:].ptp(0)
            self.ncovs = self.covariates[0].shape[1]
            self.covsize = array([c.size for c in self.covariates])
            self.covstart = concatenate([[0], self.covsize.cumsum()[:-1]])
            self.cova = concatenate(self.covariates)

    def _init_parameters(self):
        self.ps = ParameterSet()
        self._init_p_orbit()
        self._init_p_planet()
        self._init_p_limb_darkening()
        self._init_p_baseline()
        self._init_p_noise()
        self.ps.freeze()

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('tc',  'zero_epoch',       'd',      N(0.0,  0.1), (-inf, inf)),
            GParameter('pr',  'period',           'd',      N(1.0, 1e-5), (0,    inf)),
            GParameter('rho', 'stellar_density',  'g/cm^3', U(0.1, 25.0), (0,    inf)),
            GParameter('b',   'impact_parameter', 'R_s',    U(0.0,  1.0), (0,      1))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        """Planet parameter initialisation.
        """
        pk2 = [PParameter('k2', 'area_ratio', 'A_s', GM(0.1), (0.01**2, 0.55**2))]
        self.ps.add_passband_block('k2', 1, 1, pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice

    def _init_p_limb_darkening(self):
        """Limb darkening parameter initialisation.
        """
        pld = concatenate([
            [PParameter('q1_{:d}'.format(i), 'q1_coefficient', '', U(0, 1), bounds=(0, 1)),
             PParameter('q2_{:d}'.format(i), 'q2_coefficient', '', U(0, 1), bounds=(0, 1))]
            for i in range(self.npb)])
        self.ps.add_passband_block('ldc', 2, self.npb, pld)
        self._sl_ld = self.ps.blocks[-1].slice
        self._start_ld = self.ps.blocks[-1].start

    def _init_p_baseline(self):
        """Baseline parameter initialisation.
        """
        pass

    def _init_p_noise(self):
        """Noise parameter initialisation.
        """
        pns = [LParameter('lne_{:d}'.format(i), 'log_error_{:d}'.format(i), '', U(-8, -0), bounds=(-8, -0)) for i in range(self.nlc)]
        self.ps.add_lightcurve_block('log_err', 1, self.nlc, pns)
        self._sl_err = self.ps.blocks[-1].slice
        self._start_err = self.ps.blocks[-1].start

    def create_pv_population(self, npop=50):
        pvp = self.ps.sample_from_prior(npop)
        for sl in self.ps.blocks[1].slices:
            pvp[:,sl] = uniform(0.01**2, 0.25**2, size=(npop, 1))

        # With LDTk
        # ---------
        #
        # Use LDTk to create the sample if LDTk has been initialised.
        #
        if self.ldps:
            istart = self._start_ld
            cms, ces = self.ldps.coeffs_tq()
            for i, (cm, ce) in enumerate(zip(cms.flat, ces.flat)):
                pvp[:, i + istart] = normal(cm, ce, size=pvp.shape[0])

        # No LDTk
        # -------
        #
        # Ensure that the total limb darkening decreases towards
        # red passbands.
        #
        else:
            ldsl = self._sl_ld
            for i in range(pvp.shape[0]):
                pid = argsort(pvp[i, ldsl][::2])[::-1]
                pvp[i, ldsl][::2] = pvp[i, ldsl][::2][pid]
                pvp[i, ldsl][1::2] = pvp[i, ldsl][1::2][pid]

        # Estimate white noise from the data
        # ----------------------------------
        for i in range(self.nlc):
            wn = diff(self.ofluxa).std() / sqrt(2)
            pvp[:, self._start_err] = log10(uniform(0.5*wn, 2*wn, size=npop))
        return pvp

    def baseline(self, pv):
        """Multiplicative baseline"""
        return 1.

    def trends(self, pv):
        """Additive trends"""
        return 0.

    def transit_model(self, pv):
        pv = atleast_2d(pv)
        pvp = map_pv(pv)
        ldc = map_ldc(pv[:,self._sl_ld])
        flux = self.tm.evaluate_pv(pvp, ldc)
        return flux

    def flux_model(self, pv):
        baseline    = self.baseline(pv)
        trends      = self.trends(pv)
        model_flux = self.transit_model(pv)
        return baseline * model_flux + trends

    def residuals(self, pv):
        return self.ofluxa - self.flux_model(pv)

    def set_prior(self, pid: int, prior) -> None:
            self.ps[pid].prior = prior

    def add_t14_prior(self, mean: float, std: float) -> None:
        """Add a normal prior on the transit duration.

        Parameters
        ----------
        mean
        std

        Returns
        -------

        """
        def T14(pv):
            a = as_from_rhop(pv[2], pv[1])
            t14 = duration_eccentric(pv[1], sqrt(pv[4]), a, mt.acos(pv[3] / a), 0, 0, 1)
            return norm.logpdf(t14, mean, std)
        self.lnpriors.append(T14)

    def add_as_prior(self, mean: float, std: float) -> None:
        """Add a prior on the scaled semi-major axis

        Parameters
        ----------
        mean
        std

        Returns
        -------

        """
        def as_prior(pv):
            a = as_from_rhop(pv[2], pv[1])
            return norm.logpdf(a, mean, std)
        self.lnpriors.append(as_prior)

    def add_ldtk_prior(self, teff: tuple, logg: tuple, z: tuple,
                       uncertainty_multiplier: float = 3,
                       pbs: tuple = ('g', 'r', 'i', 'z')) -> None:
        """Add a LDTk-based prior on the limb darkening.

        Parameters
        ----------
        teff
        logg
        z
        uncertainty_multiplier
        pbs

        Returns
        -------

        """
        fs = {n: f for n, f in zip('g r i z'.split(), (sdss_g, sdss_r, sdss_i, sdss_z))}
        filters = [fs[k] for k in pbs]
        self.ldsc = LDPSetCreator(teff, logg, z, filters)
        self.ldps = self.ldsc.create_profiles(1000)
        self.ldps.resample_linear_z()
        self.ldps.set_uncertainty_multiplier(uncertainty_multiplier)
        def ldprior(pv):
            return self.ldps.lnlike_tq(pv[self._sl_ld])
        self.lnpriors.append(ldprior)

    def remove_outliers(self, sigma=5):
        fmodel = self.flux_model(self.de.minimum_location)
        times, fluxes, pbids, errors = [], [], [], []
        for i in range(len(self.times)):
            res = self.fluxes[i] - fmodel[i]
            mask = ~sigma_clip(res, sigma=sigma).mask
            times.append(self.times[i][mask])
            fluxes.append(self.fluxes[i][mask])
            if self.errors is not None:
                errors.append(self.errors[i][mask])
        self._init_data(times, fluxes, self.pbids, (errors if self.errors is not None else None))

    def remove_transits(self, tids):
        m = ones(len(self.times), bool)
        m[tids] = False
        self._init_data(self.times[m], self.fluxes[m], self.pbids[m],
                        self.covariates[m] if self.covariates is not None else None,
                        self.errors[m], self.nsamples[m], self.exptimes[m])
        self._init_parameters()

    def lnprior(self, pv):
        return self.ps.lnprior(pv) + self.additional_priors(pv)

    def additional_priors(self, pv):
        """Additional priors."""
        return sum([f(pv) for f in self.lnpriors])

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        wn = 10**(atleast_2d(pv)[:,self._sl_err])
        return lnlike_normal_v(self.ofluxa, flux_m, wn, self.lcids)

    def lnposterior(self, pv):
        lnp = self.lnprior(pv) + self.lnlikelihood(pv)
        return where(isfinite(lnp), lnp, -inf)

    def __call__(self, pv):
        return self.lnposterior(pv)

    def optimize_global(self, niter=200, npop=50, population=None, label='Global optimisation', leave=False):
        if self.de is None:
            self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True, vectorize=True)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:,:] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label, leave=leave):
            pass

    def sample_mcmc(self, niter=500, thin=5, label='MCMC sampling', reset=False, leave=True):
        if self.sampler is None:
            self.sampler = EnsembleSampler(self.de.n_pop, self.de.n_par, self.lnposterior, vectorize=True)
            pop0 = self.de.population
        else:
            pop0 = self.sampler.chain[:,-1,:].copy()
        if reset:
            self.sampler.reset()
        for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter, desc=label, leave=False):
            pass

    def posterior_samples(self, burn: int=0, thin: int=1, include_ldc: bool=False):
        ldstart = self._sl_ld.start
        fc = self.sampler.chain[:, burn::thin, :].reshape([-1, self.de.n_par])
        d = fc if include_ldc else fc[:, :ldstart]
        n = self.ps.names if include_ldc else self.ps.names[:ldstart]
        return pd.DataFrame(d, columns=n)

    def plot_mcmc_chains(self, pid: int=0, alpha: float=0.1, thin: int=1, ax=None):
        fig, ax = (None, ax) if ax is not None else subplots()
        ax.plot(self.sampler.chain[:, ::thin, pid].T, 'k', alpha=alpha)
        fig.tight_layout()
        return fig


    def __repr__(self):
        s  = f"""Target: {self.name}
  LPF: {self._lpf_name}
  Passbands: {self.passbands}"""
        return s
