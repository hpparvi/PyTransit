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

from matplotlib.pyplot import subplots
from numba import njit
from numpy import (inf, sqrt, ones, zeros_like, concatenate, diff, log, ones_like,
                   clip, argsort, any, s_, zeros, arccos, nan, isnan, full, pi, sum, repeat, asarray, ndarray)
from numpy.random import uniform, normal
from scipy.stats import norm
from tqdm.auto import tqdm

try:
    import pandas as pd
    with_pandas = True
except ImportError:
    with_pandas = False

try:
    from pyde import DiffEvol
    with_pyde = True
except ImportError:
    with_pyde = False

try:
    from emcee import EnsembleSampler
    with_emcee = True
except ImportError:
    with_emcee = False

try:
    from george import GP
    from george.kernels import ExpKernel as EK, ExpSquaredKernel as ESK, ConstantKernel as CK, Matern32Kernel as M32
    with_george = True
except ImportError:
    with_george = False

try:
    from ldtk import LDPSetCreator
    with_ldtk = True
except ImportError:
    with_ldtk = False

from pytransit.supersampler import SuperSampler
from pytransit.transitmodel import TransitModel
from pytransit import MandelAgol as MA
from pytransit.mandelagol_py import eval_quad_ip_mp
from pytransit.orbits_py import z_circular, duration_eccentric
from pytransit.param.parameter import ParameterSet, PParameter, GParameter
from pytransit.param.parameter import UniformPrior as U, NormalPrior as N, GammaPrior as GM
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z
from pytransit.utils.orbits import as_from_rhop


@njit(cache=False)
def lnlike_normal(o, m, e):
    return -sum(log(e)) -0.5*p.size*log(2.*pi) - 0.5*sum((o-m)**2/e**2)


@njit("f8(f8[:], f8[:], f8)", cache=False)
def lnlike_normal_s(o, m, e):
    return -o.size*log(e) -0.5*o.size*log(2.*pi) - 0.5*sum((o-m)**2)/e**2


@njit("f8[:](f8[:], f8[:])", fastmath=False, cache=False)
def unpack_orbit(pv, zpv):
    zpv[:2] = pv[:2]
    zpv[2] = as_from_rhop(pv[2], pv[1])
    if zpv[2] <= 1.:
        zpv[2] = nan
    else:
        zpv[3] = arccos(pv[3] / zpv[2])
    return zpv


@njit("f8[:,:](f8[:], f8[:,:])", fastmath=True, cache=False)
def qq_to_uv(pv, uv):
    a, b = sqrt(pv[::2]), 2.*pv[1::2]
    uv[:,0] = a * b
    uv[:,1] = a * (1. - b)
    return uv


class BaseLPF:
    _lpf_name = 'base'

    def __init__(self, target: str, passbands: list, times: list=None, fluxes:list=None,
                 pbids:list=None, tm:TransitModel=None, nsamples: int=1, exptime: float = 0.020433598):
        self.tm = tm or MA(interpolate=True, klims=(0.01, 0.75), nk=512, nz=512)

        self.target = target            # Name of the planet
        self.passbands = passbands      # Passbands, should be arranged from the bluest to reddest
        self.npb = npb = len(passbands) # Number of passbands

        # Declare high-level objects
        # --------------------------
        self.ps = None          # Parametrisation
        self.de = None          # Differential evolution optimiser
        self.sampler = None     # MCMC sampler
        self.instrument = None  # Instrument
        self.ldsc = None        # Limb darkening set creator
        self.ldps = None        # Limb darkening profile set
        self.cntm = None        # Contamination model

        self.ss = SuperSampler(nsamples=nsamples, exptime=exptime) if nsamples > 1 else None

        # Declare data arrays and variables
        # ---------------------------------
        self.nlc: int = 0                # Number of light curves
        self.times: list = None          # List of time arrays
        self.fluxes: list = None         # List of flux arrays
        self.covariates: list = None     # List of covariates
        self.wn: ndarray = None          # Array of white noise estimates
        self.timea: ndarray = None       # Array of concatenated (and possibly supersampled) times
        self.ofluxa: ndarray = None      # Array of concatenated observed fluxes
        self.mfluxa: ndarray = None      # Array of concatenated model fluxes
        self.pbida: ndarray = None       # Array of passband indices for each datapoint
        self.lcida: ndarray = None       # Array of light curve indices for each datapoint
        self.lcslices: list = None       # List of light curve slices

        self.timea_orig: ndarray = None  # Array of concatenated times
        self.lcida_orig: ndarray = None  # Array of concatenated light curve indices
        self.pbida_orig: ndarray = None  # Array of concatenated passband indices

        # Set up the observation data
        # ---------------------------
        if times and fluxes and pbids:
            self._init_data(times, fluxes, pbids)

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


    def _init_data(self, times, fluxes, pbids):
        self.nlc = len(times)
        self.times = asarray(times)
        self.fluxes = asarray(fluxes)
        self.pbids = asarray(pbids)
        self.wn = [diff(f).std() / sqrt(2) for f in fluxes]
        self.timea = concatenate(self.times)
        self.ofluxa = concatenate(self.fluxes)
        self.mfluxa = zeros_like(self.ofluxa)
        self.pbida = concatenate([full(t.size, pbid) for t, pbid in zip(self.times, self.pbids)])
        self.lcida = concatenate([full(t.size, i) for i, t in enumerate(self.times)])

        self.timea_orig = self.timea
        self.lcida_orig = self.lcida
        self.pbida_orig = self.pbida
        if self.ss:
            self.timea = self.ss.sample(self.timea)
            self.lcida = repeat(self.lcida, self.ss.nsamples)
            self.pbida = repeat(self.pbida, self.ss.nsamples)

        self.lcslices = []
        sstart = 0
        for i in range(self.nlc):
            s = self.times[i].size
            self.lcslices.append(s_[sstart:sstart + s])
            sstart += s


    def _init_parameters(self):
        self.ps = ParameterSet()
        self._init_p_orbit()
        self._init_p_planet()
        self._init_p_limb_darkening()
        self._init_p_baseline()
        self.ps.freeze()

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('tc',  'zero_epoch',       'd',      N(0.0,  1.0), (-inf, inf)),
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
        return pvp

    def baseline(self, pv):
        """Flux baseline (multiplicative)"""
        return ones(self.nlc)

    def trends(self, pv):
        """Systematic trends (additive)"""
        return zeros(self.nlc)

    def _compute_z(self, pv):
        zpv = unpack_orbit(pv, self._zpv)
        if isnan(zpv[2]):
            return None
        else:
            return z_circular(self.timea, zpv)

    def _compute_transit(self, pv, z):
        _k = sqrt(pv[self._pid_k2])
        uv = qq_to_uv(pv[self._sl_ld], self._tuv)
        fluxes = eval_quad_ip_mp(z, self.pbida, _k, uv, self._zeros, self.tm.ed, self.tm.ld, self.tm.le, self.tm.kt,
                                     self.tm.zt)
        fluxes = self.ss.average(fluxes) if self.ss else fluxes
        fluxes = [fluxes[sl] for sl in self.lcslices]
        return fluxes

    def transit_model(self, pv):
        z = self._compute_z(pv)
        if z is not None:
            return self._compute_transit(pv, z)
        else:
            return self._bad_fluxes

    def flux_model(self, pv):
        bls = self.baseline(pv)
        trs = self.trends(pv)
        tms = self.transit_model(pv)
        return [tm*bl + tr for tm,bl,tr in zip(tms, bls, trs)]

    def residuals(self, pv):
        return [fo - fm for fo, fm in zip(self.fluxes, self.flux_model(pv))]

    def add_t14_prior(self, m, s):
        def T14(pv):
            a = as_from_rhop(pv[2], pv[1])
            t14 = duration_eccentric(pv[1], sqrt(pv[4]), a, mt.acos(pv[3] / a), 0, 0, 1)
            return norm.logpdf(t14, m, s)
        self.lnpriors.append(T14)

    def add_as_prior(self, m, s):
        def as_prior(pv):
            a = as_from_rhop(pv[2], pv[1])
            return norm.logpdf(a, m, s)
        self.lnpriors.append(as_prior)

    def add_ldtk_prior(self, teff, logg, z, uncertainty_multiplier=3, pbs=('g', 'r', 'i', 'z')):
        fs = {n: f for n, f in zip('g r i z'.split(), (sdss_g, sdss_r, sdss_i, sdss_z))}
        filters = [fs[k] for k in pbs]
        self.ldsc = LDPSetCreator(teff, logg, z, filters)
        self.ldps = self.ldsc.create_profiles(1000)
        self.ldps.resample_linear_z()
        self.ldps.set_uncertainty_multiplier(uncertainty_multiplier)
        def ldprior(pv):
            return self.ldps.lnlike_tq(pv[self._sl_ld])
        self.lnpriors.append(ldprior)


    def lnprior(self, pv):
        return self.ps.lnprior(pv)

    def lnprior_hooks(self, pv):
        """Additional constraints."""
        return sum([f(pv) for f in self.lnpriors])

    def lnlikelihood(self, pv):
        flux_m = self.flux_model(pv)
        lnlike = 0.0
        for i in range(self.nlc):
            lnlike += lnlike_normal_s(self.fluxes[i], flux_m[i], self.wn[i])
        return lnlike

    def lnposterior(self, pv):
        if any(pv < self.ps.bounds[:, 0]) or any(pv > self.ps.bounds[:, 1]):
            return -inf
        else:
            return self.lnprior(pv) + self.lnlikelihood(pv) + self.lnprior_hooks(pv)

    def __call__(self, pv):
        return self.lnposterior(pv)

    def optimize_global(self, niter=200, npop=50, population=None, label='Global optimisation'):
        if not with_pyde:
            raise ImportError("PyDE not installed.")

        if self.de is None:
            self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:,:] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label):
            pass

    def sample_mcmc(self, niter=500, thin=5, label='MCMC sampling', reset=False):
        if not with_emcee:
            raise ImportError('Emcee not installed.')
        if self.sampler is None:
            self.sampler = EnsembleSampler(self.de.n_pop, self.de.n_par, self.lnposterior)
            pop0 = self.de.population
        else:
            pop0 = self.sampler.chain[:,-1,:].copy()
        if reset:
            self.sampler.reset()
        for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter, desc=label):
            pass

    def posterior_samples(self, burn: int=0, thin: int=1, include_ldc: bool=False):
        ldstart = self._sl_ld.start
        fc = self.sampler.chain[:, burn::thin, :].reshape([-1, self.de.n_par])
        d = fc if include_ldc else fc[:, :ldstart]
        n = self.ps.names if include_ldc else self.ps.names[:ldstart]
        return pd.DataFrame(d, columns=n) if with_pandas else d

    def plot_mcmc_chains(self, pid: int=0, alpha: float=0.1, thin: int=1, ax=None):
        fig, ax = (None, ax) if ax is not None else subplots()
        ax.plot(self.sampler.chain[:, ::thin, pid].T, 'k', alpha=alpha)
        fig.tight_layout()
        return fig


    def __repr__(self):
        s  = f"""Target: {self.target}
  LPF: {self._lpf_name}
  Passbands: {self.passbands}"""
        return s
