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

import seaborn as sb
import pandas as pd
import xarray as xa

from pathlib import Path
from time import strftime
from typing import Union, Iterable

from scipy.optimize import minimize
from numpy import ndarray, atleast_2d, inf, isfinite, where, clip, diag, full
from numpy.random import multivariate_normal
from emcee import EnsembleSampler
from matplotlib.pyplot import subplots, setp
from tqdm.auto import tqdm

from pytransit.utils.de import DiffEvol
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP


class LogPosteriorFunction:
    _lpf_name = 'LogPosteriorFunction'

    def __init__(self, name: str, result_dir: Union[Path, str] = '.'):
        """The Log Posterior Function class.

        Parameters
        ----------
        name: str
            Name of the log posterior function instance.
        """
        self.name = name
        self.result_dir = Path(result_dir if result_dir is not None else '.')

        # Declare high-level objects
        # --------------------------
        self.ps = None  # Parametrisation
        self.de = None  # Differential evolution optimiser
        self.sampler = None  # MCMC sampler
        self._local_minimization = None

        # Initialise the additional lnprior list
        # --------------------------------------
        self._additional_log_priors = []

        self._old_de_fitness = None
        self._old_de_population = None

    def print_parameters(self, columns: int = 2):
        columns = max(1, columns)
        for i, p in enumerate(self.ps):
            print(p.__repr__(), end=('\n' if i % columns == columns - 1 else '\t'))

    def _init_parameters(self):
        self.ps = ParameterSet()
        self.ps.freeze()

    def create_pv_population(self, npop=50):
        return self.ps.sample_from_prior(npop)

    def set_prior(self, parameter, prior, *nargs) -> None:
        if isinstance(parameter, str):
            descriptions = self.ps.descriptions
            names = self.ps.names
            if parameter in descriptions:
                parameter = descriptions.index(parameter)
            elif parameter in names:
                parameter = names.index(parameter)
            else:
                params = ', '.join([f"{ln} ({sn})" for ln, sn in zip(self.ps.descriptions, self.ps.names)])
                raise ValueError(f'Parameter "{parameter}" not found from the parameter set: {params}')

        if isinstance(prior, str):
            if prior.lower() in ['n', 'np', 'normal']:
                prior = NP(nargs[0], nargs[1])
            elif prior.lower() in ['u', 'up', 'uniform']:
                prior = UP(nargs[0], nargs[1])
            else:
                raise ValueError(f'Unknown prior "{prior}". Allowed values are (N)ormal and (U)niform.')

        self.ps[parameter].prior = prior

    def lnprior(self, pv: ndarray) -> Union[Iterable, float]:
        """Log prior density for a 1D or 2D array of model parameters.

        Parameters
        ----------
        pv: ndarray
            Either a 1D parameter vector or a 2D parameter array.

        Returns
        -------
            Log prior density for the given parameter vector(s).
        """
        return self.ps.lnprior(pv) + self.additional_priors(pv)

    def additional_priors(self, pv):
        pv = atleast_2d(pv)
        return sum([f(pv) for f in self._additional_log_priors], 0)

    def lnlikelihood(self, pv):
        raise NotImplementedError

    def lnposterior(self, pv):
        lnp = self.lnprior(pv) + self.lnlikelihood(pv)
        return where(isfinite(lnp), lnp, -inf)

    def __call__(self, pv):
        return self.lnposterior(pv)

    def optimize_local(self, pv0=None, method='powell'):
        if pv0 is None:
            if self.de is not None:
                pv0 = self.de.minimum_location
            else:
                pv0 = self.ps.mean_pv
        res = minimize(lambda pv: -self.lnposterior(pv), pv0, method=method)
        self._local_minimization = res

    def optimize_global(self, niter=200, npop=50, population=None, label='Global optimisation', leave=False,
                        plot_convergence: bool = True, use_tqdm: bool = True, plot_parameters: tuple = (0, 2, 3, 4)):

        if self.de is None:
            self.de = DiffEvol(self.lnposterior, clip(self.ps.bounds, -1, 1), npop, maximize=True, vectorize=True)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:, :] = population
        for _ in tqdm(self.de(niter), total=niter, desc=label, leave=leave, disable=(not use_tqdm)):
            pass

        if plot_convergence:
            fig, axs = subplots(1, 1 + len(plot_parameters), figsize=(13, 2), constrained_layout=True)
            rfit = self.de._fitness
            mfit = isfinite(rfit)

            if self._old_de_fitness is not None:
                m = isfinite(self._old_de_fitness)
                axs[0].hist(-self._old_de_fitness[m], facecolor='midnightblue', bins=25, alpha=0.25)
            axs[0].hist(-rfit[mfit], facecolor='midnightblue', bins=25)

            for i, ax in zip(plot_parameters, axs[1:]):
                if self._old_de_fitness is not None:
                    m = isfinite(self._old_de_fitness)
                    ax.plot(self._old_de_population[m, i], -self._old_de_fitness[m], 'kx', alpha=0.25)
                ax.plot(self.de.population[mfit, i], -rfit[mfit], 'k.')
                ax.set_xlabel(self.ps.descriptions[i])
            setp(axs, yticks=[])
            setp(axs[1], ylabel='Log posterior')
            setp(axs[0], xlabel='Log posterior')
            sb.despine(fig, offset=5)
        self._old_de_population = self.de.population.copy()
        self._old_de_fitness = self.de._fitness.copy()

    def sample_mcmc(self, niter: int = 500, thin: int = 5, repeats: int = 1, npop: int = None, population=None,
                    label='MCMC sampling', reset=True, leave=True, save=False, use_tqdm: bool = True):

        if save and self.result_dir is None:
            raise ValueError('The MCMC sampler is set to save the results, but the result directory is not set.')

        if self.sampler is None:
            if population is not None:
                pop0 = population
            elif hasattr(self, '_local_minimization') and self._local_minimization is not None:
                pop0 = multivariate_normal(self._local_minimization.x, diag(full(len(self.ps), 0.001 ** 2)), size=npop)
            elif self.de is not None:
                pop0 = self.de.population.copy()
            else:
                raise ValueError('Sample MCMC needs an initial population.')
            self.sampler = EnsembleSampler(pop0.shape[0], pop0.shape[1], self.lnposterior, vectorize=True)
        else:
            pop0 = self.sampler.chain[:, -1, :].copy()

        for i in tqdm(range(repeats), desc=label, disable=(not use_tqdm), leave=leave):
            if reset or i > 0:
                self.sampler.reset()
            for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin), total=niter,
                          desc='Run {:d}/{:d}'.format(i + 1, repeats), leave=False, disable=(not use_tqdm)):
                pass
            if save:
                self.save(self.result_dir)
            pop0 = self.sampler.chain[:, -1, :].copy()

    def posterior_samples(self, burn: int = 0, thin: int = 1):
        fc = self.sampler.chain[:, burn::thin, :].reshape([-1, len(self.ps)])
        df = pd.DataFrame(fc, columns=self.ps.names)
        return df

    def plot_mcmc_chains(self, pid: int = 0, alpha: float = 0.1, thin: int = 1, ax=None):
        fig, ax = (None, ax) if ax is not None else subplots()
        ax.plot(self.sampler.chain[:, ::thin, pid].T, 'k', alpha=alpha)
        fig.tight_layout()
        return fig

    def save(self, save_path: Path = '.'):
        save_path = Path(save_path)
        npar = len(self.ps)

        if self.de:
            de = xa.DataArray(self.de.population, dims='pvector parameter'.split(), coords={'parameter': self.ps.names})
        else:
            de = None

        if self.sampler is not None:
            mc = xa.DataArray(self.sampler.chain, dims='pvector step parameter'.split(),
                              coords={'parameter': self.ps.names}, attrs={'ndim': npar, 'npop': self.sampler.nwalkers})
        else:
            mc = None

        ds = xa.Dataset(data_vars={'de_population': de, 'mcmc_samples': mc},
                        attrs={'created': strftime('%Y-%m-%d %H:%M:%S'), 'name': self.name})
        ds.to_netcdf(save_path.joinpath(f'{self.name}.nc'))

    def __repr__(self):
        return f"Target: {self.name}\nLPF: {self._lpf_name}"
