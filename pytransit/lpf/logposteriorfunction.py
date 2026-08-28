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
import astropy.io.fits as pf

from contextlib import contextmanager
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from time import strftime
from typing import Union, Iterable
from warnings import warn

from astropy.table import Table
from scipy.optimize import minimize
from numpy import ndarray, atleast_2d, inf, isfinite, where, clip, diag, full, arange, repeat, tile
from numpy.random import multivariate_normal
from emcee import EnsembleSampler
from matplotlib.pyplot import subplots, setp
from tqdm.auto import tqdm

from pytransit.utils.de import DiffEvol
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP


def _init_pool_worker():
    """Initialise a worker process in a pool created by PyTransit.

    Restricts Numba to a single thread per worker process. Without this, every worker
    would spawn its own set of Numba threads, oversubscribing the machine badly (an
    n-core pool would end up running n x n threads).
    """
    try:
        from numba import set_num_threads
        set_num_threads(1)
    except (ImportError, ValueError):
        pass


def _check_parallelisation(pool, ncores, vectorize) -> None:
    """Raise an error if a pool is requested together with a vectorised posterior.
    """
    if vectorize and (pool is not None or (ncores is not None and ncores > 1)):
        raise ValueError("Parallelisation using a multiprocessing pool cannot be combined with a vectorised log "
                         "posterior function because both DiffEvol and emcee bypass the pool when 'vectorize=True'. "
                         "Either set 'vectorize=False' to parallelise the posterior evaluation over the population "
                         "using the pool, or drop the 'pool' and 'ncores' arguments and rely on the vectorised "
                         "posterior.")


@contextmanager
def _resolve_pool(pool=None, ncores: int = None, start_method: str = None):
    """Yield the parallelisation pool to use, creating and closing one if necessary.

    A user-provided pool takes precedence and is never closed here: its lifetime belongs
    to whoever created it. A pool created from ``ncores`` is owned by this context
    manager and is always terminated on exit, also if the run raises or is interrupted.
    """
    if pool is not None:
        if ncores is not None:
            warn("Both 'pool' and 'ncores' were given: using the user-provided pool and ignoring 'ncores'.")
        yield pool
    elif ncores is not None and ncores > 1:
        if start_method is None:
            methods = get_all_start_methods()
            start_method = 'forkserver' if 'forkserver' in methods else 'spawn'
        pool = get_context(start_method).Pool(ncores, initializer=_init_pool_worker)
        try:
            yield pool
        finally:
            pool.terminate()
            pool.join()
    else:
        yield None


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

    def __getstate__(self):
        """Return the picklable state of the log posterior function.

        The DE optimiser and the MCMC sampler are excluded from the pickled state. Both
        hold a reference to the parallelisation pool while running, and pool objects
        cannot be pickled, which would make the whole log posterior function unpicklable
        (and thus unusable with a multiprocessing pool) as soon as either exists. The
        workers need only to evaluate the posterior, so neither is of any use to them.
        """
        state = self.__dict__.copy()
        for key in ('de', 'sampler', '_old_de_population', '_old_de_fitness'):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.de = None
        self.sampler = None
        self._old_de_population = None
        self._old_de_fitness = None

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

    def optimize_global(self, niter=200, npop=50, population=None, pool=None, lnpost=None, vectorize=True,
                        label='Global optimisation', leave=False, plot_convergence: bool = True, use_tqdm: bool = True,
                        plot_parameters: tuple = (0, 2, 3, 4), min_ptp: float = 1e-2, ncores: int = None,
                        start_method: str = None):
        """Optimise the log posterior function globally using Differential Evolution.

        Parameters
        ----------
        pool
            A parallelisation pool providing a `map` method (`multiprocessing.Pool`,
            `schwimmbad.MPIPool`, etc.). The pool is used only for the duration of this
            call and is not closed: its lifetime belongs to the caller. Requires
            `vectorize=False`.
        ncores
            Number of processes in a pool created for the duration of this call and
            closed afterwards. Ignored if `pool` is given. Requires `vectorize=False`.
        start_method
            Multiprocessing start method used for a pool created from `ncores`. Defaults
            to 'forkserver' if available and 'spawn' otherwise. Forking is avoided by
            default because it is unsafe to fork a process that has already run
            multithreaded Numba code or initialised an OpenCL context. Note that these
            start methods import the calling script in the worker processes, so a script
            using `ncores` must guard its main code with `if __name__ == '__main__':`.
        vectorize
            If True (default), the whole population is passed to the log posterior
            function in a single call and the parallelisation is left to Numba. This is
            usually the fastest option on a single machine, but it is incompatible with
            `pool` and `ncores`.
        """
        lnpost = lnpost or self.lnposterior
        if self.de is None:
            _check_parallelisation(pool, ncores, vectorize)
            self.de = DiffEvol(lnpost, clip(self.ps.bounds, -1, 1), npop, maximize=True, vectorize=vectorize,
                               min_ptp=min_ptp)
            if population is None:
                self.de._population[:, :] = self.create_pv_population(npop)
            else:
                self.de._population[:, :] = population
        else:
            _check_parallelisation(pool, ncores, self.de.vectorize)

        # The pool is attached to the optimiser only for the duration of the run. Storing
        # it permanently would leave the optimiser holding a reference to a pool that may
        # already have been closed by the caller, and would make the log posterior
        # function unpicklable.
        with _resolve_pool(pool, ncores, start_method) as run_pool:
            self.de.pool = run_pool
            try:
                for _ in tqdm(self.de(niter), total=niter, desc=label, leave=leave, disable=(not use_tqdm)):
                    pass
            finally:
                self.de.pool = None

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
                    label='MCMC sampling', reset=True, leave=True, save=False, use_tqdm: bool = True, pool=None,
                    lnpost=None, vectorize: bool = True, ncores: int = None, start_method: str = None):
        """Sample the log posterior function using emcee.

        Parameters
        ----------
        pool
            A parallelisation pool providing a `map` method (`multiprocessing.Pool`,
            `schwimmbad.MPIPool`, etc.). The pool is used only for the duration of this
            call and is not closed: its lifetime belongs to the caller. Requires
            `vectorize=False`.
        ncores
            Number of processes in a pool created for the duration of this call and
            closed afterwards. Ignored if `pool` is given. Requires `vectorize=False`.
        start_method
            Multiprocessing start method used for a pool created from `ncores`. Defaults
            to 'forkserver' if available and 'spawn' otherwise. Forking is avoided by
            default because it is unsafe to fork a process that has already run
            multithreaded Numba code or initialised an OpenCL context. Note that these
            start methods import the calling script in the worker processes, so a script
            using `ncores` must guard its main code with `if __name__ == '__main__':`.
        vectorize
            If True (default), the whole ensemble is passed to the log posterior function
            in a single call and the parallelisation is left to Numba. This is usually the
            fastest option on a single machine, but it is incompatible with `pool` and
            `ncores`.
        """
        if save and self.result_dir is None:
            raise ValueError('The MCMC sampler is set to save the results, but the result directory is not set.')

        lnpost = lnpost or self.lnposterior
        if population is not None:
            pop0 = population
        else:
            if self.sampler is None:
                if hasattr(self, '_local_minimization') and self._local_minimization is not None:
                    pop0 = multivariate_normal(self._local_minimization.x, diag(full(len(self.ps), 0.001 ** 2)), size=npop)
                elif self.de is not None:
                    pop0 = self.de.population.copy()
                else:
                    raise ValueError('Sample MCMC needs an initial population.')
            else:
                pop0 = self.sampler.chain[:, -1, :].copy()

        if self.sampler is None:
            _check_parallelisation(pool, ncores, vectorize)
            self.sampler = EnsembleSampler(pop0.shape[0], pop0.shape[1], lnpost, vectorize=vectorize)
        else:
            _check_parallelisation(pool, ncores, self.sampler.vectorize)

        # The pool is attached to the sampler only for the duration of the run. Storing it
        # permanently would leave the sampler holding a reference to a pool that may
        # already have been closed by the caller, and would make the log posterior
        # function unpicklable.
        with _resolve_pool(pool, ncores, start_method) as run_pool:
            self.sampler.pool = run_pool
            try:
                for i in tqdm(range(repeats), desc=label, disable=(not use_tqdm), leave=leave):
                    if (self.sampler is not None and reset) or i > 0:
                        self.sampler.reset()
                    for _ in tqdm(self.sampler.sample(pop0, iterations=niter, thin=thin,
                                                      skip_initial_state_check=False),
                                  total=niter, desc='Run {:d}/{:d}'.format(i + 1, repeats), leave=False,
                                  disable=(not use_tqdm)):
                        pass
                    if save:
                        self.save(self.result_dir)
                    pop0 = self.sampler.chain[:, -1, :].copy()
            finally:
                self.sampler.pool = None

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

        try:
            if self.sampler is not None:
                fname = save_path / f'{self.name}.fits'
                chains = self.sampler.chain
                nchains = chains.shape[0]
                nsteps = chains.shape[1]
                idch = repeat(arange(nchains), nsteps)
                idst = tile(arange(nsteps), nchains)
                flc = chains.reshape([-1, chains.shape[2]])
                tb1 = Table([idch, idst], names=['chain', 'step'])
                tb1.add_columns(flc.T, names=self.ps.names)
                tb2 = Table([idch, idst], names=['chain', 'step'])
                tb2.add_column(self.sampler.lnprobability.ravel(), name='lnp')
                tbhdu1 = pf.BinTableHDU(tb1, name='posterior')
                tbhdu2 = pf.BinTableHDU(tb2, name='sample_stats')
                hdul = pf.HDUList([pf.PrimaryHDU(), tbhdu1, tbhdu2])
                hdul.writeto(fname, overwrite=True)
        except ValueError:
            print('Could not save the samples in fits format.')

    def __repr__(self):
        return f"Target: {self.name}\nLPF: {self._lpf_name}"
