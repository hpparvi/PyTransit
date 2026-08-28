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

"""Tests for the parallelisation pool handling in LogPosteriorFunction."""

import pickle
from builtins import map as serial_map
from multiprocessing import Pool

import pytest
from numpy import atleast_2d, linspace, ones

from pytransit import BaseLPF, RoadRunnerModel
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
from pytransit.param import GParameter, ParameterSet, UniformPrior

NPAR = 4


class GaussianLPF(LogPosteriorFunction):
    """A minimal log posterior function with an analytic Gaussian likelihood."""

    def __init__(self, name: str = 'gaussian'):
        super().__init__(name)
        self._init_parameters()

    def _init_parameters(self):
        self.ps = ParameterSet()
        self.ps.add_global_block('x', [GParameter(f'x_{i}', f'parameter {i}', '', UniformPrior(-1.0, 1.0), (-1.0, 1.0))
                                       for i in range(NPAR)])
        self.ps.freeze()

    def lnlikelihood(self, pv):
        pv2 = atleast_2d(pv)
        ll = -0.5 * (pv2 ** 2).sum(1)
        return ll[0] if pv2.shape[0] == 1 and pv.ndim == 1 else ll


def failing_lnpost(pv):
    raise RuntimeError('posterior failure')


class SerialPool:
    """A pool-like object that maps serially and counts how often it is used."""

    def __init__(self):
        self.ncalls = 0

    def map(self, f, x):
        self.ncalls += 1
        return list(serial_map(f, x))


class TestVectorizationCheck:
    """A pool combined with a vectorised posterior must fail loudly, not silently."""

    def test_optimize_global_raises_for_pool_and_vectorize(self):
        lpf = GaussianLPF()
        with pytest.raises(ValueError, match='vectoris'):
            lpf.optimize_global(niter=1, npop=10, pool=SerialPool(), vectorize=True, plot_convergence=False)

    def test_optimize_global_raises_for_ncores_and_vectorize(self):
        lpf = GaussianLPF()
        with pytest.raises(ValueError, match='vectoris'):
            lpf.optimize_global(niter=1, npop=10, ncores=2, vectorize=True, plot_convergence=False)

    def test_sample_mcmc_raises_for_pool_and_vectorize(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, vectorize=True, plot_convergence=False, use_tqdm=False)
        with pytest.raises(ValueError, match='vectoris'):
            lpf.sample_mcmc(niter=2, pool=SerialPool(), vectorize=True, use_tqdm=False)

    def test_sample_mcmc_raises_for_ncores_and_vectorize(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, vectorize=True, plot_convergence=False, use_tqdm=False)
        with pytest.raises(ValueError, match='vectoris'):
            lpf.sample_mcmc(niter=2, ncores=2, vectorize=True, use_tqdm=False)

    def test_check_uses_the_existing_optimiser_vectorization(self):
        """A repeated call must be checked against how the optimiser was actually created."""
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, vectorize=True, plot_convergence=False, use_tqdm=False)
        with pytest.raises(ValueError, match='vectoris'):
            lpf.optimize_global(niter=2, npop=10, pool=SerialPool(), vectorize=False, plot_convergence=False)


class TestPoolUsage:
    def test_de_uses_the_pool_when_not_vectorized(self):
        lpf = GaussianLPF()
        pool = SerialPool()
        lpf.optimize_global(niter=3, npop=10, pool=pool, vectorize=False, plot_convergence=False, use_tqdm=False)
        assert pool.ncalls > 0

    def test_mcmc_uses_the_pool_when_not_vectorized(self):
        lpf = GaussianLPF()
        pool = SerialPool()
        lpf.optimize_global(niter=2, npop=10, vectorize=False, plot_convergence=False, use_tqdm=False)
        lpf.sample_mcmc(niter=3, thin=1, pool=pool, vectorize=False, use_tqdm=False)
        assert pool.ncalls > 0


class TestPoolLifetime:
    """The pool must not outlive the call it was given to."""

    def test_optimiser_releases_the_pool(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, pool=SerialPool(), vectorize=False, plot_convergence=False,
                            use_tqdm=False)
        assert lpf.de.pool is None
        assert lpf.de.map is serial_map

    def test_sampler_releases_the_pool(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, vectorize=False, plot_convergence=False, use_tqdm=False)
        lpf.sample_mcmc(niter=3, thin=1, pool=SerialPool(), vectorize=False, use_tqdm=False)
        assert lpf.sampler.pool is None

    def test_optimiser_releases_the_pool_on_failure(self):
        """An interrupted or failed run must not leave a stale pool behind."""
        lpf = GaussianLPF()
        with pytest.raises(RuntimeError):
            lpf.optimize_global(niter=2, npop=10, pool=SerialPool(), lnpost=failing_lnpost, vectorize=False,
                                plot_convergence=False, use_tqdm=False)
        assert lpf.de.pool is None

    def test_run_after_a_closed_pool_works(self):
        """A pool closed by the caller must not break the following run."""
        lpf = GaussianLPF()
        with Pool(2) as pool:
            lpf.optimize_global(niter=2, npop=10, pool=pool, vectorize=False, plot_convergence=False, use_tqdm=False)
        lpf.optimize_global(niter=2, npop=10, vectorize=False, plot_convergence=False, use_tqdm=False)
        assert lpf.de.pool is None

    def test_ncores_is_ignored_with_a_warning_if_a_pool_is_given(self):
        lpf = GaussianLPF()
        pool = SerialPool()
        with pytest.warns(UserWarning, match='ncores'):
            lpf.optimize_global(niter=2, npop=10, pool=pool, ncores=2, vectorize=False, plot_convergence=False,
                                use_tqdm=False)
        assert pool.ncalls > 0


class TestPicklability:
    """The workers receive the posterior as a bound method, so the LPF must be picklable."""

    def test_lpf_is_picklable_while_holding_a_pool(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, vectorize=False, plot_convergence=False, use_tqdm=False)
        lpf.de.pool = Pool(1)
        try:
            lpf2 = pickle.loads(pickle.dumps(lpf))
            assert lpf2.de is None and lpf2.sampler is None
        finally:
            lpf.de.pool.terminate()
            lpf.de.pool = None

    def test_parameter_set_survives_pickling(self):
        """ParameterSet is a list subclass, so its unpickling path needs `frozen` to exist."""
        lpf = GaussianLPF()
        ps = pickle.loads(pickle.dumps(lpf.ps))
        assert ps.frozen
        assert ps.names == lpf.ps.names

    def test_transit_lpf_posterior_is_picklable(self):
        """A real transit LPF must survive pickling after an optimisation run."""
        time = linspace(0.9, 1.1, 60)
        lpf = BaseLPF('pickletest', ['g'], times=[time], fluxes=[ones(60)], tm=RoadRunnerModel('quadratic'))
        lpf.de = object()          # stand-ins for the unpicklable optimiser and sampler state
        lpf.sampler = object()
        lnpost = pickle.loads(pickle.dumps(lpf.lnposterior))
        assert lnpost(lpf.ps.sample_from_prior(1)[0]) is not None


class TestInternalPool:
    """End-to-end runs with a pool created and closed by PyTransit itself."""

    def test_optimize_global_with_ncores(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, ncores=2, vectorize=False, plot_convergence=False, use_tqdm=False)
        assert lpf.de.pool is None

    def test_sample_mcmc_with_ncores(self):
        lpf = GaussianLPF()
        lpf.optimize_global(niter=2, npop=10, vectorize=False, plot_convergence=False, use_tqdm=False)
        lpf.sample_mcmc(niter=3, thin=1, ncores=2, vectorize=False, use_tqdm=False)
        assert lpf.sampler.pool is None
        assert lpf.sampler.chain.shape[0] == 10
