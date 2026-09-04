"""Tests for the JAX backend of `WNLogLikelihood`.

This path is reachable only via `UniformDiskModel(backend='jax', return_grad=True)` --
`RoadRunnerModel` raises `NotImplementedError` for `backend='jax', return_grad=True`, and the
Numba backend raises inside `WNLogLikelihood.__init__` (device-side reduction needs OpenCL).
It was untested before this file, and this branch changed `argnums` from 8 to 9 when the
`baseline` parameter was added (`pytransit/loglikelihoods/wnloglikelihood.py`), so a wrong
gradient column count or a swapped baseline/jitter order would previously have gone unnoticed.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from pytransit import UniformDiskModel
from pytransit.loglikelihoods import WNLogLikelihood

T0, P, A, I, E, W = 0.0, 2.5, 8.0, np.radians(87.0), 0.0, 0.5 * np.pi
K = 0.1
TIMES = np.linspace(-0.15, 0.15, 300)


def _model(return_grad):
    m = UniformDiskModel(backend='jax', return_grad=return_grad)
    m.set_data(TIMES)
    return m


def _pars1d():
    """Parameters shaped (1,), as `get_callable`'s JAX model expects (npb=ntc=nor=1)."""
    return (jnp.array([K]), jnp.array([T0]), jnp.array([P]), jnp.array([A]),
           jnp.array([I]), jnp.array([E]), jnp.array([W]))


def _dataset(seed=0, jitter=4e-4, baseline=0.9997):
    rng = np.random.default_rng(seed)
    err = 1e-3 * np.ones(TIMES.size)
    mv = _model(return_grad=False)
    dflux = np.asarray(mv.evaluate(K, T0, P, A, I, E, W))    # flux deficit dF = F - 1
    obs = baseline + dflux + err * rng.normal(size=TIMES.size)
    return obs, err, jitter, baseline, dflux


class TestJaxWNLogLikelihood:

    def test_forward_matches_evaluate(self):
        """Without return_grad, the JAX loglike must use the same forward model as evaluate()."""
        obs, err, jitter, baseline, dflux = _dataset()
        mval = _model(return_grad=False)
        ll = WNLogLikelihood(mval, obs, err)
        lnl = ll(*_pars1d(), baseline, jitter)

        r = (obs - baseline) - dflux
        e2 = err ** 2 + jitter ** 2
        expected = (-0.5 * obs.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(e2))
                   - 0.5 * np.sum(r ** 2 / e2))
        np.testing.assert_allclose(float(lnl), expected, rtol=1e-12)

    def test_gradient_columns_and_baseline_derivative(self):
        """9 gradient entries (k, t0, p, a, i, e, w, baseline, jitter); grad[7] (baseline) must
        match both the analytic sum(r/e2) and a central finite difference of lnL."""
        obs, err, jitter, baseline, dflux = _dataset()
        mgrad = _model(return_grad=True)
        ll = WNLogLikelihood(mgrad, obs, err)

        lnl, grad = ll(*_pars1d(), baseline, jitter)
        assert len(grad) == 9

        r = (obs - baseline) - dflux
        e2 = err ** 2 + jitter ** 2
        analytic_dbaseline = np.sum(r / e2)
        np.testing.assert_allclose(float(grad[7]), analytic_dbaseline, rtol=1e-10)

        # Central finite difference of lnL with respect to baseline, using the
        # ungraded forward callable so it is independent of the code path under test.
        mval = _model(return_grad=False)
        ll_val = WNLogLikelihood(mval, obs, err)
        eps = 1e-6
        lnl_plus = ll_val(*_pars1d(), baseline + eps, jitter)
        lnl_minus = ll_val(*_pars1d(), baseline - eps, jitter)
        fd_dbaseline = float((lnl_plus - lnl_minus) / (2 * eps))
        np.testing.assert_allclose(float(grad[7]), fd_dbaseline, rtol=1e-5)

        # grad[8] is the trailing jitter derivative: jitter * (sum(r^2/e2^2) - sum(1/e2)).
        analytic_djitter = jitter * (np.sum(r ** 2 / e2 ** 2) - np.sum(1.0 / e2))
        np.testing.assert_allclose(float(grad[8]), analytic_djitter, rtol=1e-8)

        fd_lnl_plus = ll_val(*_pars1d(), baseline, jitter + eps)
        fd_lnl_minus = ll_val(*_pars1d(), baseline, jitter - eps)
        fd_djitter = float((fd_lnl_plus - fd_lnl_minus) / (2 * eps))
        np.testing.assert_allclose(float(grad[8]), fd_djitter, rtol=1e-4)
