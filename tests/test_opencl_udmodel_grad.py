"""Gradient tests for the OpenCL uniform-disk transit model backend.

The Numba gradient kernel is the reference: it is the one verified against
central finite differences, and it carries the epoch chain-rule term for the
period derivative that the JAX backend omits. These tests check the OpenCL
gradient both against Numba and, independently, against finite differences of
the OpenCL *forward* model, so a shared misconception in both analytic kernels
would still be caught.
"""

import numpy as np
import pytest
from math import radians, pi

pyopencl = pytest.importorskip("pyopencl")

# `get_platforms` raises rather than returning an empty list when no ICD is
# installed, so the guard has to catch as well as check.
try:
    _HAS_OPENCL_DEVICE = bool(pyopencl.get_platforms())
except pyopencl.Error:
    _HAS_OPENCL_DEVICE = False

if not _HAS_OPENCL_DEVICE:
    pytest.skip("No OpenCL platform available", allow_module_level=True)

from numba import njit
from pytransit.backends.numba.udmodel_grad import udmodel_grad as nb_udmodel_grad

udmodel_grad_nb = njit(nb_udmodel_grad)

T0 = 0.0
P = 2.5
A = 8.0
I = radians(87.0)
E_TEST = 0.1
W = pi / 2
K = 0.1
EPS = 1e-7
TIMES = np.linspace(-0.15, 0.15, 300)


def _make_2d_params(k, t0=T0, p=P, a=A, inc=I, e=E_TEST, w=W, npv=1, npb=1, ntc=1, nor=1):
    return dict(k=np.full((npv, npb), k), t0=np.full((npv, ntc), t0),
                p=np.full((npv, nor), p), a=np.full((npv, nor), a),
                i=np.full((npv, nor), inc), e=np.full((npv, nor), e),
                w=np.full((npv, nor), w))


def _simple_data(npt):
    return dict(lcids=np.zeros(npt, dtype=np.int32), pbids=np.zeros(1, dtype=np.int32),
                epids=np.zeros(1, dtype=np.int32), nsamples=np.ones(1, dtype=np.int32),
                exptimes=np.zeros(1, dtype=np.float64))


def _call_numba_grad(times, params, data, npb=1, nor=1):
    return udmodel_grad_nb(times, params['k'], params['t0'], params['p'], params['a'],
                           params['i'], params['e'], params['w'], data['lcids'],
                           data['pbids'], data['epids'], data['nsamples'],
                           data['exptimes'], npb, nor)


def _call_cl(times, params, data, npb=1, nor=1, return_grad=True, precision='double'):
    from pytransit.backends.opencl.udmodel import UDModelCL

    model = UDModelCL(return_grad=return_grad, precision=precision)
    model.set_data(times, **data)
    return model(times, params['k'], params['t0'], params['p'], params['a'],
                 params['i'], params['e'], params['w'], data['lcids'],
                 data['pbids'], data['epids'], data['nsamples'],
                 data['exptimes'], npb, nor)


def _finite_diff_grad(times, name, eps, **base):
    """Central difference of the OpenCL forward model w.r.t. one scalar parameter."""
    data = _simple_data(times.size)

    def forward(value):
        params = _make_2d_params(**{**base, name: value})
        return _call_cl(times, params, data, return_grad=False)[0]

    return (forward(base[name] + eps) - forward(base[name] - eps)) / (2 * eps)


class TestGradientMatchesNumba:
    def test_simple_transit(self):
        params, data = _make_2d_params(K), _simple_data(TIMES.size)
        flux_nb, dflux_nb = _call_numba_grad(TIMES, params, data)
        flux_cl, dflux_cl = _call_cl(TIMES, params, data)

        assert flux_cl.shape == (1, TIMES.size)
        assert dflux_cl.shape == (1, TIMES.size, 7)
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)


class TestGradientFiniteDiff:
    """Analytic gradient against central finite differences of the OpenCL forward model."""

    BASE = dict(k=K, t0=T0, p=P, a=A, inc=I, e=E_TEST, w=W)

    @pytest.mark.parametrize("name,idx", [('k', 0), ('t0', 1), ('p', 2), ('a', 3),
                                          ('inc', 4), ('e', 5), ('w', 6)])
    def test_gradient(self, name, idx):
        params, data = _make_2d_params(K), _simple_data(TIMES.size)
        flux, dflux = _call_cl(TIMES, params, data)

        fd = _finite_diff_grad(TIMES, name, EPS, **self.BASE)
        # The deficit convention makes in-transit points negative. Contact points
        # carry finite-difference truncation noise, not gradient error, so compare
        # only where the model is well inside the transit.
        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(dflux[0, mask, idx], fd[mask],
                                   rtol=1e-3, atol=1e-8, err_msg=f"{name} gradient")

    @pytest.mark.parametrize("epoch", [0, 1, 3])
    def test_period_gradient_at_nonzero_epoch(self, epoch):
        """Catches an omitted `epoch * dflux[1]` chain-rule term, which is zero at epoch 0."""
        times = TIMES + epoch * P
        params, data = _make_2d_params(K), _simple_data(times.size)
        flux, dflux = _call_cl(times, params, data)

        fd = _finite_diff_grad(times, 'p', EPS, **self.BASE)
        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(dflux[0, mask, 2], fd[mask], rtol=1e-3, atol=1e-8,
                                   err_msg=f"period gradient at epoch {epoch}")


def _hetero_data():
    """Three light curves: two passbands, three epochs, mixed supersampling."""
    times = np.concatenate([np.linspace(-0.15, 0.15, 400) + n * P for n in (0, 1, 3)])
    return times, dict(lcids=np.repeat([0, 1, 2], 400).astype(np.int32),
                       pbids=np.array([0, 1, 0], dtype=np.int32),
                       epids=np.array([0, 1, 2], dtype=np.int32),
                       nsamples=np.array([1, 10, 5], dtype=np.int32),
                       exptimes=np.array([0.0, 0.02, 0.01]))


def _hetero_params(npv, nor):
    """Per-passband radius ratios and genuinely per-epoch transit centres and orbits."""
    pv = np.arange(npv)[:, None] * 1e-3
    ep = np.arange(nor)[None, :] * 1e-2
    return dict(k=np.array([[0.10, 0.11]]) + pv,
                t0=np.array([[0.0, P, 3 * P]]) + pv,
                p=P + pv + 1e-4 * ep,
                a=A + pv + ep,
                i=I + pv - 1e-3 * ep,
                e=0.1 + pv + ep,
                w=W + pv + ep)


class TestGradientMatchesNumbaHeterogeneous:
    """Multiple parameter vectors, passbands, epochs and supersampling rates at once."""

    @pytest.mark.parametrize("npv", [1, 5])
    @pytest.mark.parametrize("nor", [1, 3])
    def test_matches_numba(self, npv, nor):
        times, data = _hetero_data()
        params = _hetero_params(npv, nor)

        flux_nb, dflux_nb = _call_numba_grad(times, params, data, npb=2, nor=nor)
        flux_cl, dflux_cl = _call_cl(times, params, data, npb=2, nor=nor)

        assert flux_cl.shape == (npv, times.size)
        assert dflux_cl.shape == (npv, times.size, 7)
        assert (flux_nb < -1e-6).any(), "test setup should produce in-transit points"
        assert np.abs(dflux_nb[..., 2]).max() > 1e-6, "period gradient should be nonzero"
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)
