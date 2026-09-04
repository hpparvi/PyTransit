"""Gradient tests for the OpenCL RoadRunner transit model backend.

The Numba gradient kernel is the reference. These tests check the OpenCL
gradient both against it and, independently, against central finite differences
of the OpenCL *forward* model, so a shared misconception in both analytic
kernels would still be caught.

RoadRunner returns the flux deficit, so the in-transit mask is ``flux < -1e-6``.
"""

import numpy as np
import pytest

pyopencl = pytest.importorskip("pyopencl")

try:
    _HAS_OPENCL_DEVICE = bool(pyopencl.get_platforms())
except pyopencl.Error:
    _HAS_OPENCL_DEVICE = False

if not _HAS_OPENCL_DEVICE:
    pytest.skip("No OpenCL platform available", allow_module_level=True)

from numba import njit
from pytransit.backends.numba.rrmodel_grad import rrmodel_grad as nb_rrmodel_grad
from tests.test_opencl_rrmodel import (_integration, _ld_inputs, _kernel_args,
                                       _make_2d, _simple_data, P, A, I, W)

rrmodel_grad_nb = njit(nb_rrmodel_grad)

K = 0.1
LDC = np.array([0.3, 0.2])
EPS = 1e-6
# Wider than the transit window so out-of-window points, where the gradient
# row must be exactly zero, are covered.
TIMES = np.linspace(-0.13, 0.13, 300)


def _call(times, params, data, ldc=LDC, npb=1, nor=1, return_grad=True,
          ldmodel='quadratic', precision='double'):
    from pytransit.backends.opencl.rrmodel import RRModelCL

    model = _integration(ldmodel)
    ld = _ld_inputs(model, ldc, npv=params['k'].shape[0], return_grad=return_grad)
    args = _kernel_args(model, params, data, ld, npb, nor)

    cl_model = RRModelCL(return_grad=return_grad, precision=precision)
    cl_model.set_data(times, **data)
    return cl_model(times, *args)


def _call_numba(times, params, data, ldc=LDC, npb=1, nor=1, ldmodel='quadratic'):
    model = _integration(ldmodel)
    ld = _ld_inputs(model, ldc, npv=params['k'].shape[0], return_grad=True)
    return rrmodel_grad_nb(times, *_kernel_args(model, params, data, ld, npb, nor))


def _params(base):
    """The orbital half of a base parameter dict, as the 2D kernel arrays."""
    return _make_2d(**{key: value for key, value in base.items() if key != 'ldc'})


def _forward(times, base, **overrides):
    """OpenCL forward flux for the base parameter set with some values replaced."""
    kwargs = {**base, **overrides}
    ldc = kwargs.pop('ldc')
    return _call(times, _make_2d(**kwargs), _simple_data(times.size),
                 ldc=ldc, return_grad=False)[0]


def _fd(times, name, eps, base):
    """Central difference of the OpenCL forward model w.r.t. one scalar parameter."""
    return (_forward(times, base, **{name: base[name] + eps})
            - _forward(times, base, **{name: base[name] - eps})) / (2 * eps)


class TestGradientMatchesNumba:
    def test_simple_transit(self):
        params, data = _make_2d(K), _simple_data(TIMES.size)
        flux_nb, dflux_nb = _call_numba(TIMES, params, data)
        flux_cl, dflux_cl = _call(TIMES, params, data)

        assert flux_cl.shape == (1, TIMES.size)
        assert dflux_cl.shape == (1, TIMES.size, 7 + 2)   # 7 orbital + nldc=2
        assert (flux_nb < -1e-6).any(), "test setup should produce in-transit points"
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)


class TestGradientFiniteDiff:
    """Analytic gradient against central differences of the OpenCL forward model."""

    BASE = dict(k=K, t0=0.0, p=P, a=A, inc=I, e=0.1, w=W, ldc=LDC)

    # `k` uses the dldm/dk ~= 0 approximation, so it gets the loose tolerance the
    # Numba tests use for the same reason.
    @pytest.mark.parametrize("name,idx,rtol,atol", [
        ('k', 0, 0.1, 1e-5), ('t0', 1, 2e-3, 1e-6), ('p', 2, 2e-3, 1e-6),
        ('a', 3, 2e-3, 1e-6), ('inc', 4, 2e-3, 1e-6), ('e', 5, 2e-3, 1e-6),
        ('w', 6, 2e-3, 1e-6),
    ])
    def test_orbital_gradients(self, name, idx, rtol, atol):
        params, data = _params(self.BASE), _simple_data(TIMES.size)
        flux, dflux = _call(TIMES, params, data)
        fd = _fd(TIMES, name, EPS, self.BASE)

        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(dflux[0, mask, idx], fd[mask], rtol=rtol, atol=atol,
                                   err_msg=f"{name} gradient")

    @pytest.mark.parametrize("j", [0, 1])
    def test_ld_coefficient_gradients(self, j):
        params, data = _params(self.BASE), _simple_data(TIMES.size)
        flux, dflux = _call(TIMES, params, data)

        lp, lm = LDC.copy(), LDC.copy()
        lp[j] += EPS
        lm[j] -= EPS
        fd = (_forward(TIMES, self.BASE, ldc=lp)
              - _forward(TIMES, self.BASE, ldc=lm)) / (2 * EPS)

        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(dflux[0, mask, 7 + j], fd[mask], rtol=1e-3, atol=1e-7,
                                   err_msg=f"ld coefficient {j}")

    @pytest.mark.parametrize("epoch", [0, 1, 3])
    def test_period_gradient_at_nonzero_epoch(self, epoch):
        """Catches an omitted epoch chain-rule term, which is zero only at epoch 0."""
        times = TIMES + epoch * P
        params, data = _params(self.BASE), _simple_data(times.size)
        flux, dflux = _call(times, params, data)
        fd = _fd(times, 'p', EPS, self.BASE)

        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(dflux[0, mask, 2], fd[mask], rtol=2e-3, atol=1e-6,
                                   err_msg=f"period gradient at epoch {epoch}")


class TestGradientMatchesNumbaHeterogeneous:
    """Multiple parameter vectors, passbands, epochs and supersampling rates.

    This is the decisive cross-backend test: it exercises the per-passband LD
    gradient blocks, the per-epoch indexing, and the period folding term at once.
    """

    @pytest.mark.parametrize("npv", [1, 5])
    @pytest.mark.parametrize("nor", [1, 3])
    def test_matches_numba(self, npv, nor):
        from tests.test_opencl_rrmodel import _hetero_data, _hetero_params

        times, data = _hetero_data()
        params = _hetero_params(npv, nor)
        ldc = np.array([[0.30, 0.20], [0.55, 0.10]])

        flux_nb, dflux_nb = _call_numba(times, params, data, ldc=ldc, npb=2, nor=nor)
        flux_cl, dflux_cl = _call(times, params, data, ldc=ldc, npb=2, nor=nor)

        assert dflux_cl.shape == (npv, times.size, 7 + 2 * 2)  # 7 + npb*nldc
        assert (flux_nb < -1e-6).any(), "test setup should produce in-transit points"
        assert np.abs(dflux_nb[..., 2]).max() > 1e-6, "period gradient should be nonzero"
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)

    def test_ld_blocks_are_per_passband(self):
        """A point's LD derivative must be zero outside its own passband's block."""
        from tests.test_opencl_rrmodel import _hetero_data, _hetero_params

        times, data = _hetero_data()
        params = _hetero_params(1, 1)
        _, dflux = _call(times, params, data, ldc=np.array([[0.30, 0.20], [0.55, 0.10]]),
                         npb=2, nor=1)

        pb_of_point = data['pbids'][data['lcids']]
        for ipb in range(2):
            other = pb_of_point != ipb
            for j in range(2):
                np.testing.assert_allclose(dflux[0, other, 7 + ipb * 2 + j], 0.0, atol=1e-12,
                                           err_msg=f"passband {ipb} coeff {j} leaked")

    @pytest.mark.parametrize("ldmodel,ldc", [
        ('power-2',   np.array([0.5, 0.7])),
        ('nonlinear', np.array([0.4, 0.1, 0.2, -0.1])),
    ])
    def test_other_limb_darkening_models(self, ldmodel, ldc):
        """`nonlinear` has nldc=4, exercising the lazy -DNLDC rebuild for gradients."""
        params, data = _make_2d(K), _simple_data(TIMES.size)
        flux_nb, dflux_nb = _call_numba(TIMES, params, data, ldc=ldc, ldmodel=ldmodel)
        flux_cl, dflux_cl = _call(TIMES, params, data, ldc=ldc, ldmodel=ldmodel)

        assert dflux_cl.shape == (1, TIMES.size, 7 + ldc.size)
        assert (flux_nb < -1e-6).any()
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)

    def test_invalid_vector_gradient_is_nan(self):
        params, data = _make_2d(K, npv=2), _simple_data(TIMES.size)
        params['a'][1, :] = 0.5
        flux_nb, dflux_nb = _call_numba(TIMES, params, data)
        flux_cl, dflux_cl = _call(TIMES, params, data)

        assert np.isnan(dflux_nb[1]).all(), "reference should mark the vector invalid"
        assert np.isnan(dflux_cl[1]).all(), "invalid vector gradient should be all NaN"
        np.testing.assert_allclose(dflux_cl[0], dflux_nb[0], atol=1e-12)



class TestExtremeRadiusRatioGradient:
    """The gradient tables at very large and very small radius ratios."""

    @pytest.mark.parametrize("k", [0.6, 0.002])
    def test_matches_numba(self, k):
        params, data = _make_2d(k), _simple_data(TIMES.size)
        flux_nb, dflux_nb = _call_numba(TIMES, params, data)
        flux_cl, dflux_cl = _call(TIMES, params, data)
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)


class TestSinglePrecisionGradient:
    """The fp32 gradient path, which the forward-only precision test does not reach."""

    def test_close_to_numba(self):
        params, data = _make_2d(K), _simple_data(TIMES.size)
        flux_nb, dflux_nb = _call_numba(TIMES, params, data)
        flux_cl, dflux_cl = _call(TIMES, params, data, precision='single')

        assert np.asarray(dflux_cl).dtype == np.float32
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-5)
        scale = np.abs(dflux_nb).max()
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-3 * scale)
