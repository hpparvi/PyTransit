"""Tests for the OpenCL RoadRunner transit model backend.

The Numba backend is the reference: it is the finite-difference-tested one and
carries the period chain-rule term. Every numerical check here compares against
it directly.

Note the convention that differs from the uniform-disk backend: RoadRunner has
a real per-parameter-vector validity guard.
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
from pytransit.backends.numba.rrmodel import rrmodel as nb_rrmodel
from pytransit.models.rrmodel import RoadRunnerModel

rrmodel_nb = njit(nb_rrmodel)

T0 = 0.0
P = 3.0
A = 6.0
I = radians(88.0)
E = 0.0
W = pi / 2
# Wider than the transit window (~+/-0.09 d) so out-of-window points,
# where the model must return exactly 0.0 with a zero gradient, are covered.
TIMES = np.linspace(-0.13, 0.13, 300)


def _integration(ldmodel='quadratic'):
    """A configured RoadRunnerModel, used purely as a source of the integration tables."""
    return RoadRunnerModel(backend='numba', ldmodel=ldmodel)


def _ld_inputs(model, ldc, npv, return_grad=False):
    """Build ldp/ldg/ldi/dldi exactly as RoadRunnerModel.evaluate does."""
    from pytransit.backends.numba.limb_darkening import (evaluate_ld, evaluate_ldg,
                                                         evaluate_ldi, evaluate_distar)
    ldc = np.atleast_1d(ldc)
    ldp = evaluate_ld(model.ldmodel, model.mu, ldc)
    ldi = evaluate_ldi(model.ldmmean, ldc)
    ldg = evaluate_ldg(model.ldgrad, model.mu, ldc) if return_grad else None
    dldi = evaluate_distar(model.ldigmean, ldc) if return_grad else None
    return [x if (x is None or x.shape[0] >= npv) else np.repeat(x, npv, axis=0)
            for x in (ldp, ldg, ldi, dldi)]


def _make_2d(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npv=1, npb=1, ntc=1, nor=1):
    return dict(k=np.full((npv, npb), k), t0=np.full((npv, ntc), t0),
                p=np.full((npv, nor), p), a=np.full((npv, nor), a),
                i=np.full((npv, nor), inc), e=np.full((npv, nor), e),
                w=np.full((npv, nor), w))


def _simple_data(npt):
    return dict(lcids=np.zeros(npt, dtype=np.int32), pbids=np.zeros(1, dtype=np.int32),
                epids=np.zeros(1, dtype=np.int32), nsamples=np.ones(1, dtype=np.int32),
                exptimes=np.zeros(1, dtype=np.float64))


def _kernel_args(model, params, data, ld, npb, nor):
    ldp, ldg, ldi, dldi = ld
    return (params['k'], params['t0'], params['p'], params['a'], params['i'],
            params['e'], params['w'], data['lcids'], data['pbids'], data['epids'],
            data['nsamples'], data['exptimes'], ldp, ldg, ldi, dldi,
            model._rules, model._pt0, model._pdt, model.ng, npb, nor)


class TestForwardMatchesNumba:
    def test_simple_transit(self):
        from pytransit.backends.opencl.rrmodel import RRModelCL

        model = _integration()
        params, data = _make_2d(0.1), _simple_data(TIMES.size)
        ld = _ld_inputs(model, np.array([0.3, 0.2]), npv=1)
        args = _kernel_args(model, params, data, ld, 1, 1)

        expected = rrmodel_nb(TIMES, *args)
        assert (expected < -1e-6).any(), "test setup should produce in-transit points"

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(TIMES, **data)
        flux = cl_model(TIMES, *args)

        assert flux.shape == (1, TIMES.size)
        np.testing.assert_allclose(flux, expected, atol=1e-12)


BJD = 2458325.5


def _hetero_data(origin=BJD, n=400):
    """Three light curves: two passbands, three epochs, mixed supersampling."""
    times = np.concatenate([np.linspace(-0.08, 0.08, n) + origin + m * P for m in (0, 1, 3)])
    return times, dict(lcids=np.repeat([0, 1, 2], n).astype(np.int32),
                       pbids=np.array([0, 1, 0], dtype=np.int32),
                       epids=np.array([0, 1, 2], dtype=np.int32),
                       nsamples=np.array([1, 10, 5], dtype=np.int32),
                       exptimes=np.array([0.0, 0.02, 0.01]))


def _hetero_params(npv, nor, origin=BJD, npb=2):
    """Per-passband radius ratios and genuinely per-epoch transit centres and orbits.

    The orbital parameters must actually differ between epochs, otherwise the
    ``nor > 1`` indexing is unobservable and a comparison against Numba passes
    even if every point reads epoch 0.
    """
    pv = np.arange(npv)[:, None] * 1e-3
    ep = np.arange(nor)[None, :] * 1e-2
    return dict(k=np.linspace(0.10, 0.12, npb)[None, :] + pv,
                t0=np.array([[origin, origin + P, origin + 3 * P]]) + pv,
                p=P + pv + 1e-4 * ep, a=A + pv + ep, i=I + pv - 1e-3 * ep,
                e=0.1 + pv + ep, w=W + pv + ep)


class TestForwardMatchesNumbaHeterogeneous:
    @pytest.mark.parametrize("npv", [1, 5])
    @pytest.mark.parametrize("nor", [1, 3])
    def test_matches_numba(self, npv, nor):
        from pytransit.backends.opencl.rrmodel import RRModelCL

        times, data = _hetero_data()
        params = _hetero_params(npv, nor)
        model = _integration()
        ld = _ld_inputs(model, np.array([[0.30, 0.20], [0.55, 0.10]]), npv=npv)
        args = _kernel_args(model, params, data, ld, 2, nor)

        expected = rrmodel_nb(times, *args)
        assert (expected < -1e-6).any(), "test setup should produce in-transit points"

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(times, **data)
        flux = cl_model(times, *args)

        assert flux.shape == (npv, times.size)
        np.testing.assert_allclose(flux, expected, atol=1e-12)


class TestLimbDarkeningModels:
    """The device code is limb-darkening-agnostic: it only sees a tabulated mean.

    `nonlinear` also exercises the lazy `-DNLDC` program rebuild with nldc=4.
    """

    @pytest.mark.parametrize("ldmodel,ldc", [
        ('quadratic', np.array([0.3, 0.2])),
        ('power-2',   np.array([0.5, 0.7])),
        ('nonlinear', np.array([0.4, 0.1, 0.2, -0.1])),
        ('linear',    np.array([0.4])),
    ])
    def test_matches_numba(self, ldmodel, ldc):
        from pytransit.backends.opencl.rrmodel import RRModelCL

        model = _integration(ldmodel)
        params, data = _make_2d(0.1), _simple_data(TIMES.size)
        ld = _ld_inputs(model, ldc, npv=1)
        args = _kernel_args(model, params, data, ld, 1, 1)

        expected = rrmodel_nb(TIMES, *args)
        assert (expected < -1e-6).any(), f"{ldmodel} setup should transit"

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(TIMES, **data)
        np.testing.assert_allclose(cl_model(TIMES, *args), expected, atol=1e-12)


class TestMorePassbandsThanCoefficients:
    """npb != nldc, the configuration the `ldc` reshape bug used to corrupt."""

    def test_matches_numba(self):
        from pytransit.backends.opencl.rrmodel import RRModelCL

        n = 200
        times = np.concatenate([np.linspace(-0.08, 0.08, n)] * 3)
        data = dict(lcids=np.repeat([0, 1, 2], n).astype(np.int32),
                    pbids=np.array([0, 1, 2], dtype=np.int32),
                    epids=np.zeros(3, dtype=np.int32),
                    nsamples=np.ones(3, dtype=np.int32),
                    exptimes=np.zeros(3))
        params = _make_2d(0.1, npb=3)
        params['k'] = np.array([[0.10, 0.11, 0.12]])

        model = _integration()
        ld = _ld_inputs(model, np.array([[0.3, 0.2], [0.4, 0.2], [0.5, 0.1]]), npv=1)
        args = _kernel_args(model, params, data, ld, 3, 1)

        expected = rrmodel_nb(times, *args)
        # Before the `ldc` axis fix this configuration returned ~-8e10.
        assert np.all(expected > -0.1), \
            f"reference flux should be physical once the ldc axis is right, got {expected.min()}"

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(times, **data)
        np.testing.assert_allclose(cl_model(times, *args), expected, atol=1e-12)


class TestInvalidParameterVectors:
    """The per-parameter-vector validity guard, which the uniform disk lacks."""

    @pytest.mark.parametrize("name,override", [
        ('a_below_one', dict(a=0.5)),
        ('a_nan',       dict(a=np.nan)),
        ('e_negative',  dict(e=-0.1)),
    ])
    def test_invalid_vector_is_nan(self, name, override):
        from pytransit.backends.opencl.rrmodel import RRModelCL

        params, data = _make_2d(0.1, npv=2), _simple_data(TIMES.size)
        for key, value in override.items():
            params[key][1, :] = value          # only the second vector is invalid

        model = _integration()
        ld = _ld_inputs(model, np.array([0.3, 0.2]), npv=2)
        args = _kernel_args(model, params, data, ld, 1, 1)
        expected = rrmodel_nb(TIMES, *args)

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(TIMES, **data)
        flux = cl_model(TIMES, *args)

        assert np.isnan(expected[1]).all(), "reference should mark the vector invalid"
        assert np.isnan(flux[1]).all(), f"{name}: invalid vector should be all NaN"
        np.testing.assert_allclose(flux[0], expected[0], atol=1e-12)



class TestExtremeRadiusRatios:
    """Very large and very small radius ratios, where the quadrature regimes are unusual.

    A k of 0.6 keeps the planet crossing the limb over most of the transit, and a k of
    0.002 makes the on-disk table segment tiny.
    """

    @pytest.mark.parametrize("k", [0.6, 0.002])
    def test_forward_matches_numba(self, k):
        from pytransit.backends.opencl.rrmodel import RRModelCL

        model = _integration()
        params, data = _make_2d(k), _simple_data(TIMES.size)
        ld = _ld_inputs(model, np.array([0.3, 0.2]), npv=1)
        args = _kernel_args(model, params, data, ld, 1, 1)
        expected = rrmodel_nb(TIMES, *args)

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(TIMES, **data)
        np.testing.assert_allclose(cl_model(TIMES, *args), expected, atol=1e-12)

    def test_mixed_passbands_with_very_different_radius_ratios(self):
        """A moderate and a very large radius ratio in the same evaluation."""
        from pytransit.backends.opencl.rrmodel import RRModelCL

        n = 200
        times = np.concatenate([np.linspace(-0.13, 0.13, n)] * 2)
        data = dict(lcids=np.repeat([0, 1], n).astype(np.int32),
                    pbids=np.array([0, 1], dtype=np.int32),
                    epids=np.zeros(2, dtype=np.int32),
                    nsamples=np.ones(2, dtype=np.int32), exptimes=np.zeros(2))
        params = _make_2d(0.1, npb=2)
        params['k'] = np.array([[0.10, 0.60]])

        model = _integration()
        ld = _ld_inputs(model, np.array([[0.3, 0.2], [0.4, 0.1]]), npv=1)
        args = _kernel_args(model, params, data, ld, 2, 1)
        expected = rrmodel_nb(times, *args)

        cl_model = RRModelCL(return_grad=False)
        cl_model.set_data(times, **data)
        np.testing.assert_allclose(cl_model(times, *args), expected, atol=1e-12)
