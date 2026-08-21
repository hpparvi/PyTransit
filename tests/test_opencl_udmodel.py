"""Tests for the OpenCL uniform-disk transit model backend.

The OpenCL backend must reproduce the Numba backend, which is the
finite-difference-tested reference. Every numerical check here is therefore
either a direct comparison against `backends.numba.udmodel` or one of the
analytic properties the Numba tests already assert.
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
from pytransit.backends.numba.udmodel import udmodel as nb_udmodel

udmodel_nb = njit(nb_udmodel)

T0 = 0.0
P = 2.5
A = 8.0
I = radians(87.0)
E = 0.0
W = pi / 2
TIMES = np.linspace(-0.15, 0.15, 300)


def _make_2d_params(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npv=1, npb=1, ntc=1, nor=1):
    return dict(k=np.full((npv, npb), k), t0=np.full((npv, ntc), t0),
                p=np.full((npv, nor), p), a=np.full((npv, nor), a),
                i=np.full((npv, nor), inc), e=np.full((npv, nor), e),
                w=np.full((npv, nor), w))


def _simple_data(npt):
    """Single light curve, single passband, single epoch, no supersampling."""
    return dict(lcids=np.zeros(npt, dtype=np.int32), pbids=np.zeros(1, dtype=np.int32),
                epids=np.zeros(1, dtype=np.int32), nsamples=np.ones(1, dtype=np.int32),
                exptimes=np.zeros(1, dtype=np.float64))


def _call_numba(times, params, data, npb=1, nor=1):
    return udmodel_nb(times, params['k'], params['t0'], params['p'], params['a'],
                      params['i'], params['e'], params['w'], data['lcids'],
                      data['pbids'], data['epids'], data['nsamples'],
                      data['exptimes'], npb, nor)


class TestForwardMatchesNumba:
    def test_simple_transit_matches_numba(self):
        from pytransit.backends.opencl.udmodel import UDModelCL

        params, data = _make_2d_params(0.1), _simple_data(TIMES.size)
        expected = _call_numba(TIMES, params, data)

        model = UDModelCL(return_grad=False)
        model.set_data(TIMES, **data)
        flux = model(TIMES, params['k'], params['t0'], params['p'], params['a'],
                     params['i'], params['e'], params['w'], data['lcids'],
                     data['pbids'], data['epids'], data['nsamples'],
                     data['exptimes'], 1, 1)

        assert flux.shape == (1, TIMES.size)
        np.testing.assert_allclose(flux, expected, atol=1e-12)


def _hetero_data():
    """Three light curves: two passbands, three epochs, mixed supersampling."""
    times = np.concatenate([np.linspace(-0.15, 0.15, 400) + n * P for n in (0, 1, 3)])
    return times, dict(lcids=np.repeat([0, 1, 2], 400).astype(np.int32),
                       pbids=np.array([0, 1, 0], dtype=np.int32),
                       epids=np.array([0, 1, 2], dtype=np.int32),
                       nsamples=np.array([1, 10, 5], dtype=np.int32),
                       exptimes=np.array([0.0, 0.02, 0.01]))


def _hetero_params(npv, nor):
    """Per-passband radius ratios and per-epoch transit centres and orbits.

    The orbital parameters must genuinely differ between epochs, otherwise the
    `nor > 1` indexing in the kernel is unobservable and the comparison against
    Numba passes even if every point reads epoch 0.
    """
    pv = np.arange(npv)[:, None] * 1e-3      # varies over parameter vectors
    ep = np.arange(nor)[None, :] * 1e-2      # varies over epochs
    return dict(k=np.array([[0.10, 0.11]]) + pv,
                t0=np.array([[0.0, P, 3 * P]]) + pv,
                p=P + pv + 1e-4 * ep,
                a=A + pv + ep,
                i=I + pv - 1e-3 * ep,
                e=0.1 + pv + ep,
                w=W + pv + ep)


class TestForwardMatchesNumbaHeterogeneous:
    @pytest.mark.parametrize("npv", [1, 5])
    @pytest.mark.parametrize("nor", [1, 3])
    def test_matches_numba(self, npv, nor):
        from pytransit.backends.opencl.udmodel import UDModelCL

        times, data = _hetero_data()
        params = _hetero_params(npv, nor)
        expected = _call_numba(times, params, data, npb=2, nor=nor)

        model = UDModelCL(return_grad=False)
        model.set_data(times, **data)
        flux = model(times, params['k'], params['t0'], params['p'], params['a'],
                     params['i'], params['e'], params['w'], data['lcids'],
                     data['pbids'], data['epids'], data['nsamples'],
                     data['exptimes'], 2, nor)

        assert flux.shape == (npv, times.size)
        assert (expected < -1e-6).any(), "test setup should produce in-transit points"
        np.testing.assert_allclose(flux, expected, atol=1e-12)


class TestEdgeCasesMatchNumba:
    """Geometric configurations that exercise the rarer `ccia` branches.

    `k > 1` is the only way to reach the `b <= r2 - r1` branch (star fully inside
    the planet's disk); `a` at or below 1 and a grazing impact parameter cover the
    degenerate and partial-overlap ends.
    """

    # `depth` is the expected minimum of the Numba reference, so each case asserts
    # the geometry it is named for actually occurred rather than silently comparing
    # two arrays of zeros.
    @pytest.mark.parametrize("name,k,a,inc,e,depth", [
        ('small_a',            0.10, 0.5, radians(87.0),          0.0, -0.0100),
        ('degenerate_a',       0.10, 1.0, radians(87.0),          0.0, -0.0100),
        ('grazing',            0.10, 8.0, np.arccos(0.95 / 8.0),  0.0, -0.0080),
        ('no_transit',         0.10, 8.0, radians(60.0),          0.0,  0.0),
        ('equal_radii',        1.00, 8.0, radians(89.99),         0.0, -0.9935),
        ('planet_bigger',      1.50, 8.0, radians(89.99),         0.0, -1.0000),
        ('high_eccentricity',  0.10, 8.0, radians(87.0),          0.9, -0.0100),
        ('tiny_k',             1e-4, 8.0, radians(89.99),         0.0, -1e-8),
    ])
    def test_forward(self, name, k, a, inc, e, depth):
        from pytransit.backends.opencl.udmodel import UDModelCL

        params = _make_2d_params(k, a=a, inc=inc, e=e)
        data = _simple_data(TIMES.size)
        expected = _call_numba(TIMES, params, data)

        model = UDModelCL(return_grad=False)
        model.set_data(TIMES, **data)
        flux = model(TIMES, params['k'], params['t0'], params['p'], params['a'],
                     params['i'], params['e'], params['w'], data['lcids'],
                     data['pbids'], data['epids'], data['nsamples'],
                     data['exptimes'], 1, 1)

        np.testing.assert_allclose(expected.min(), depth, rtol=1e-2, atol=1e-9,
                                   err_msg=f"the '{name}' setup no longer produces "
                                           f"the geometry it is testing")
        np.testing.assert_allclose(flux, expected, atol=1e-12)
