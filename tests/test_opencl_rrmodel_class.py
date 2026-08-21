"""Model-class level tests for the OpenCL RoadRunner backend.

These exercise `RoadRunnerModel(backend='opencl')` end to end, including the
host-side limb-darkening tabulation, parameter normalisation, and gradient
expansion that the low-level kernel tests bypass.
"""

import numpy as np
import pytest
from math import radians, pi

pyopencl = pytest.importorskip("pyopencl")

try:
    _HAS_OPENCL_DEVICE = bool(pyopencl.get_platforms())
except pyopencl.Error:
    _HAS_OPENCL_DEVICE = False

if not _HAS_OPENCL_DEVICE:
    pytest.skip("No OpenCL platform available", allow_module_level=True)

from pytransit.models.rrmodel import RoadRunnerModel

BJD = 2458325.5
P = 3.0
A = 6.0
I = radians(88.0)
W = pi / 2


def _hetero_setup(origin=BJD, n=300):
    """Three light curves: two passbands, three epochs, mixed supersampling."""
    times = np.concatenate([np.linspace(-0.08, 0.08, n) + origin + m * P for m in (0, 1, 3)])
    data = dict(lcids=np.repeat([0, 1, 2], n).astype(np.int32),
                pbids=np.array([0, 1, 0], dtype=np.int32),
                epids=np.array([0, 1, 2], dtype=np.int32),
                nsamples=np.array([1, 10, 5], dtype=np.int32),
                exptimes=np.array([0.0, 0.02, 0.01]))
    # Per-passband k and ldc, per-epoch t0, so the gradient expands into one
    # column per passband and per epoch instead of staying shared.
    pars = dict(k=np.array([0.10, 0.11]), t0=np.array([origin, origin + P, origin + 3 * P]),
                p=P, a=A, i=I, e=0.1, w=W,
                ldc=np.array([[0.30, 0.20], [0.55, 0.10]]))
    return times, data, pars


def _evaluate(backend, return_grad, times, data, pars, ldmodel='quadratic', **kwargs):
    model = RoadRunnerModel(backend=backend, return_grad=return_grad,
                            ldmodel=ldmodel, **kwargs)
    model.set_data(times, **data)
    return model, model.evaluate(**pars)


class TestMatchesNumbaBackend:
    def test_flux(self):
        times, data, pars = _hetero_setup()
        _, flux_nb = _evaluate('numba', False, times, data, pars)
        _, flux_cl = _evaluate('opencl', False, times, data, pars)

        assert flux_cl.shape == (times.size,)
        assert (flux_nb < 0.9999).any(), "test setup should produce in-transit points"
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)

    def test_expanded_gradient(self):
        times, data, pars = _hetero_setup()
        _, (flux_nb, dflux_nb) = _evaluate('numba', True, times, data, pars)
        _, (flux_cl, dflux_cl) = _evaluate('opencl', True, times, data, pars)

        # k -> 2 passband columns, t0 -> 3 epoch columns, p/a/i/e/w shared (5),
        # ldc -> npb*nldc = 4 columns.  2 + 3 + 5 + 4 = 14.
        assert dflux_cl.shape == (times.size, 14)
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)

    def test_orbit_variations(self):
        times, data, pars = _hetero_setup()
        pars = {**pars, 'a': np.array([A, A + 0.01, A + 0.02])}
        out = {}
        for backend in ('numba', 'opencl'):
            model = RoadRunnerModel(backend=backend, return_grad=True, ldmodel='quadratic')
            model.set_data(times, **data, include_orbit_variations=True)
            out[backend] = model.evaluate(**pars)
        for expected, got in zip(out['numba'], out['opencl']):
            np.testing.assert_allclose(got, expected, atol=1e-12)

    # `general` at two widths exercises the lazy -DNLDC rebuild for an arbitrary
    # coefficient count, not just the 1/2/4 the named profiles happen to use.
    @pytest.mark.parametrize("ldmodel,ldc", [
        ('linear',        np.array([0.4])),
        ('quadratic',     np.array([0.3, 0.2])),
        ('quadratic-tri', np.array([0.4, 0.3])),
        ('power-2',       np.array([0.5, 0.7])),
        ('nonlinear',     np.array([0.4, 0.1, 0.2, -0.1])),
        ('general',       np.array([0.1, 0.05, 0.02])),
        ('general',       np.array([0.1, 0.05, 0.02, 0.01, 0.005])),
    ])
    def test_limb_darkening_models(self, ldmodel, ldc):
        times = np.linspace(-0.08, 0.08, 300) + BJD
        pars = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=ldc)
        _, (f_nb, d_nb) = _evaluate('numba', True, times, {}, pars, ldmodel=ldmodel)
        _, (f_cl, d_cl) = _evaluate('opencl', True, times, {}, pars, ldmodel=ldmodel)

        assert (f_nb < 0.9999).any(), f"{ldmodel} setup should transit"
        assert d_cl.shape == (times.size, 7 + ldc.size)
        np.testing.assert_allclose(f_cl, f_nb, atol=1e-12)
        np.testing.assert_allclose(d_cl, d_nb, atol=1e-12)


class TestBackendSelection:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="[Uu]nknown backend"):
            RoadRunnerModel(backend='cuda')

    def test_jax_gradients_still_rejected(self):
        with pytest.raises(NotImplementedError):
            RoadRunnerModel(backend='jax', return_grad=True)

    def test_unknown_opencl_option_raises(self):
        with pytest.raises(ValueError, match="[Uu]nknown OpenCL"):
            RoadRunnerModel(backend='opencl', precission='single')


class TestCopyFlag:
    def test_copy_false_returns_none_and_keeps_device_result(self):
        times, data, pars = _hetero_setup()
        model = RoadRunnerModel(backend='opencl', return_grad=False, ldmodel='quadratic')
        model.set_data(times, **data)

        expected = model.evaluate(**pars)
        assert model.evaluate(**pars, copy=False) is None

        flux = np.zeros((1, times.size))
        pyopencl.enqueue_copy(model._model.queue, flux, model._model._b_flux)
        np.testing.assert_allclose(flux[0], expected, atol=1e-12)

    def test_copy_false_with_gradients_keeps_the_compact_jacobian(self):
        times, data, pars = _hetero_setup()
        model = RoadRunnerModel(backend='opencl', return_grad=True, ldmodel='quadratic')
        model.set_data(times, **data)

        expected_flux, _ = model.evaluate(**pars)
        assert model.evaluate(**pars, copy=False) is None

        backend = model._model
        npb, nldc = 2, 2
        flux = np.zeros((1, times.size))
        dflux = np.zeros((1, times.size, 7 + npb * nldc))
        pyopencl.enqueue_copy(backend.queue, flux, backend._b_flux)
        pyopencl.enqueue_copy(backend.queue, dflux, backend._b_dflux)

        np.testing.assert_allclose(flux[0], expected_flux, atol=1e-12)
        # The device holds the compact per-passband layout, not the expanded one.
        assert np.isfinite(dflux).all() and np.abs(dflux).max() > 0.0

    def test_copy_false_rejected_by_other_backends(self):
        times, data, pars = _hetero_setup()
        model = RoadRunnerModel(backend='numba', return_grad=False, ldmodel='quadratic')
        model.set_data(times, **data)
        with pytest.raises(ValueError, match="copy"):
            model.evaluate(**pars, copy=False)


class TestSinglePrecision:
    def test_close_to_numba(self):
        """RoadRunner's (ldi - Ip*A)/ldi loses more to fp32 than the uniform disk does."""
        times, data, pars = _hetero_setup()
        _, flux_nb = _evaluate('numba', False, times, data, pars)
        _, flux_cl = _evaluate('opencl', False, times, data, pars, precision='single')
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-5)

    def test_transit_shape_survives_a_bjd_epoch(self):
        times, data, pars = _hetero_setup()
        _, flux = _evaluate('opencl', False, times, data, pars, precision='single')
        assert np.unique(np.asarray(flux).round(6)).size > 20, \
            "single precision collapsed the transit shape"


class TestBufferLifecycle:
    """One model reused across shape changes must not serve stale device buffers."""

    @pytest.mark.parametrize("return_grad", [False, True])
    def test_changing_npv(self, return_grad):
        times = np.linspace(-0.08, 0.08, 400) + BJD
        base = dict(t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=np.array([0.3, 0.2]))
        model = RoadRunnerModel(backend='opencl', return_grad=return_grad, ldmodel='quadratic')
        model.set_data(times)

        for npv in (1, 7, 3, 7, 20, 2):
            pars = dict(k=0.1 + np.arange(npv)[:, None] * 1e-3, **base)
            reference = RoadRunnerModel(backend='numba', return_grad=return_grad,
                                        ldmodel='quadratic')
            reference.set_data(times)
            expected = reference.evaluate(**pars)
            got = model.evaluate(**pars)
            got = got if isinstance(got, tuple) else (got,)
            expected = expected if isinstance(expected, tuple) else (expected,)
            for g, e in zip(got, expected):
                np.testing.assert_allclose(g, e, atol=1e-12, err_msg=f"npv={npv}")

    @pytest.mark.parametrize("npt", [137, 900])
    def test_set_data_called_again_with_different_length(self, npt):
        """The first evaluation is load-bearing: it allocates buffers to invalidate."""
        times = np.linspace(-0.08, 0.08, 400) + BJD
        pars = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=np.array([0.3, 0.2]))
        model = RoadRunnerModel(backend='opencl', return_grad=True, ldmodel='quadratic')
        model.set_data(times)
        model.evaluate(**pars)

        times2 = np.linspace(-0.1, 0.1, npt) + BJD
        model.set_data(times2)
        reference = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        reference.set_data(times2)
        for g, e in zip(model.evaluate(**pars), reference.evaluate(**pars)):
            np.testing.assert_allclose(g, e, atol=1e-12)

    def test_changing_nldc_rebuilds_the_program(self):
        """nldc is a compile-time kernel constant, so a new width must rebuild.

        This has to reuse ONE model instance: constructing a fresh one each time
        would start with an empty program cache and never resize `_b_dldm` or
        `_b_dflux` on a live object, so it would not test either mechanism.
        `general` accepts any coefficient count, so the width can vary freely.
        """
        times = np.linspace(-0.13, 0.13, 300) + BJD
        pars = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W)

        model = RoadRunnerModel(backend='opencl', return_grad=True, ldmodel='general')
        model.set_data(times)
        reference = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='general')
        reference.set_data(times)

        widths = [3, 5, 3, 1]
        for nldc in widths:
            ldc = 0.1 / np.arange(1, nldc + 1)
            f_nb, d_nb = reference.evaluate(**pars, ldc=ldc)
            f_cl, d_cl = model.evaluate(**pars, ldc=ldc)
            assert d_cl.shape == (times.size, 7 + nldc), f"width {nldc}"
            np.testing.assert_allclose(f_cl, f_nb, atol=1e-12, err_msg=f"width {nldc}")
            np.testing.assert_allclose(d_cl, d_nb, atol=1e-12, err_msg=f"width {nldc}")

        assert set(model._model._programs) == {1, 3, 5}, \
            "one cached program per distinct nldc"


class TestDeviceMemoryLimits:
    """A Jacobian larger than the device's per-buffer ceiling must fail clearly.

    `cl.Buffer` allocation is lazy, so an oversized request otherwise surfaces as
    a bare MEM_OBJECT_ALLOCATION_FAILURE at kernel-enqueue time, naming neither
    the buffer nor the size.
    """

    def test_oversized_gradient_raises_informative_error(self):
        from pytransit.backends.opencl.context import get_context

        device = get_context().devices[0]
        # Pick npv so the (npv, npt, 7+nldc) fp64 Jacobian exceeds the ceiling.
        npt = 100_000
        per_pv = npt * 9 * 8
        npv = int(device.max_mem_alloc_size // per_pv) + 2

        times = np.linspace(-0.08, 0.08, npt) + BJD
        model = RoadRunnerModel(backend='opencl', return_grad=True, ldmodel='quadratic')
        model.set_data(times)
        pars = dict(k=0.1 + np.arange(npv)[:, None] * 1e-6, t0=BJD, p=P, a=A, i=I,
                    e=0.0, w=W, ldc=np.array([0.3, 0.2]))

        with pytest.raises(MemoryError, match="max_mem_alloc_size|too large"):
            model.evaluate(**pars)


class TestNonFiniteRadiusRatio:
    """A NaN radius ratio must degrade like Numba, not fault the device.

    `k` is not covered by the parameter-vector validity guard, and `g = z/(1+k)`
    is then NaN. NaN fails both `g < 0` and `g > 1`, and `(int)floor(NaN)` is
    INT_MIN on this device, so an unguarded interpolation indexes the table at a
    -16 GB offset. That is a memory fault, and it poisons the shared OpenCL
    context for every later call in the process -- so the failure is not a wrong
    number in one evaluation but a sampler that cannot continue.
    """

    TIMES = np.linspace(-0.08, 0.08, 200)
    PARS = dict(t0=0.0, p=P, a=A, i=I, e=0.0, w=W, ldc=np.array([0.3, 0.2]))

    @pytest.mark.parametrize("return_grad", [False, True])
    def test_matches_numba_and_leaves_the_context_usable(self, return_grad):
        reference = RoadRunnerModel(backend='numba', return_grad=return_grad,
                                    ldmodel='quadratic')
        reference.set_data(self.TIMES)
        expected = reference.evaluate(k=np.nan, **self.PARS)
        expected = expected if isinstance(expected, tuple) else (expected,)
        assert all(np.isnan(np.asarray(e)).all() for e in expected), \
            "the reference should return all NaN for a NaN radius ratio"

        model = RoadRunnerModel(backend='opencl', return_grad=return_grad,
                                ldmodel='quadratic')
        model.set_data(self.TIMES)
        got = model.evaluate(k=np.nan, **self.PARS)
        got = got if isinstance(got, tuple) else (got,)
        for g in got:
            assert np.isnan(np.asarray(g)).all(), "NaN k should give an all-NaN result"

        # The decisive part: the context must still be usable afterwards.
        good = model.evaluate(k=0.1, **self.PARS)
        good = good if isinstance(good, tuple) else (good,)
        assert np.isfinite(np.asarray(good[0])).all(), \
            "a NaN evaluation poisoned the OpenCL context"
        ref_good = reference.evaluate(k=0.1, **self.PARS)
        ref_good = ref_good if isinstance(ref_good, tuple) else (ref_good,)
        for g, e in zip(good, ref_good):
            np.testing.assert_allclose(g, e, atol=1e-12)

    def test_nan_in_one_passband_only(self):
        n = 200
        times = np.concatenate([np.linspace(-0.08, 0.08, n)] * 2)
        data = dict(lcids=np.repeat([0, 1], n).astype(np.int32),
                    pbids=np.array([0, 1], dtype=np.int32),
                    epids=np.zeros(2, dtype=np.int32),
                    nsamples=np.ones(2, dtype=np.int32), exptimes=np.zeros(2))
        pars = dict(k=np.array([0.1, np.nan]), t0=0.0, p=P, a=A, i=I, e=0.0, w=W,
                    ldc=np.array([[0.3, 0.2], [0.4, 0.1]]))

        _, expected = _evaluate('numba', False, times, data, pars)
        _, got = _evaluate('opencl', False, times, data, pars)
        expected, got = np.asarray(expected), np.asarray(got)

        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
        finite = ~np.isnan(expected)
        assert finite.any(), "the good passband should still evaluate"
        np.testing.assert_allclose(got[finite], expected[finite], atol=1e-12)
