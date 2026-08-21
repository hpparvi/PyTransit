"""Model-class level tests for the OpenCL uniform-disk backend.

These exercise `UniformDiskModel(backend='opencl')` end to end, including the
host-side parameter normalisation and gradient expansion that the low-level
kernel tests bypass.
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

from pytransit.models.udmodel import UniformDiskModel

P = 2.5
A = 8.0
I = radians(87.0)
W = pi / 2


def _hetero_setup():
    """Three light curves: two passbands, three epochs, mixed supersampling."""
    times = np.concatenate([np.linspace(-0.15, 0.15, 400) + n * P for n in (0, 1, 3)])
    data = dict(lcids=np.repeat([0, 1, 2], 400).astype(np.int32),
                pbids=np.array([0, 1, 0], dtype=np.int32),
                epids=np.array([0, 1, 2], dtype=np.int32),
                nsamples=np.array([1, 10, 5], dtype=np.int32),
                exptimes=np.array([0.0, 0.02, 0.01]))
    # Per-passband k and per-epoch t0 given as arrays, so the gradient expands
    # into one column per passband and per epoch rather than staying shared.
    pars = dict(k=np.array([0.10, 0.11]), t0=np.array([0.0, P, 3 * P]),
                p=P, a=A, i=I, e=0.1, w=W)
    return times, data, pars


def _evaluate(backend, return_grad, times, data, pars, **kwargs):
    model = UniformDiskModel(backend=backend, return_grad=return_grad, **kwargs)
    model.set_data(times, **data)
    return model, model.evaluate(**pars)


class TestMatchesNumbaBackend:
    def test_flux(self):
        times, data, pars = _hetero_setup()
        _, flux_nb = _evaluate('numba', False, times, data, pars)
        _, flux_cl = _evaluate('opencl', False, times, data, pars)

        assert flux_cl.shape == (times.size,)
        assert (flux_nb < -1e-6).any(), "test setup should produce in-transit points"
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)

    def test_expanded_gradient(self):
        times, data, pars = _hetero_setup()
        _, (flux_nb, dflux_nb) = _evaluate('numba', True, times, data, pars)
        _, (flux_cl, dflux_cl) = _evaluate('opencl', True, times, data, pars)

        # k expands to 2 passband columns, t0 to 3 epoch columns, p/a/i/e/w stay
        # shared: 2 + 3 + 5 = 10 columns.
        assert dflux_cl.shape == (times.size, 10)
        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-12)
        np.testing.assert_allclose(dflux_cl, dflux_nb, atol=1e-12)

    def test_orbit_variations(self):
        times, data, pars = _hetero_setup()
        pars = {**pars, 'a': np.array([A, A + 0.01, A + 0.02])}
        out = {}
        for backend in ('numba', 'opencl'):
            model = UniformDiskModel(backend=backend, return_grad=True)
            model.set_data(times, **data, include_orbit_variations=True)
            out[backend] = model.evaluate(**pars)

        for expected, got in zip(out['numba'], out['opencl']):
            np.testing.assert_allclose(got, expected, atol=1e-12)


class TestBackendSelection:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="[Uu]nknown backend"):
            UniformDiskModel(backend='cuda')


class TestCopyFlag:
    def test_copy_false_returns_none_and_keeps_device_result(self):
        times, data, pars = _hetero_setup()
        model = UniformDiskModel(backend='opencl', return_grad=False)
        model.set_data(times, **data)

        expected = model.evaluate(**pars)
        assert model.evaluate(**pars, copy=False) is None

        # The device still holds the result, and it is the same one.
        flux = np.zeros((1, times.size))
        pyopencl.enqueue_copy(model._model.queue, flux, model._model._b_flux)
        np.testing.assert_allclose(flux[0], expected, atol=1e-12)

    def test_copy_false_rejected_by_other_backends(self):
        times, data, pars = _hetero_setup()
        model = UniformDiskModel(backend='numba', return_grad=False)
        model.set_data(times, **data)
        with pytest.raises(ValueError, match="copy"):
            model.evaluate(**pars, copy=False)


class TestSinglePrecision:
    def test_close_to_numba(self):
        times, data, pars = _hetero_setup()
        _, flux_nb = _evaluate('numba', False, times, data, pars)
        _, flux_cl = _evaluate('opencl', False, times, data, pars, precision='single')

        np.testing.assert_allclose(flux_cl, flux_nb, atol=1e-5)

    def test_unknown_precision_raises(self):
        with pytest.raises(ValueError, match="[Pp]recision"):
            UniformDiskModel(backend='opencl', precision='half')


def _numba_reference(times, pars, return_grad, **data):
    model = UniformDiskModel(backend='numba', return_grad=return_grad)
    model.set_data(times, **data)
    return model.evaluate(**pars)


def _as_tuple(result):
    return result if isinstance(result, tuple) else (result,)


class TestBufferLifecycle:
    """One model reused across shape changes must not serve stale device buffers."""

    TIMES = np.linspace(-0.15, 0.15, 500)
    PARS = dict(t0=0.0, p=P, a=A, i=I, e=0.1, w=W)

    @pytest.mark.parametrize("return_grad", [False, True])
    def test_changing_npv(self, return_grad):
        model = UniformDiskModel(backend='opencl', return_grad=return_grad)
        model.set_data(self.TIMES)

        # Grow, shrink, and repeat a previously seen size.
        for npv in (1, 7, 3, 7, 20, 2):
            pars = dict(k=0.1 + np.arange(npv)[:, None] * 1e-3, **self.PARS)
            got = _as_tuple(model.evaluate(**pars))
            expected = _as_tuple(_numba_reference(self.TIMES, pars, return_grad))
            for g, e in zip(got, expected):
                np.testing.assert_allclose(g, e, atol=1e-12,
                                           err_msg=f"npv={npv}, return_grad={return_grad}")

    @pytest.mark.parametrize("npt", [137, 900])
    def test_set_data_called_again_with_different_length(self, npt):
        """Covers both shrinking and *growing*; growing is what overruns a stale buffer.

        The first evaluation is load-bearing: it forces the output buffers to be
        allocated at the original length, so the second `set_data` has a stale
        allocation to invalidate.
        """
        model = UniformDiskModel(backend='opencl', return_grad=True)
        pars = dict(k=0.1, **self.PARS)
        model.set_data(self.TIMES)
        model.evaluate(**pars)

        times = np.linspace(-0.2, 0.2, npt)
        model.set_data(times)

        got = model.evaluate(**pars)
        expected = _numba_reference(times, pars, True)
        for g, e in zip(got, expected):
            np.testing.assert_allclose(g, e, atol=1e-12)

    def test_copy_false_leaves_compact_gradient_on_device(self):
        """With return_grad, copy=False must leave the compact (npv, npt, 7) gradient."""
        model = UniformDiskModel(backend='opencl', return_grad=True)
        model.set_data(self.TIMES)
        pars = dict(k=0.1, **self.PARS)

        expected_flux, expected_dflux = _numba_reference(self.TIMES, pars, True)
        assert model.evaluate(**pars, copy=False) is None

        backend = model._model
        flux = np.zeros((1, self.TIMES.size))
        dflux = np.zeros((1, self.TIMES.size, 7))
        pyopencl.enqueue_copy(backend.queue, flux, backend._b_flux)
        pyopencl.enqueue_copy(backend.queue, dflux, backend._b_dflux)

        np.testing.assert_allclose(flux[0], expected_flux, atol=1e-12)
        # All seven parameters share one column here, so the compact gradient and the
        # expanded one coincide; the point is that the device buffer is populated.
        np.testing.assert_allclose(dflux[0], expected_dflux, atol=1e-12)

    def test_copy_false_then_copy_true(self):
        """A skipped readback must not corrupt the next evaluation."""
        model = UniformDiskModel(backend='opencl', return_grad=True)
        model.set_data(self.TIMES)
        pars = dict(k=0.1, **self.PARS)

        assert model.evaluate(**pars, copy=False) is None
        got = model.evaluate(**pars)
        expected = _numba_reference(self.TIMES, pars, True)
        for g, e in zip(got, expected):
            np.testing.assert_allclose(g, e, atol=1e-12)


class TestRealisticTimeStamps:
    """Absolute epochs must not destroy precision, especially in single precision.

    Light curve times are normally given as BJD (~2.46e6) or a mission-specific
    offset such as TESS BTJD (~3e3). A float32 ULP at BJD 2458325 is 0.25 d, so a
    kernel that folds the *absolute* time in single precision collapses the whole
    transit onto a couple of representable values while raising no error at all.
    """

    ORIGINS = [('zero', 0.0), ('btjd', 3000.0), ('bjd', 2458325.5)]

    @pytest.mark.parametrize("label,origin", ORIGINS)
    def test_double_precision(self, label, origin):
        times = origin + np.linspace(-0.15, 0.15, 300)
        pars = dict(k=0.1, t0=origin, p=P, a=A, i=I, e=0.0, w=W)

        expected = _numba_reference(times, pars, False)
        _, got = _evaluate('opencl', False, times, {}, pars)

        assert (expected < -1e-6).any(), "test setup should produce in-transit points"
        np.testing.assert_allclose(got, expected, atol=1e-12)

    @pytest.mark.parametrize("label,origin", ORIGINS)
    def test_single_precision(self, label, origin):
        times = origin + np.linspace(-0.15, 0.15, 300)
        pars = dict(k=0.1, t0=origin, p=P, a=A, i=I, e=0.0, w=W)

        expected = _numba_reference(times, pars, False)
        _, got = _evaluate('opencl', False, times, {}, pars, precision='single')

        # The documented single-precision accuracy, which must hold at any epoch.
        np.testing.assert_allclose(got, expected, atol=1e-5)
        # The transit must still be resolved, not flattened into a step function.
        assert np.unique(np.asarray(got).round(6)).size > 5, \
            "single precision collapsed the transit shape"
