"""Tests for the OpenCL device-side log-likelihood and gradient reduction.

The model flux and its Jacobian are the largest arrays in a fit -- at npv=1,
npt=5e4 the Jacobian alone is 3.6 MB, and reading it back dominates the runtime.
Reducing it to a log-likelihood and its gradient on the device replaces that with
a handful of doubles.

Noise model: independent Gaussian errors with a fixed per-point `err` and a
sampled `jitter`, so ``e2 = err**2 + jitter**2``. Setting `err` to a constant
recovers the i.i.d. case.
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

from pytransit import RoadRunnerModel, UniformDiskModel

BJD = 2458325.5
P, A, I, W = 3.0, 6.0, radians(88.0), pi / 2
LDC = np.array([0.3, 0.2])


def reference_loglikelihood(model, obs, err, jitter, baseline=1.0, **pars):
    """NumPy oracle: reduce the model's own expanded Jacobian on the host.

    The residual is (obs - baseline) - dF, matching reduction.cl. Gradient columns are
    the Jacobian's, then d lnL/d baseline, then d lnL/d jitter -- jitter stays last.
    """
    flux, jac = model.evaluate(**pars)
    flux, jac = np.asarray(flux), np.asarray(jac)
    e2 = np.asarray(err) ** 2 + jitter ** 2
    r = (np.asarray(obs) - baseline) - flux
    lnl = (-0.5 * r.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(e2))
           - 0.5 * np.sum(r ** 2 / e2))
    grad = (r / e2) @ jac
    db = np.sum(r / e2)
    dj = jitter * (np.sum(r ** 2 / e2 ** 2) - np.sum(1.0 / e2))
    return lnl, np.concatenate([grad, [db], [dj]])


def _dataset(npt=2000, origin=BJD, seed=0):
    rng = np.random.default_rng(seed)
    times = np.linspace(-0.12, 0.12, npt) + origin
    err = 1e-3 * (1.0 + 0.5 * rng.random(npt))       # per-point, not i.i.d.
    return times, err, rng


def _rr(times, return_grad=True, **kwargs):
    m = RoadRunnerModel(backend='opencl', return_grad=return_grad,
                        ldmodel='quadratic', **kwargs)
    m.set_data(times)
    return m


class TestReferenceMathIsRight:
    """Validate the oracle itself against finite differences of lnL.

    Without this the device could agree with a wrong reference.
    """

    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def _lnl(self, model, obs, err, jitter, baseline=1.0, **overrides):
        pars = {**self.PARS, **overrides}
        flux = np.asarray(model.evaluate(**pars))
        e2 = err ** 2 + jitter ** 2
        # (obs - baseline) before subtracting flux: both operands are near 1.0, so
        # this subtraction is exact (Sterbenz), and the rest stays on the deficit scale.
        r = (obs - baseline) - flux
        return (-0.5 * r.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(e2))
                - 0.5 * np.sum(r ** 2 / e2))

    # `k` gets the loose tolerance the RoadRunner gradient tests use for the same
    # reason: the analytic k derivative neglects dldm/dk, the weight table's own
    # k-dependence, which costs a few percent. Every other column is exact.
    @pytest.mark.parametrize("name,idx,eps,rtol", [
        ('k', 0, 1e-7, 0.1), ('t0', 1, 1e-7, 2e-3), ('p', 2, 1e-6, 2e-3),
        ('a', 3, 1e-6, 2e-3), ('i', 4, 1e-7, 2e-3), ('e', 5, 1e-6, 2e-3),
        ('w', 6, 1e-5, 2e-3),
    ])
    def test_parameter_gradient(self, name, idx, eps, rtol):
        times, err, rng = _dataset()
        grad_model = _rr(times)
        fwd_model = _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter = 5e-4

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, **self.PARS)
        base = self.PARS[name]
        fd = (self._lnl(fwd_model, obs, err, jitter, **{name: base + eps})
              - self._lnl(fwd_model, obs, err, jitter, **{name: base - eps})) / (2 * eps)
        np.testing.assert_allclose(grad[idx], fd, rtol=rtol,
                                   atol=1e-3 * max(abs(fd), 1.0), err_msg=name)

    @pytest.mark.parametrize("j", [0, 1])
    def test_limb_darkening_gradient(self, j):
        times, err, rng = _dataset()
        grad_model, fwd_model = _rr(times), _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, eps = 5e-4, 1e-7

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, **self.PARS)
        lp, lm = LDC.copy(), LDC.copy()
        lp[j] += eps
        lm[j] -= eps
        fd = (self._lnl(fwd_model, obs, err, jitter, ldc=lp)
              - self._lnl(fwd_model, obs, err, jitter, ldc=lm)) / (2 * eps)
        np.testing.assert_allclose(grad[7 + j], fd, rtol=2e-3, atol=1e-3 * abs(fd))

    def test_jitter_gradient(self):
        times, err, rng = _dataset()
        grad_model, fwd_model = _rr(times), _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, eps = 5e-4, 1e-9

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, **self.PARS)
        fd = (self._lnl(fwd_model, obs, err, jitter + eps)
              - self._lnl(fwd_model, obs, err, jitter - eps)) / (2 * eps)
        np.testing.assert_allclose(grad[-1], fd, rtol=1e-5)

    def test_baseline_gradient(self):
        times, err, rng = _dataset()
        grad_model, fwd_model = _rr(times), _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, baseline, eps = 5e-4, 1.0, 1e-6

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, baseline, **self.PARS)
        fd = (self._lnl(fwd_model, obs, err, jitter, baseline + eps)
              - self._lnl(fwd_model, obs, err, jitter, baseline - eps)) / (2 * eps)
        # baseline is second from last; jitter keeps the last column.
        np.testing.assert_allclose(grad[-2], fd, rtol=1e-5)

    def test_jitter_is_still_the_last_column(self):
        times, err, rng = _dataset()
        grad_model, fwd_model = _rr(times), _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, eps = 5e-4, 1e-9

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, 1.0, **self.PARS)
        fd = (self._lnl(fwd_model, obs, err, jitter + eps, 1.0)
              - self._lnl(fwd_model, obs, err, jitter - eps, 1.0)) / (2 * eps)
        np.testing.assert_allclose(grad[-1], fd, rtol=1e-5)


class TestDeviceMatchesReference:
    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def test_simple(self):
        times, err, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(_rr(times, return_grad=False).evaluate(**self.PARS)) + rng.normal(0, err)
        jitter = 5e-4

        expected_lnl, expected_grad = reference_loglikelihood(model, obs, err, jitter, **self.PARS)
        lnl, grad = model.loglikelihood(obs, err, jitter, **self.PARS)

        assert np.ndim(lnl) == 0
        assert grad.shape == expected_grad.shape
        np.testing.assert_allclose(lnl, expected_lnl, rtol=1e-12)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-10, atol=1e-12)

    def test_device_matches_reference_with_baseline(self):
        times, err, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(_rr(times, return_grad=False).evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, baseline = 5e-4, 0.9997

        want_lnl, want_grad = reference_loglikelihood(model, obs, err, jitter, baseline, **self.PARS)
        got_lnl, got_grad = model.loglikelihood(obs, err, jitter, baseline=baseline, **self.PARS)
        np.testing.assert_allclose(got_lnl, want_lnl, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(got_grad), want_grad, rtol=1e-8, atol=1e-12)


def _hetero(npt_per_lc=500, origin=BJD, seed=1):
    """Three light curves: two passbands, three epochs, mixed supersampling."""
    rng = np.random.default_rng(seed)
    times = np.concatenate([np.linspace(-0.09, 0.09, npt_per_lc) + origin + m * P
                            for m in (0, 1, 3)])
    data = dict(lcids=np.repeat([0, 1, 2], npt_per_lc).astype(np.int32),
                pbids=np.array([0, 1, 0], dtype=np.int32),
                epids=np.array([0, 1, 2], dtype=np.int32),
                nsamples=np.array([1, 10, 5], dtype=np.int32),
                exptimes=np.array([0.0, 0.02, 0.01]))
    err = 1e-3 * (1.0 + 0.5 * rng.random(times.size))
    return times, data, err, rng


class TestDeviceMatchesReferenceHeterogeneous:
    """The slot table is what makes the reduction land in the expanded layout.

    Each parameter passed per passband or per epoch must reduce into its own
    column, and each shared parameter into a single one -- so these cases are
    where a wrong grouping rule shows up.
    """

    @pytest.mark.parametrize("orbit_variations", [False, True])
    @pytest.mark.parametrize("npv", [1, 4])
    def test_matches_reference(self, orbit_variations, npv, ):
        times, data, err, rng = _hetero()
        pv = np.arange(npv)[:, None] * 1e-4
        pars = dict(k=np.array([[0.10, 0.11]]) + pv,
                    t0=np.array([[BJD, BJD + P, BJD + 3 * P]]) + pv,
                    p=P, a=A, i=I, e=0.1, w=W,
                    ldc=np.array([[0.30, 0.20], [0.55, 0.10]]))
        if orbit_variations:
            pars['a'] = np.array([[A, A + 0.01, A + 0.02]]) + pv

        def build(return_grad):
            m = RoadRunnerModel(backend='opencl', return_grad=return_grad,
                                ldmodel='quadratic')
            m.set_data(times, **data, include_orbit_variations=orbit_variations)
            return m

        def single(ipv):
            """One parameter vector. `ldc` is (npb, nldc), not per-vector, so it
            must not be sliced -- doing so silently collapses npb to 1."""
            return {key: (value if key == 'ldc' or np.ndim(value) < 2 else value[ipv])
                    for key, value in pars.items()}

        # Observations come from the first parameter vector plus noise.
        obs = 1.0 + np.asarray(build(False).evaluate(**single(0))) + rng.normal(0, err)
        jitter = 4e-4

        reference = build(True)
        expected = [reference_loglikelihood(reference, obs, err, jitter, **single(ipv))
                    for ipv in range(npv)]
        expected_lnl = [lnl for lnl, _ in expected]
        expected_grad = [grad for _, grad in expected]

        model = build(True)

        lnl, grad = model.loglikelihood(obs, err, jitter, **pars)
        lnl, grad = np.atleast_1d(lnl), np.atleast_2d(grad)

        np.testing.assert_allclose(lnl, expected_lnl, rtol=1e-11)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-9, atol=1e-11)


class TestUniformDiskModel:
    """The reduction is model-agnostic: the uniform disk has no LD block."""

    def test_matches_reference(self):
        times, err, rng = _dataset()
        pars = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W)
        model = UniformDiskModel(backend='opencl', return_grad=True)
        model.set_data(times)
        obs = 1.0 + np.asarray(model.evaluate(**pars)[0]) + rng.normal(0, err)
        jitter = 5e-4

        expected_lnl, expected_grad = reference_loglikelihood(model, obs, err, jitter, **pars)
        lnl, grad = model.loglikelihood(obs, err, jitter, **pars)

        assert grad.size == 7 + 2, "7 orbital columns plus d/dbaseline and d/djitter, no LD block"
        np.testing.assert_allclose(lnl, expected_lnl, rtol=1e-12)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-10, atol=1e-12)


class TestNoiseHandling:
    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def test_scalar_err_is_the_iid_case(self):
        times, err_array, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0]) + rng.normal(0, err_array)
        sigma, jitter = 1.2e-3, 4e-4

        expected = reference_loglikelihood(model, obs, np.full(times.size, sigma),
                                           jitter, **self.PARS)
        got = model.loglikelihood(obs, sigma, jitter, **self.PARS)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-12)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-10, atol=1e-12)

    def test_zero_jitter(self):
        times, err, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0]) + rng.normal(0, err)

        expected = reference_loglikelihood(model, obs, err, 0.0, **self.PARS)
        got = model.loglikelihood(obs, err, 0.0, **self.PARS)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-12)
        np.testing.assert_allclose(got[1][:-1], expected[1][:-1], rtol=1e-10, atol=1e-12)

        # d lnL / d jitter carries a factor of jitter, so it vanishes here. Assert
        # the surrounding physics instead, since `== 0.0` alone cannot fail:
        # at a small non-zero jitter it must match a finite difference of lnL.
        assert got[1][-1] == 0.0
        eps = 1e-6
        small = model.loglikelihood(obs, err, eps, **self.PARS)[1][-1]
        fd = (reference_loglikelihood(model, obs, err, 2 * eps, **self.PARS)[0]
              - reference_loglikelihood(model, obs, err, 0.0, **self.PARS)[0]) / (2 * eps)
        np.testing.assert_allclose(small, fd, rtol=1e-4)


class TestInvalidParameterVectors:
    def test_nan_propagates(self):
        times, err, rng = _dataset()
        model = _rr(times)
        pars = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)
        obs = 1.0 + np.asarray(model.evaluate(**pars)[0]) + rng.normal(0, err)

        lnl, grad = model.loglikelihood(obs, err, 5e-4, **{**pars, 'a': 0.5})
        assert np.isnan(lnl), "an invalid parameter vector should give a NaN lnL"
        assert np.isnan(grad[:-1]).all(), "and a NaN parameter gradient"


class TestErrorHandling:
    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def test_requires_return_grad(self):
        times, err, _ = _dataset()
        model = _rr(times, return_grad=False)
        with pytest.raises(ValueError, match="return_grad"):
            model.loglikelihood(np.ones(times.size), err, 0.0, **self.PARS)

    def test_rejects_other_backends(self):
        times, err, _ = _dataset()
        model = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        model.set_data(times)
        with pytest.raises(NotImplementedError, match="opencl"):
            model.loglikelihood(np.ones(times.size), err, 0.0, **self.PARS)

    def test_rejects_mismatched_observations(self):
        times, err, _ = _dataset()
        model = _rr(times)
        with pytest.raises(ValueError, match="points"):
            model.loglikelihood(np.ones(times.size + 1), np.ones(times.size + 1),
                                0.0, **self.PARS)


class TestWNLogLikelihoodClass:
    """The backend-agnostic wrapper around the same reduction."""

    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def _setup(self):
        from pytransit.loglikelihoods import WNLogLikelihood
        times, err, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0]) + rng.normal(0, err)
        return model, obs, err, WNLogLikelihood(model, obs, err)

    def test_matches_the_model_method(self):
        model, obs, err, ll = self._setup()
        jitter = 4e-4
        expected = model.loglikelihood(obs, err, jitter, **self.PARS)
        got = ll(self.PARS['k'], self.PARS['t0'], self.PARS['p'], self.PARS['a'],
                 self.PARS['i'], self.PARS['e'], self.PARS['w'],
                 jitter=jitter, ldc=self.PARS['ldc'])
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-14)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-14)

    def test_scalar_err_is_broadcast(self):
        from pytransit.loglikelihoods import WNLogLikelihood
        times, err_array, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0]) + rng.normal(0, err_array)
        sigma = 1.1e-3

        ll = WNLogLikelihood(model, obs, sigma)
        assert ll.err.shape == (times.size,), "a scalar err must broadcast to one per point"
        got = ll(*[self.PARS[n] for n in 'k t0 p a i e w'.split()],
                 jitter=3e-4, ldc=LDC)
        expected = model.loglikelihood(obs, np.full(times.size, sigma), 3e-4, **self.PARS)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-14)

    def test_unsupported_backend_raises(self):
        from pytransit.loglikelihoods import WNLogLikelihood
        times, err, _ = _dataset()
        model = RoadRunnerModel(backend='numba', return_grad=True, ldmodel='quadratic')
        model.set_data(times)
        with pytest.raises(NotImplementedError, match="backend"):
            WNLogLikelihood(model, np.ones(times.size), err)


class TestObservationCache:
    """The upload is cached on array identity, which is only sound if the cache
    keeps the arrays alive -- `id()` is unique among live objects only, so a
    freed array's address can be recycled by a different one."""

    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def test_cached_arrays_are_kept_alive(self):
        import weakref
        times, err, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0]) + rng.normal(0, err)
        model.loglikelihood(obs, err, 4e-4, **self.PARS)

        ref = weakref.ref(model._model._obs_cache[0])
        del obs
        assert ref() is not None, "the cache must hold the array it uploaded"

    def test_new_observations_are_re_uploaded(self):
        times, err, rng = _dataset()
        model = _rr(times)
        base = 1.0 + np.asarray(model.evaluate(**self.PARS)[0])

        for offset in (0.0, 0.01, 0.02):
            obs = base + offset + rng.normal(0, err)
            expected, _ = reference_loglikelihood(model, obs, err, 4e-4, **self.PARS)
            got, _ = model.loglikelihood(obs, err, 4e-4, **self.PARS)
            np.testing.assert_allclose(got, expected, rtol=1e-11,
                                       err_msg=f"stale observations at offset {offset}")

    @pytest.mark.parametrize("npt", [1, 3, 255, 256, 257])
    def test_fewer_points_than_a_work_group(self, npt):
        """The reduction's tree collapse runs over a fixed work-group size."""
        times = np.linspace(-0.12, 0.12, npt) + BJD
        model = _rr(times)
        err = np.full(npt, 1e-3)
        obs = 1.0 + np.atleast_1d(np.asarray(model.evaluate(**self.PARS)[0])) + 1e-4

        flux, jac = model.evaluate(**self.PARS)
        flux = np.atleast_1d(np.asarray(flux))
        jac = np.asarray(jac).reshape(flux.size, -1)
        # (obs - baseline) before subtracting flux: baseline defaults to 1.0 here,
        # and both operands are near 1.0, so this stays on the small deficit scale.
        e2, r = err ** 2 + 4e-4 ** 2, (obs - 1.0) - flux
        expected_lnl = (-0.5 * npt * np.log(2 * np.pi) - 0.5 * np.sum(np.log(e2))
                        - 0.5 * np.sum(r ** 2 / e2))
        expected_grad = (r / e2) @ jac

        lnl, grad = model.loglikelihood(obs, err, 4e-4, **self.PARS)
        np.testing.assert_allclose(lnl, expected_lnl, rtol=1e-10)
        np.testing.assert_allclose(grad[:-2], expected_grad, rtol=1e-8, atol=1e-11)


class TestCopyArgumentIsRejected:
    def test_copy_kwarg_gives_a_clear_error(self):
        """`copy` belongs to evaluate; forwarding it here would collide."""
        times, err, _ = _dataset()
        model = _rr(times)
        pars = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)
        with pytest.raises(TypeError, match="copy"):
            model.loglikelihood(np.ones(times.size), err, 0.0, **pars, copy=True)


class TestSinglePrecision:
    """Single precision must stay usable, and its cache must not go stale.

    Two hazards live here: the identity-keyed upload cache pins arrays that
    `ascontiguousarray` copies in fp32, and the model-independent `sum(log e2)`
    term is large and moves with `jitter`, so a naive fp32 accumulation biases the
    jitter posterior rather than adding an ignorable constant.
    """

    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def _exact(self, model, obs, err, jitter, baseline=1.0):
        flux, jac = model.evaluate(**self.PARS)
        flux = np.asarray(flux, np.float64)
        jac = np.asarray(jac, np.float64)
        e2 = np.asarray(err, np.float64) ** 2 + jitter ** 2
        # (obs - baseline) before subtracting flux: both operands are near 1.0, so
        # this subtraction is exact (Sterbenz), and the rest stays on the deficit scale.
        r = (np.asarray(obs, np.float64) - baseline) - flux
        lnl = (-0.5 * r.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(e2))
               - 0.5 * np.sum(r ** 2 / e2))
        dj = jitter * (np.sum(r ** 2 / e2 ** 2) - np.sum(1.0 / e2))
        return lnl, (r / e2) @ jac, dj

    @pytest.mark.parametrize("npt", [2_000, 200_000])
    def test_accuracy(self, npt):
        times, err, rng = _dataset(npt=npt)
        model = _rr(times, precision='single')
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0], np.float64) + rng.normal(0, err)
        jitter = 4e-4

        exact_lnl, exact_grad, exact_dj = self._exact(model, obs, err, jitter)
        lnl, grad = model.loglikelihood(obs, err, jitter, **self.PARS)

        # A naive fp32 sum of log(e2) costs ~15 nats at npt=2e5; compensated
        # summation must keep the whole log-likelihood well inside one nat.
        assert abs(lnl - exact_lnl) < 0.5, f"lnL off by {lnl - exact_lnl:.3f} nats"
        np.testing.assert_allclose(grad[-1], exact_dj, rtol=1e-3)

    def test_changing_scalar_err_is_not_cached_stale(self):
        """A scalar `err` is broadcast to a fresh array; the cache must still see
        each distinct value. In fp32 the uploaded copy differs from the keyed
        object, which is exactly when a naive identity cache goes stale."""
        times, _, rng = _dataset()
        model = _rr(times, precision='single')
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0], np.float64) + 1e-4

        seen = []
        for sigma in (1e-3, 5e-3, 2e-2, 5e-3, 1e-3):
            lnl, _ = model.loglikelihood(obs, sigma, 4e-4, **self.PARS)
            exact, _, _ = self._exact(model, obs, np.full(times.size, sigma), 4e-4)
            np.testing.assert_allclose(lnl, exact, rtol=1e-4,
                                       err_msg=f"stale uncertainties at sigma={sigma}")
            seen.append(lnl)
        assert len(set(np.round(seen, 6))) == 3, "each distinct sigma must give its own lnL"


class TestMixedExpansion:
    """A shared parameter alongside an expanded one, which the heterogeneous
    tests never separate -- they always expand `k` and `t0` together."""

    @pytest.mark.parametrize("expand_k,expand_t0", [(True, False), (False, True), (False, False)])
    def test_matches_reference(self, expand_k, expand_t0):
        times, data, err, rng = _hetero()
        pars = dict(k=np.array([0.10, 0.11]) if expand_k else 0.105,
                    t0=np.array([BJD, BJD + P, BJD + 3 * P]) if expand_t0 else BJD,
                    p=P, a=A, i=I, e=0.1, w=W,
                    ldc=np.array([[0.30, 0.20], [0.55, 0.10]]))

        model = RoadRunnerModel(backend='opencl', return_grad=True, ldmodel='quadratic')
        model.set_data(times, **data)
        obs = 1.0 + np.asarray(model.evaluate(**pars)[0]) + rng.normal(0, err)

        expected_lnl, expected_grad = reference_loglikelihood(model, obs, err, 4e-4, **pars)
        lnl, grad = model.loglikelihood(obs, err, 4e-4, **pars)
        assert grad.size == expected_grad.size
        np.testing.assert_allclose(lnl, expected_lnl, rtol=1e-11)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-9, atol=1e-11)


class TestReductionGuards:
    PARS = dict(k=0.1, t0=BJD, p=P, a=A, i=I, e=0.1, w=W, ldc=LDC)

    def test_reduce_without_evaluating_raises(self):
        times, _, _ = _dataset()
        model = _rr(times)
        slots, ncol = model._gradient_slots(
            tuple(self.PARS[n] for n in 'k t0 p a i e w'.split()), 1, ldc=LDC)
        with pytest.raises(RuntimeError, match="copy=False"):
            model._model.reduce_loglikelihood(0.0, 1.0, slots, ncol, 1)

    def test_reduce_without_observations_raises(self):
        times, _, _ = _dataset()
        model = _rr(times)
        model.evaluate(**self.PARS, copy=False)
        slots, ncol = model._gradient_slots(
            tuple(self.PARS[n] for n in 'k t0 p a i e w'.split()), 1, ldc=LDC)
        with pytest.raises(RuntimeError, match="set_observations"):
            model._model.reduce_loglikelihood(0.0, 1.0, slots, ncol, 1)

    def test_mismatched_row_width_raises(self):
        """A gradient row width disagreeing with the Jacobian on the device must
        fail loudly; striding it wrongly would return plausible nonsense."""
        times, err, _ = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(model.evaluate(**self.PARS)[0]) + 1e-4
        model._model.set_observations(obs, err)
        model.evaluate(**self.PARS, copy=False)
        slots, ncol = model._gradient_slots(
            tuple(self.PARS[n] for n in 'k t0 p a i e w'.split()), 1, ldc=LDC)
        with pytest.raises(ValueError, match="row width"):
            model._model.reduce_loglikelihood(0.0, 1.0, slots, ncol + 1, 1)

    def test_missing_ldc_raises(self):
        times, err, _ = _dataset()
        model = _rr(times)
        pars = {key: value for key, value in self.PARS.items() if key != 'ldc'}
        with pytest.raises(ValueError, match="[Ll]imb darkening"):
            model.loglikelihood(np.ones(times.size), err, 0.0, **pars)
