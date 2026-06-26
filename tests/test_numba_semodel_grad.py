"""Tests for the Numba secondary-eclipse model gradient backend.

Verifies the low-level functions:
1. `_semodel_grad` — single-timestamp eclipse flux + gradient computation
2. `semodel_grad`  — full array-level model with orbit folding and supersampling

Gradient correctness is checked against central finite differences using the
forward-only `semodel` as the reference. The gradient layout is
``[k, t0, p, a, i, e, w]``.

The eclipse-specific subtleties exercised here:
- the ``pi * k**2`` baseline contributes a ``2 * pi * k`` term to the k-gradient
  that the transit (uniform-disk) model does not have;
- the eclipse-time offset cancels against the Taylor expansion point (so the
  e/w gradients need no extra terms), while the light-travel time does not and
  is corrected via ``d(flux)/d(t0) * d(ltt)/dX``.
"""

import numpy as np
import pytest
from math import radians, pi

from numba import njit
from meepmeep.numba2d import solve2d_d
from meepmeep.backends.numba.utils import eclipse_time_offset
from pytransit.backends.numba.semodel import semodel
from pytransit.backends.numba.semodel_grad import _semodel_grad, semodel_grad

semodel_jit = njit(semodel)
semodel_grad_jit = njit(semodel_grad)

# Shared orbital parameters
T0 = 0.0
P = 2.5
A = 8.0
I = radians(89.0)
E = 0.0
W = pi / 2
RSTAR = 1.0
ECL = T0 + P / 2


def _make_2d_params(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npv=1, npb=1, nep=1):
    return dict(k=np.full((npv, npb), k),
                t0=np.full((npv, nep), t0),
                p=np.full((npv, nep), p),
                a=np.full((npv, nep), a),
                i=np.full((npv, nep), inc),
                e=np.full((npv, nep), e),
                w=np.full((npv, nep), w))


def _call_semodel_grad(times, k=0.1, t0=T0, p=P, a=A, inc=I, e=E, w=W, rstar=RSTAR):
    npt = times.size
    pp = _make_2d_params(k, t0, p, a, inc, e, w)
    lcids = np.zeros(npt, dtype=np.int32)
    pbids = np.zeros(1, dtype=np.int32)
    epids = np.zeros(1, dtype=np.int32)
    nsamples = np.ones(1, dtype=np.int32)
    exptimes = np.zeros(1, dtype=np.float64)
    return semodel_grad_jit(times, pp['k'], pp['t0'], pp['p'], pp['a'], pp['i'], pp['e'], pp['w'],
                            rstar, lcids, pbids, epids, nsamples, exptimes, 1, 1)


def _call_semodel(times, k=0.1, t0=T0, p=P, a=A, inc=I, e=E, w=W, rstar=RSTAR):
    npt = times.size
    pp = _make_2d_params(k, t0, p, a, inc, e, w)
    lcids = np.zeros(npt, dtype=np.int32)
    pbids = np.zeros(1, dtype=np.int32)
    epids = np.zeros(1, dtype=np.int32)
    nsamples = np.ones(1, dtype=np.int32)
    exptimes = np.zeros(1, dtype=np.float64)
    return semodel_jit(times, pp['k'], pp['t0'], pp['p'], pp['a'], pp['i'], pp['e'], pp['w'],
                       rstar, lcids, pbids, epids, nsamples, exptimes, 1, 1)


def _finite_diff_grad(times, param_name, eps, **base):
    """Central finite-difference gradient by perturbing one parameter."""
    plus = dict(base); plus[param_name] = base[param_name] + eps
    minus = dict(base); minus[param_name] = base[param_name] - eps
    fp = _call_semodel(times, **plus)
    fm = _call_semodel(times, **minus)
    return (fp[0] - fm[0]) / (2 * eps)


class TestSemodelGradKernel:
    """Tests for `_semodel_grad` — single-timestamp flux + gradient."""

    def test_out_of_eclipse_baseline_gradient(self):
        """Out of eclipse: flux = pi*k**2 and only the k-gradient is nonzero (2*pi*k)."""
        k = 0.1
        cf, dcf = solve2d_d(eclipse_time_offset(P, I, E, W), P, A, I, E, W)
        flux = np.zeros(1)
        dflux = np.zeros(7)
        _semodel_grad(0.5, k, cf, dcf, flux, dflux)  # planet clear of the star
        np.testing.assert_allclose(flux[0], pi * k**2, atol=1e-12)
        np.testing.assert_allclose(dflux[0], 2 * pi * k, atol=1e-12)
        np.testing.assert_allclose(dflux[1:], 0.0, atol=1e-12)

    def test_gradient_shape(self):
        cf, dcf = solve2d_d(eclipse_time_offset(P, I, E, W), P, A, I, E, W)
        flux = np.zeros(1)
        dflux = np.zeros(7)
        _semodel_grad(0.0, 0.1, cf, dcf, flux, dflux)
        assert dflux.shape == (7,)

    def test_k_gradient_partial_overlap(self):
        """The k-gradient matches finite differences during ingress/egress."""
        from pytransit.backends.numba.semodel import _semodel
        k = 0.1
        cf, dcf = solve2d_d(eclipse_time_offset(P, I, E, W), P, A, I, E, W)
        # Find a partial-overlap time (flux strictly between 0 and pi*k**2).
        for t in np.linspace(0.02, 0.10, 50):
            flux = np.zeros(1)
            dflux = np.zeros(7)
            _semodel_grad(t, k, cf, dcf, flux, dflux)
            if 0.0 < flux[0] < pi * k**2:
                eps = 1e-7
                fpv = np.zeros(1); _semodel(t, k + eps, cf, fpv)
                fmv = np.zeros(1); _semodel(t, k - eps, cf, fmv)
                fd = (fpv[0] - fmv[0]) / (2 * eps)
                np.testing.assert_allclose(dflux[0], fd, rtol=1e-4, atol=1e-7)
                return
        pytest.fail("No partial overlap point found in the scanned range")


class TestSemodelGradFull:
    """Tests for `semodel_grad` — the full array-level model with gradients."""

    def test_output_shapes(self):
        times = np.linspace(ECL - 0.15, ECL + 0.15, 200)
        flux, dflux = _call_semodel_grad(times, k=0.1)
        assert flux.shape == (1, times.size)
        assert dflux.shape == (1, times.size, 7)

    def test_flux_matches_forward_model(self):
        times = np.linspace(ECL - 0.15, ECL + 0.15, 200)
        flux_grad, _ = _call_semodel_grad(times, k=0.1, e=0.1, w=radians(80.0))
        flux_fwd = _call_semodel(times, k=0.1, e=0.1, w=radians(80.0))
        np.testing.assert_allclose(flux_grad[0], flux_fwd[0], atol=1e-14)

    def test_out_of_eclipse_gradient(self):
        """Far from eclipse only the baseline k-gradient survives."""
        k = 0.1
        times = np.array([ECL - 0.5, ECL + 0.5])
        flux, dflux = _call_semodel_grad(times, k=k)
        np.testing.assert_allclose(flux[0], pi * k**2, atol=1e-12)
        np.testing.assert_allclose(dflux[0, :, 0], 2 * pi * k, atol=1e-12)
        np.testing.assert_allclose(dflux[0, :, 1:], 0.0, atol=1e-12)


class TestSemodelGradFiniteDiff:
    """Gradient correctness verified against central finite differences."""

    EPS = 1e-7
    K = 0.1

    # (label, e, w) configurations: circular, moderate-e, high-e.
    CONFIGS = [
        ("circular", 0.0, pi / 2),
        ("eccentric", 0.2, radians(60.0)),
        ("high_e", 0.3, radians(120.0)),
    ]

    def _window(self, e, w):
        center = T0 + eclipse_time_offset(P, I, e, w)
        return np.linspace(center - 0.12, center + 0.12, 600)

    @pytest.mark.parametrize("label,e,w", CONFIGS)
    @pytest.mark.parametrize("name,idx", [
        ('k', 0), ('t0', 1), ('p', 2), ('a', 3), ('inc', 4), ('e', 5), ('w', 6)
    ])
    def test_gradient(self, label, e, w, name, idx):
        """Each analytic gradient matches finite differences on partial-overlap points."""
        times = self._window(e, w)
        base = dict(k=self.K, t0=T0, p=P, a=A, inc=I, e=e, w=w)
        flux, dflux = _call_semodel_grad(times, **base)
        fd = _finite_diff_grad(times, name, self.EPS, **base)
        anal = dflux[0, :, idx]

        # Restrict to partial-overlap points: the flat baseline and the flat
        # total-eclipse floor carry trivial (constant) gradients, and the
        # area-function kinks at the contact points make finite differences
        # unreliable there.
        baseline = pi * self.K**2
        mask = (flux[0] > 1e-6) & (flux[0] < baseline - 1e-6)
        assert mask.any(), "Should have partial-overlap points"
        np.testing.assert_allclose(anal[mask], fd[mask], rtol=1e-3, atol=1e-8,
                                   err_msg=f"{name} gradient ({label})")
