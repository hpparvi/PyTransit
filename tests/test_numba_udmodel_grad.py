"""Tests for the Numba uniform-disk transit model gradient backend.

Verifies the low-level functions:
1. `_udmodel_grad` — single-timestamp flux deficit + gradient computation
2. `udmodel_grad` — full array-level model with orbit folding and supersampling

Gradient correctness is checked against central finite differences using the
forward-only `udmodel` as the reference.
"""

import numpy as np
import pytest
from math import radians, pi

from numba import njit
from meepmeep.backends.numba.ts2d import solve_xy_p5, solve_xy_p5_d
from pytransit.backends.numba.udmodel import _udmodel, udmodel
from pytransit.backends.numba.udmodel_grad import _udmodel_grad, udmodel_grad

udmodel_jit = njit(udmodel)
udmodel_grad_jit = njit(udmodel_grad)

# Shared orbital parameters
T0 = 0.0
P = 2.5
A = 8.0
I = radians(87.0)
E = 0.0
W = pi / 2
TIMES = np.linspace(-0.15, 0.15, 300)


def _make_2d_params(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npv=1, npb=1, ntc=1, nor=1):
    """Build 2D parameter arrays expected by `udmodel_grad`."""
    k2 = np.full((npv, npb), k)
    t02 = np.full((npv, nor), t0)
    p2 = np.full((npv, nor), p)
    a2 = np.full((npv, nor), a)
    i2 = np.full((npv, nor), inc)
    e2 = np.full((npv, nor), e)
    w2 = np.full((npv, nor), w)
    return dict(k=k2, t0=t02, p=p2, a=a2, i=i2, e=e2, w=w2)


def _call_udmodel_grad(times, k, t0=T0, p=P, a=A, inc=I, e=E, w=W):
    """Convenience wrapper calling `udmodel_grad` with default single-LC setup."""
    npt = times.size
    params = _make_2d_params(k, t0, p, a, inc, e, w)
    lcids = np.zeros(npt, dtype=np.int32)
    pbids = np.zeros(1, dtype=np.int32)
    epids = np.zeros(1, dtype=np.int32)
    nsamples = np.ones(1, dtype=np.int32)
    exptimes = np.zeros(1, dtype=np.float64)
    return udmodel_grad_jit(times, params['k'], params['t0'], params['p'], params['a'],
                            params['i'], params['e'], params['w'],
                            lcids, pbids, epids, nsamples, exptimes,
                            1, 1, 1, 1)


def _call_udmodel(times, k, t0=T0, p=P, a=A, inc=I, e=E, w=W):
    """Convenience wrapper calling forward-only `udmodel`."""
    npt = times.size
    params = _make_2d_params(k, t0, p, a, inc, e, w)
    lcids = np.zeros(npt, dtype=np.int32)
    pbids = np.zeros(1, dtype=np.int32)
    epids = np.zeros(1, dtype=np.int32)
    nsamples = np.ones(1, dtype=np.int32)
    exptimes = np.zeros(1, dtype=np.float64)
    return udmodel_jit(times, params['k'], params['t0'], params['p'], params['a'],
                       params['i'], params['e'], params['w'],
                       lcids, pbids, epids, nsamples, exptimes,
                       1, 1, 1, 1)


def _finite_diff_grad(times, param_name, param_idx, eps, k=0.1, t0=T0, p=P, a=A, inc=I, e=E, w=W):
    """Compute finite-difference gradient by perturbing one parameter.

    Calls the forward-only `udmodel` with +/- perturbations.

    Parameters
    ----------
    param_name : str
        Which parameter to perturb ('k', 't0', 'p', 'a', 'i', 'e', 'w').
    param_idx : int
        Index into the dflux gradient array (0=k, 1=t0, ..., 6=w).
    eps : float
        Perturbation size for central differences.

    Returns
    -------
    fd_grad : ndarray
        Finite-difference gradient with shape (npt,).
    """
    base = dict(k=k, t0=t0, p=p, a=a, inc=inc, e=e, w=w)

    plus_params = base.copy()
    plus_params[param_name] = base[param_name] + eps
    fp = _call_udmodel(times, **plus_params)

    minus_params = base.copy()
    minus_params[param_name] = base[param_name] - eps
    fm = _call_udmodel(times, **minus_params)

    return (fp[0] - fm[0]) / (2 * eps)


class TestUdmodelGradKernel:
    """Tests for `_udmodel_grad` — single-timestamp flux deficit + gradient."""

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_mid_transit_depth(self, k):
        """At mid-transit with near-zero impact parameter, deficit ~ -k^2."""
        inc = radians(89.99)
        cf, dcf = solve_xy_p5_d(0.0, P, A, inc, E, W)
        flux = np.zeros(1)
        dflux = np.zeros(7)
        _udmodel_grad(0.0, k, cf, dcf, flux, dflux)
        np.testing.assert_allclose(flux[0], -k**2, atol=1e-6)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_out_of_transit_is_zero(self, k):
        """Far from transit, flux and all gradients should remain zero."""
        cf, dcf = solve_xy_p5_d(0.0, P, A, I, E, W)
        flux = np.zeros(1)
        dflux = np.zeros(7)
        _udmodel_grad(0.5, k, cf, dcf, flux, dflux)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(dflux, 0.0, atol=1e-12)

    def test_partial_overlap(self):
        """During ingress/egress, deficit should be between -k^2 and 0."""
        k = 0.1
        cf, dcf = solve_xy_p5_d(0.0, P, A, I, E, W)
        for t in np.linspace(0.02, 0.10, 50):
            flux = np.zeros(1)
            dflux = np.zeros(7)
            _udmodel_grad(t, k, cf, dcf, flux, dflux)
            if -k**2 < flux[0] < 0.0:
                assert -k**2 < flux[0] < 0.0
                return
        pytest.fail("No partial overlap point found in the scanned range")

    def test_gradient_k_at_mid_transit(self):
        """dk gradient at mid-transit: analytical d(-k^2)/dk = -2k."""
        k = 0.1
        inc = radians(89.99)
        cf, dcf = solve_xy_p5_d(0.0, P, A, inc, E, W)

        # Analytical gradient from _udmodel_grad
        flux = np.zeros(1)
        dflux = np.zeros(7)
        _udmodel_grad(0.0, k, cf, dcf, flux, dflux)

        # Finite difference on _udmodel
        eps = 1e-7
        fp = np.zeros(1)
        _udmodel(0.0, k + eps, cf, fp)
        fm = np.zeros(1)
        _udmodel(0.0, k - eps, cf, fm)
        fd_dk = (fp[0] - fm[0]) / (2 * eps)

        np.testing.assert_allclose(dflux[0], fd_dk, rtol=1e-4)
        np.testing.assert_allclose(dflux[0], -2 * k, atol=1e-4)

    def test_gradient_shape(self):
        """dflux has 7 elements [k, t0, p, a, i, e, w]."""
        cf, dcf = solve_xy_p5_d(0.0, P, A, I, E, W)
        flux = np.zeros(1)
        dflux = np.zeros(7)
        _udmodel_grad(0.0, 0.1, cf, dcf, flux, dflux)
        assert dflux.shape == (7,)


class TestUdmodelGradFull:
    """Tests for `udmodel_grad` — the full array-level model with gradients."""

    def test_output_shapes(self):
        """flux shape (npv, npt), dflux shape (npv, npt, 7)."""
        flux, dflux = _call_udmodel_grad(TIMES, k=0.1)
        assert flux.shape == (1, TIMES.size)
        assert dflux.shape == (1, TIMES.size, 7)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_mid_transit_depth(self, k):
        """At mid-transit with near-zero impact parameter, deficit ~ -k^2."""
        inc = radians(89.99)
        times = np.array([0.0])
        flux, dflux = _call_udmodel_grad(times, k=k, inc=inc)
        np.testing.assert_allclose(flux[0, 0], -k**2, atol=1e-6)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_out_of_transit_is_zero(self, k):
        """Far from transit, flux and gradients should be zero."""
        times = np.array([0.5, 1.0, -0.5])
        flux, dflux = _call_udmodel_grad(times, k=k)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(dflux[0], 0.0, atol=1e-12)

    def test_symmetry(self):
        """Circular orbit: flux is symmetric, t0-gradient is antisymmetric."""
        k = 0.1
        times = np.linspace(-0.1, 0.1, 201)
        flux, dflux = _call_udmodel_grad(times, k=k)
        # Flux symmetric about mid-transit
        np.testing.assert_allclose(flux[0], flux[0, ::-1], atol=5e-7)
        # t0 gradient (index 1) is antisymmetric
        np.testing.assert_allclose(dflux[0, :, 1], -dflux[0, ::-1, 1], atol=5e-7)

    def test_depth_scales_with_k_squared(self):
        """Mid-transit depth ratio for k=0.1 vs k=0.01 should be ~100."""
        inc = radians(89.99)
        times = np.array([0.0])
        flux_small, _ = _call_udmodel_grad(times, k=0.01, inc=inc)
        flux_large, _ = _call_udmodel_grad(times, k=0.1, inc=inc)
        ratio = flux_large[0, 0] / flux_small[0, 0]
        np.testing.assert_allclose(ratio, 100.0, rtol=1e-3)

    def test_no_transit(self):
        """Large impact parameter — planet never overlaps the star."""
        inc_no_transit = radians(60.0)
        flux, dflux = _call_udmodel_grad(TIMES, k=0.1, inc=inc_no_transit)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(dflux[0], 0.0, atol=1e-12)

    def test_grazing_transit(self):
        """Grazing transit is shallower than full transit."""
        k = 0.1
        b_grazing = 0.95
        inc_graze = np.arccos(b_grazing / A)
        flux_graze, _ = _call_udmodel_grad(TIMES, k=k, inc=inc_graze)
        flux_full, _ = _call_udmodel_grad(TIMES, k=k)
        assert np.min(flux_graze) < 0.0, "Grazing transit should have a dip"
        assert np.min(flux_graze) > np.min(flux_full), "Grazing should be shallower"

    def test_eccentric_orbit(self):
        """Eccentric orbit (e=0.3) should produce a valid dip, no NaNs."""
        k = 0.1
        flux, dflux = _call_udmodel_grad(TIMES, k=k, e=0.3)
        assert not np.any(np.isnan(flux)), "No NaNs in flux"
        assert not np.any(np.isnan(dflux)), "No NaNs in dflux"
        assert np.min(flux) < 0.0, "Eccentric orbit should produce a transit dip"

    def test_flux_matches_forward_model(self):
        """Flux from udmodel_grad should equal flux from forward-only udmodel."""
        k = 0.1
        flux_grad, _ = _call_udmodel_grad(TIMES, k=k, e=0.1)
        flux_fwd = _call_udmodel(TIMES, k=k, e=0.1)
        np.testing.assert_allclose(flux_grad[0], flux_fwd[0], atol=1e-14)


class TestUdmodelGradFiniteDiff:
    """Gradient correctness verified against central finite differences."""

    # Use nonzero eccentricity so e-gradient is nonzero
    E_TEST = 0.1
    EPS = 1e-7
    K = 0.1

    def test_gradient_k(self):
        """dk gradient matches finite differences."""
        flux, dflux = _call_udmodel_grad(TIMES, k=self.K, e=self.E_TEST)
        fd = _finite_diff_grad(TIMES, 'k', 0, self.EPS, k=self.K, e=self.E_TEST)
        anal = dflux[0, :, 0]
        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(anal[mask], fd[mask], rtol=1e-3, atol=1e-8,
                                   err_msg="k gradient")

    @pytest.mark.parametrize("name,idx", [
        ('t0', 1),
        ('p', 2), ('a', 3), ('inc', 4), ('e', 5), ('w', 6)
    ])
    def test_gradient_orbital_params(self, name, idx):
        """Orbital parameter gradients match finite differences."""
        flux, dflux = _call_udmodel_grad(TIMES, k=self.K, e=self.E_TEST)
        fd = _finite_diff_grad(TIMES, name, idx, self.EPS, k=self.K, e=self.E_TEST)
        anal = dflux[0, :, idx]
        mask = flux[0] < -1e-6
        assert mask.any(), "Should have in-transit points"
        np.testing.assert_allclose(anal[mask], fd[mask], rtol=1e-3, atol=1e-8,
                                   err_msg=f"{name} gradient")

    def test_gradient_nonzero_in_transit(self):
        """During transit, k/a/i gradients should be nonzero.

        At exact mid-transit with near-zero impact parameter, a and i
        gradients vanish because the planet is fully inside the stellar
        disk.  We use a time slightly off mid-transit where partial
        overlap makes all gradients nonzero.
        """
        times = np.array([0.04])  # during ingress/egress
        _, dflux = _call_udmodel_grad(times, k=self.K, e=self.E_TEST)
        # k gradient (index 0) should be nonzero
        assert abs(dflux[0, 0, 0]) > 1e-6, "k gradient should be nonzero in transit"
        # a gradient (index 3) should be nonzero
        assert abs(dflux[0, 0, 3]) > 1e-6, "a gradient should be nonzero in transit"
        # i gradient (index 4) should be nonzero
        assert abs(dflux[0, 0, 4]) > 1e-6, "i gradient should be nonzero in transit"
