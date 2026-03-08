"""Tests for the Numba uniform-disk transit model backend.

Verifies the low-level functions:
1. `_udmodel` — single-timestamp flux deficit computation
2. `udmodel` — full array-level model with orbit folding and supersampling
"""

import numpy as np
import pytest
from math import radians, pi

from numba import njit
from meepmeep.backends.numba.ts2d import solve_xy_p5
from pytransit.backends.numba.udmodel import _udmodel, udmodel

udmodel_jit = njit(udmodel)

# Shared orbital parameters
T0 = 0.0
P = 2.5
A = 8.0
I = radians(87.0)
E = 0.0
W = pi / 2
TIMES = np.linspace(-0.15, 0.15, 300)


def _make_2d_params(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npv=1, npb=1, ntc=1, nor=1):
    """Build 2D parameter arrays expected by `udmodel`.

    Parameters
    ----------
    k : float
        Planet-to-star radius ratio.
    npv : int
        Number of parameter vectors.
    npb : int
        Number of passbands.
    ntc : int
        Number of transit centres.
    nor : int
        Number of orbits (epochs).

    Returns
    -------
    dict with 2D arrays for k, t0, p, a, i, e, w.
    """
    k2 = np.full((npv, npb), k)
    t02 = np.full((npv, nor), t0)
    p2 = np.full((npv, nor), p)
    a2 = np.full((npv, nor), a)
    i2 = np.full((npv, nor), inc)
    e2 = np.full((npv, nor), e)
    w2 = np.full((npv, nor), w)
    return dict(k=k2, t0=t02, p=p2, a=a2, i=i2, e=e2, w=w2)


def _call_udmodel(times, k, t0=T0, p=P, a=A, inc=I, e=E, w=W):
    """Convenience wrapper calling `udmodel` with default single-LC setup."""
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
                       1, 1)


class TestUdmodel:
    """Tests for `_udmodel` — single-timestamp flux deficit."""

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_mid_transit_depth(self, k):
        """At mid-transit with near-zero impact parameter, deficit ≈ -k²."""
        inc = radians(89.99)
        cf = solve_xy_p5(0.0, P, A, inc, E, W)
        flux = np.zeros(1)
        _udmodel(0.0, k, cf, flux)
        np.testing.assert_allclose(flux[0], -k**2, atol=1e-6)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_out_of_transit_is_zero(self, k):
        """Far from transit, flux deficit should remain zero."""
        cf = solve_xy_p5(0.0, P, A, I, E, W)
        flux = np.zeros(1)
        _udmodel(0.5, k, cf, flux)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)

    def test_partial_overlap(self):
        """During ingress/egress, deficit should be between -k² and 0."""
        k = 0.1
        cf = solve_xy_p5(0.0, P, A, I, E, W)
        # Find a time near the limb by evaluating several points
        for t in np.linspace(0.02, 0.10, 50):
            flux = np.zeros(1)
            _udmodel(t, k, cf, flux)
            if -k**2 < flux[0] < 0.0:
                # Found a partial overlap point
                assert -k**2 < flux[0] < 0.0
                return
        pytest.fail("No partial overlap point found in the scanned range")


class TestUdmodelFull:
    """Tests for `udmodel` — the full array-level model."""

    def test_output_shape(self):
        """Output should have shape (npv, npt)."""
        flux = _call_udmodel(TIMES, k=0.1)
        assert flux.shape == (1, TIMES.size)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_mid_transit_depth(self, k):
        """At mid-transit with near-zero impact parameter, deficit ≈ -k²."""
        inc = radians(89.99)
        times = np.array([0.0])
        flux = _call_udmodel(times, k=k, inc=inc)
        np.testing.assert_allclose(flux[0, 0], -k**2, atol=1e-6)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_out_of_transit_is_zero(self, k):
        """Far from transit, flux should be zero."""
        times = np.array([0.5, 1.0, -0.5])
        flux = _call_udmodel(times, k=k)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)

    def test_symmetry(self):
        """Circular orbit produces symmetric light curve about t0."""
        k = 0.1
        times = np.linspace(-0.1, 0.1, 201)
        flux = _call_udmodel(times, k=k)
        np.testing.assert_allclose(flux[0], flux[0, ::-1], atol=5e-7)

    def test_depth_scales_with_k_squared(self):
        """Mid-transit depth ratio for k=0.1 vs k=0.01 should be ~100."""
        inc = radians(89.99)
        times = np.array([0.0])
        flux_small = _call_udmodel(times, k=0.01, inc=inc)
        flux_large = _call_udmodel(times, k=0.1, inc=inc)
        ratio = flux_large[0, 0] / flux_small[0, 0]
        np.testing.assert_allclose(ratio, 100.0, rtol=1e-3)

    def test_no_transit(self):
        """Large impact parameter — planet never overlaps the star."""
        inc_no_transit = radians(60.0)
        flux = _call_udmodel(TIMES, k=0.1, inc=inc_no_transit)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)

    def test_grazing_transit(self):
        """Grazing transit is shallower than full transit."""
        k = 0.1
        b_grazing = 0.95
        inc_graze = np.arccos(b_grazing / A)
        flux_graze = _call_udmodel(TIMES, k=k, inc=inc_graze)
        flux_full = _call_udmodel(TIMES, k=k)
        assert np.min(flux_graze) < 0.0, "Grazing transit should have a dip"
        assert np.min(flux_graze) > np.min(flux_full), "Grazing should be shallower"

    def test_eccentric_orbit(self):
        """Eccentric orbit (e=0.3) should still produce a valid dip."""
        k = 0.1
        flux = _call_udmodel(TIMES, k=k, e=0.3)
        assert np.min(flux) < 0.0, "Eccentric orbit should produce a transit dip"
        assert flux[0, 0] == 0.0 or abs(flux[0, 0]) < 1e-10, "Edges should be near zero"

    def test_small_a_constant_depth(self):
        """When a < 1, planet is always in front of star — constant -k² deficit."""
        k = 0.1
        flux = _call_udmodel(TIMES, k=k, a=0.5)
        np.testing.assert_allclose(flux[0], -k**2, atol=1e-6)
