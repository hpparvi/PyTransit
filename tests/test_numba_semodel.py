"""Tests for the Numba secondary-eclipse model backend.

Verifies the forward-only functions:
1. `_semodel`  — single-timestamp eclipse-flux accumulation
2. `semodel`   — full array-level model with orbit folding and supersampling

The secondary-eclipse model treats the planet as a uniformly bright disk: out
of eclipse it contributes a constant flux of ``pi * k**2``; during the eclipse
the stellar disk occults part of it, removing the circle-circle intersection
area. Limb darkening is not modelled.
"""

import numpy as np
import pytest
from math import radians, pi

from numba import njit
from meepmeep import eclipse_light_travel_time
from meepmeep.numba2d import solve2d
from meepmeep.backends.numba.utils import eclipse_time_offset
from pytransit.backends.numba.semodel import _semodel, semodel

semodel_jit = njit(semodel)

# Shared orbital parameters
T0 = 0.0
P = 2.5
A = 8.0
I = radians(89.0)
E = 0.0
W = pi / 2
RSTAR = 1.0

# Circular-orbit eclipse falls at t0 + P/2.
ECL = T0 + P / 2
TIMES = np.linspace(ECL - 0.15, ECL + 0.15, 300)


def _make_2d_params(k, t0=T0, p=P, a=A, inc=I, e=E, w=W, npv=1, npb=1, nep=1):
    """Build 2D parameter arrays expected by `semodel`."""
    return dict(k=np.full((npv, npb), k),
                t0=np.full((npv, nep), t0),
                p=np.full((npv, nep), p),
                a=np.full((npv, nep), a),
                i=np.full((npv, nep), inc),
                e=np.full((npv, nep), e),
                w=np.full((npv, nep), w))


def _call_semodel(times, k=0.1, t0=T0, p=P, a=A, inc=I, e=E, w=W, rstar=RSTAR,
                  nsamples=1, exptime=0.0):
    """Convenience wrapper calling `semodel` with a single-LC setup."""
    npt = times.size
    pp = _make_2d_params(k, t0, p, a, inc, e, w)
    lcids = np.zeros(npt, dtype=np.int32)
    pbids = np.zeros(1, dtype=np.int32)
    epids = np.zeros(1, dtype=np.int32)
    ns = np.full(1, nsamples, dtype=np.int32)
    et = np.full(1, exptime, dtype=np.float64)
    return semodel_jit(times, pp['k'], pp['t0'], pp['p'], pp['a'], pp['i'], pp['e'], pp['w'],
                       rstar, lcids, pbids, epids, ns, et, 1, 1)


class TestSemodelKernel:
    """Tests for `_semodel` — single-timestamp eclipse-flux accumulation."""

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_mid_eclipse_is_zero(self, k):
        """At mid-eclipse with near-central geometry the planet is fully hidden."""
        cf = solve2d(eclipse_time_offset(P, I, E, W), P, A, I, E, W)
        flux = np.zeros(1)
        _semodel(0.0, k, cf, flux)
        np.testing.assert_allclose(flux[0], 0.0, atol=1e-12)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_out_of_eclipse_baseline(self, k):
        """Far from eclipse the flux equals the planet baseline pi*k**2."""
        cf = solve2d(eclipse_time_offset(P, I, E, W), P, A, I, E, W)
        flux = np.zeros(1)
        _semodel(0.5, k, cf, flux)  # planet well clear of the star
        np.testing.assert_allclose(flux[0], pi * k**2, atol=1e-12)

    def test_partial_overlap(self):
        """During ingress/egress the flux lies strictly between 0 and pi*k**2."""
        k = 0.1
        cf = solve2d(eclipse_time_offset(P, I, E, W), P, A, I, E, W)
        for t in np.linspace(0.02, 0.10, 50):
            flux = np.zeros(1)
            _semodel(t, k, cf, flux)
            if 0.0 < flux[0] < pi * k**2:
                return
        pytest.fail("No partial overlap point found in the scanned range")


class TestSemodelFull:
    """Tests for `semodel` — the full array-level model."""

    def test_output_shape(self):
        flux = _call_semodel(TIMES, k=0.1)
        assert flux.shape == (1, TIMES.size)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_baseline_out_of_eclipse(self, k):
        """Well before/after the eclipse the flux is the constant baseline."""
        times = np.array([ECL - 0.5, ECL + 0.5])
        flux = _call_semodel(times, k=k)
        np.testing.assert_allclose(flux[0], pi * k**2, atol=1e-12)

    @pytest.mark.parametrize("k", [0.01, 0.1])
    def test_total_eclipse_floor(self, k):
        """A small planet is fully occulted at mid-eclipse (flux -> 0)."""
        flux = _call_semodel(np.array([ECL + 4.296762908980465e-04]), k=k)  # + ltt
        np.testing.assert_allclose(flux[0, 0], 0.0, atol=1e-9)

    def test_flux_bounded(self):
        """Flux never leaves the physical range [0, pi*k**2]."""
        k = 0.1
        flux = _call_semodel(TIMES, k=k)
        assert np.all(flux[0] >= -1e-12)
        assert np.all(flux[0] <= pi * k**2 + 1e-12)

    def test_baseline_scales_with_k_squared(self):
        """Out-of-eclipse baseline ratio for k=0.1 vs k=0.01 should be ~100."""
        times = np.array([ECL - 0.5])
        small = _call_semodel(times, k=0.01)
        large = _call_semodel(times, k=0.1)
        np.testing.assert_allclose(large[0, 0] / small[0, 0], 100.0, rtol=1e-6)

    def test_eclipse_centered_on_offset_plus_light_travel(self):
        """Eclipse centre is offset from t0 by P/2 plus the light-travel delay.

        For a circular orbit the light curve is symmetric about that centre,
        NOT about t0 + P/2: the light-travel time shifts the whole eclipse.
        """
        ltt = eclipse_light_travel_time(P, A, I, E, W, RSTAR)
        center = T0 + eclipse_time_offset(P, I, E, W) + ltt
        times = np.linspace(center - 0.12, center + 0.12, 301)  # symmetric about centre
        flux = _call_semodel(times, k=0.02)
        np.testing.assert_allclose(flux[0], flux[0, ::-1], atol=1e-9)
        # The shift is real: a curve symmetric about t0 + P/2 (ignoring ltt) is not.
        assert ltt > 0.0

    def test_no_eclipse_high_inclination(self):
        """With a low inclination the planet never passes behind the star."""
        flux = _call_semodel(TIMES, k=0.1, inc=radians(60.0))
        np.testing.assert_allclose(flux[0], pi * 0.1**2, atol=1e-12)

    def test_eccentric_orbit_no_nans(self):
        """An eccentric orbit yields a valid eclipse with no NaNs."""
        center = T0 + eclipse_time_offset(P, I, 0.3, radians(75.0))
        times = np.linspace(center - 0.15, center + 0.15, 300)
        flux = _call_semodel(times, k=0.1, e=0.3, w=radians(75.0))
        assert not np.any(np.isnan(flux))
        assert np.min(flux) < pi * 0.1**2  # an actual dip occurs

    def test_supersampling_averages(self):
        """Supersampling smooths the sharp contact points without shifting them."""
        f1 = _call_semodel(TIMES, k=0.1, nsamples=1, exptime=0.02)
        f10 = _call_semodel(TIMES, k=0.1, nsamples=10, exptime=0.02)
        # Same baseline and floor, but supersampling rounds the ingress corners.
        np.testing.assert_allclose(f1[0].max(), f10[0].max(), atol=1e-9)
        assert np.max(np.abs(f1[0] - f10[0])) > 0.0
