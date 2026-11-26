#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s

K = 0.1
D = K ** 2
LDC0 = np.array([0.0, 0.0])
LDC1 = np.array([0.2, 0.1])


class TestEvalQuadZS:
    @pytest.mark.parametrize("z, ldc, expected", [
        (-2.0, LDC1, 1.0),
        ( 2.0, LDC1, 1.0),
        ( 0.2, LDC0, 1.0 - D),
        ( 0.2, LDC1, 0.98914137),
    ])
    def test_basic(self, z, ldc, expected):
        assert_almost_equal(eval_quad_z_s(z, K, ldc), expected)

    @pytest.mark.parametrize("z, ldc, expected", [
        (-0.0,   LDC1, 1.0),
        ( 0.0,   LDC0, 1.0 - D),
        ( 0.0,   LDC1, 0.98909638),
        ( K,     LDC0, 1.0 - D),
        ( K,     LDC1, 0.98910745),
        ( 1 - K, LDC0, 1.0 - D),
        ( 1 - K, LDC1, 0.99075967),
        ( 1.0,   LDC0, 0.99510612),
        ( 1.0,   LDC1, 0.99573423),
        ( 1 + K, LDC0, 1.0),
        ( 1 + K, LDC1, 1.0),
    ])
    def test_standard_edges(self, z, ldc, expected):
        assert_almost_equal(eval_quad_z_s(z, K, ldc), expected)

    @pytest.mark.parametrize("z, k, expected", [
        (-0.0, 1.0, 1.0),
        ( 0.0, 1.0, 0.0),
        ( 0.0, 1.1, 0.0),
        ( 1.0, 2.0, 0.0),
    ])
    def test_large_radius_ratio(self, z, k, expected):
        assert_almost_equal(eval_quad_z_s(z, k, LDC1), expected)


def _direct_flux(z, k, u):
    """The flux by direct integration over the stellar disk, for a planet at distance z."""
    from numpy import pi, sqrt, arccos, sin, cos
    from scipy.integrate import quad

    def theta(r):
        if r <= k - z:
            return 2 * pi
        if r < z - k or r > z + k:
            return 0.0
        return 2 * arccos(min(1.0, max(-1.0, (r * r + z * z - k * k) / (2 * r * z))))

    def intensity(mu):
        return 1 - u[0] * (1 - mu) - u[1] * (1 - mu) ** 2

    blocked = 0.0
    lo, hi = max(0.0, z - k), min(1.0, z + k)
    if z < k:
        blocked += 2 * pi * quad(lambda mu: mu * intensity(mu), sqrt(1 - (k - z) ** 2), 1, epsabs=0, epsrel=1e-13)[0]
        lo = k - z
    # r = lo + (hi - lo) sin(t)**2 makes the square roots at both ends regular
    def f(t):
        r = lo + (hi - lo) * sin(t) ** 2
        return theta(r) * r * intensity(sqrt(max(1 - r * r, 0.0))) * 2 * (hi - lo) * sin(t) * cos(t)
    blocked += quad(f, 0, pi / 2, epsabs=0, epsrel=1e-13, limit=500)[0]
    return 1 - blocked / (pi * (1 - u[0] / 3 - u[1] / 6))


class TestSingularPoints:
    """The closed forms used at the singular points of the model must not be visible around them."""

    OFFSETS = (-1e-4, -3e-5, -1e-6, -1e-8, -3e-9, -1e-11, 0.0, 1e-11, 3e-9, 1e-8, 1e-6, 3e-5, 1e-4)

    @pytest.mark.parametrize("k", [0.1, 0.3, 0.7, 0.9])
    def test_edge_at_the_disk_centre(self, k):
        for d in self.OFFSETS:
            assert abs(eval_quad_z_s(k + d, k, LDC1) - _direct_flux(k + d, k, LDC1)) < 2e-8, d

    @pytest.mark.parametrize("k", [0.1, 0.3, 0.7, 0.9])
    def test_limb_contact_from_the_inside(self, k):
        for d in self.OFFSETS:
            assert abs(eval_quad_z_s(1 - k + d, k, LDC1) - _direct_flux(1 - k + d, k, LDC1)) < 2e-8, d

    @pytest.mark.parametrize("dk", [-1e-4, -5e-5, -1e-6, -1e-9, -1e-11, 0.0, 1e-11, 1e-9, 1e-6, 5e-5, 1e-4])
    def test_radius_ratios_next_to_one_half(self, dk):
        # At k = 1/2 the planet's edge reaches the disk centre and the limb at the same z.
        k = 0.5 + dk
        for z in (0.0, 0.2, k - 1e-9, k, k + 1e-9, 1 - k, 0.5, 0.8, 1.2):
            assert abs(eval_quad_z_s(z, k, LDC1) - _direct_flux(z, k, LDC1)) < 2e-8, z
