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
        ( 1 - K, LDC1, 0.99164875),
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
