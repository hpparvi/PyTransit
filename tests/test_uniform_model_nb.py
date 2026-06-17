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

from numpy import array
from numpy.testing import assert_almost_equal

from pytransit.models.numba.ma_uniform_nb import uniform_z_s, uniform_z_v

K = 0.1
D = K ** 2
Z_EDGE = array([-0.0, 0.0, K, 1.0 - K, 1.0, 1.0 + K])
F_EDGE = array([1.0, 1 - D, 1 - D, 1 - D, 0.9951061298, 1.0])


class TestUniformModelNB:

    def test_uniform_z_s_basic_cases(self):
        # Primary transit
        # ---------------
        assert_almost_equal(uniform_z_s(-2.0, K, 1.0), 1.0)
        assert_almost_equal(uniform_z_s( 2.0, K, 1.0), 1.0)
        assert_almost_equal(uniform_z_s( 0.1, K, 1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s( 0.3, K, 1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(-0.1, K, 1.0), 1.0)
        assert_almost_equal(uniform_z_s(-0.3, K, 1.0), 1.0)

        # Secondary eclipse
        # -----------------
        assert_almost_equal(uniform_z_s(-2.0, K, -1.0), 1.0)
        assert_almost_equal(uniform_z_s( 2.0, K, -1.0), 1.0)
        assert_almost_equal(uniform_z_s( 0.1, K, -1.0), 1.0)
        assert_almost_equal(uniform_z_s( 0.3, K, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(-0.1, K, -1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(-0.3, K, -1.0), 1.0 - D)

    def test_uniform_z_s_edge_cases(self):
        # Standard edge cases, primary transit
        # ------------------------------------
        assert_almost_equal(uniform_z_s(   -0.0, K, 1.0), 1.0)
        assert_almost_equal(uniform_z_s(    0.0, K, 1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(      K, K, 1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(1.0 - K, K, 1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(    1.0, K, 1.0), 0.9951061298)
        assert_almost_equal(uniform_z_s(1.0 + K, K, 1.0), 1.0)

        # Standard edge cases, secondary eclipse
        # --------------------------------------
        assert_almost_equal(uniform_z_s(    -0.0, K, -1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(     0.0, K, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(      -K, K, -1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s( K - 1.0, K, -1.0), 1.0 - D)
        assert_almost_equal(uniform_z_s(    -1.0, K, -1.0), 0.9951061298)
        assert_almost_equal(uniform_z_s(-1.0 - K, K, -1.0), 1.0)

        # Radius ratio larger or equal to unity
        # -------------------------------------
        assert_almost_equal(uniform_z_s(-0.0, 1.0, 1.0), 1.0)
        assert_almost_equal(uniform_z_s( 0.0, 1.0, 1.0), 0.0)
        assert_almost_equal(uniform_z_s( 0.0, 1.1, 1.0), 0.0)
        assert_almost_equal(uniform_z_s( 1.0, 2.0, 1.0), 0.0)

    def test_uniform_z_v_edge_cases(self):
        # Standard edge cases, primary transit
        # ------------------------------------
        assert_almost_equal(uniform_z_v(Z_EDGE, K, 1.0), F_EDGE)

        # Standard edge cases, secondary eclipse
        # --------------------------------------
        assert_almost_equal(uniform_z_v(-Z_EDGE, K, -1.0), F_EDGE)
