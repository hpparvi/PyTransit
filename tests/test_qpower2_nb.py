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

from pytransit.models.numba.qpower2_nb import qpower2_z_s

K = 0.1
LDC = array([0.23, 0.12])


class TestQPower2NB:

    def test_qpower2_z_s_basic_cases(self):
        # Primary transit
        # ---------------
        assert_almost_equal(qpower2_z_s(-2.0, K, LDC), 1.0)
        assert_almost_equal(qpower2_z_s( 2.0, K, LDC), 1.0)
        assert_almost_equal(qpower2_z_s( 0.1, K, LDC), 0.98987021)
        assert_almost_equal(qpower2_z_s( 0.3, K, LDC), 0.98988207)
        assert_almost_equal(qpower2_z_s(-0.1, K, LDC), 1.0)
        assert_almost_equal(qpower2_z_s(-0.3, K, LDC), 1.0)

    def test_qpower2_z_s_edge_cases(self):
        # Standard edge cases, primary transit
        # ------------------------------------
        assert_almost_equal(qpower2_z_s(      -0.0, K, LDC), 1.0)
        assert_almost_equal(qpower2_z_s(       0.0, K, LDC), 0.98986879)
        assert_almost_equal(qpower2_z_s(         K, K, LDC), 0.98987021)
        assert_almost_equal(qpower2_z_s(   1.0 - K, K, LDC), 0.99010578)
        assert_almost_equal(qpower2_z_s(       1.0, K, LDC), 0.99521070)
        assert_almost_equal(qpower2_z_s(   1.0 + K, K, LDC), 1.0)

        # Radius ratio larger or equal to unity
        # -------------------------------------
        assert_almost_equal(qpower2_z_s(-0.0, 1.0, LDC), 1.0)
        assert_almost_equal(qpower2_z_s( 0.0, 1.0, LDC), 0.0)
