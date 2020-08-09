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

import unittest
from numpy import array, isnan
from numpy.testing import assert_almost_equal, assert_raises

from pytransit.models.numba.qpower2_nb import qpower2_z_s

class TestUniformModelNB(unittest.TestCase):

    def setUp(self) -> None:
        self.k = k = 0.1
        self.d = d = self.k**2
        self.ldc = array([0.23, 0.12])
        self.z_edge = array([-0.0, 0.0, k, 1.0-k, 1.0, 1.0+k])
        self.f_edge = array([1.0, 1-d, 1-d, 1-d, 0.9951061298, 1.0])

    def test_uniform_z_s_basic_cases(self):
        # Primary transit
        # ---------------
        assert_almost_equal(qpower2_z_s(        -2.0, self.k, self.ldc), 1.0)
        assert_almost_equal(qpower2_z_s(         2.0, self.k, self.ldc), 1.0)
        assert_almost_equal(qpower2_z_s(         0.1, self.k, self.ldc), 0.98987021)
        assert_almost_equal(qpower2_z_s(         0.3, self.k, self.ldc), 0.98988207)
        assert_almost_equal(qpower2_z_s(        -0.1, self.k, self.ldc), 1.0)
        assert_almost_equal(qpower2_z_s(        -0.3, self.k, self.ldc), 1.0)

    def test_uniform_z_s_edge_cases(self):
        # Standard edge cases, primary transit
        # ------------------------------------
        assert_almost_equal(qpower2_z_s(        -0.0, self.k, self.ldc), 1.0)
        assert_almost_equal(qpower2_z_s(         0.0, self.k, self.ldc), 0.98986879)
        assert_almost_equal(qpower2_z_s(      self.k, self.k, self.ldc), 0.98987021)
        assert_almost_equal(qpower2_z_s(1.0 - self.k, self.k, self.ldc), 0.99010578)
        assert_almost_equal(qpower2_z_s(         1.0, self.k, self.ldc), 0.99521070)
        assert_almost_equal(qpower2_z_s(1.0 + self.k, self.k, self.ldc), 1.0)

        # Radius ratio larger or equal to unity
        # -------------------------------------
        assert_almost_equal(qpower2_z_s(-0.0, 1.0, self.ldc), 1.0)
        assert_almost_equal(qpower2_z_s( 0.0, 1.0, self.ldc), 0.0)
        assert isnan(qpower2_z_s( 0.0, 1.1, self.ldc))


if __name__ == '__main__':
    unittest.main()