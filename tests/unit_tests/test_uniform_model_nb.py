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
from numpy import array
from numpy.testing import assert_almost_equal

from pytransit.models.numba.ma_uniform_nb import uniform_z_s, uniform_z_v

class TestUniformModelNB(unittest.TestCase):

    def setUp(self) -> None:
        self.k = k = 0.1
        self.d = d = self.k**2
        self.z = array([0.0, 0.0+self.k, 0.5, 1.0-self.k, 1.0, 1.0+self.k, 2.0])
        self.z_edge = array([-0.0, 0.0, k, 1.0-k, 1.0, 1.0+k])
        self.f_edge = array([1.0, 1-d, 1-d, 1-d, 0.9951061298, 1.0])
        self.f = array([1-d, 1-d, 1-d, 1-d, 23, 23, 1.0, 1.0])

    def test_uniform_z_s_basic_cases(self):
        # Primary transit
        # ---------------
        assert_almost_equal(uniform_z_s(        -2.0, self.k, 1.0), 1.0)
        assert_almost_equal(uniform_z_s(         2.0, self.k, 1.0), 1.0)
        assert_almost_equal(uniform_z_s(         0.1, self.k, 1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(         0.3, self.k, 1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(        -0.1, self.k, 1.0), 1.0)
        assert_almost_equal(uniform_z_s(        -0.3, self.k, 1.0), 1.0)

        # Secondary eclipse
        # -----------------
        assert_almost_equal(uniform_z_s(        -2.0, self.k, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(         2.0, self.k, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(         0.1, self.k, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(         0.3, self.k, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(        -0.1, self.k, -1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(        -0.3, self.k, -1.0), 1.0-self.d)

    def test_uniform_z_s_edge_cases(self):
        # Standard edge cases, primary transit
        # ------------------------------------
        assert_almost_equal(uniform_z_s(        -0.0, self.k, 1.0), 1.0)
        assert_almost_equal(uniform_z_s(         0.0, self.k, 1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(      self.k, self.k, 1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(1.0 - self.k, self.k, 1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(         1.0, self.k, 1.0), 0.9951061298)
        assert_almost_equal(uniform_z_s(1.0 + self.k, self.k, 1.0), 1.0)

        # Standard edge cases, secondary eclipse
        # --------------------------------------
        assert_almost_equal(uniform_z_s(         -0.0, self.k, -1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(          0.0, self.k, -1.0), 1.0)
        assert_almost_equal(uniform_z_s(      -self.k, self.k, -1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s( self.k - 1.0, self.k, -1.0), 1.0-self.d)
        assert_almost_equal(uniform_z_s(         -1.0, self.k, -1.0), 0.9951061298)
        assert_almost_equal(uniform_z_s(-1.0 - self.k, self.k, -1.0), 1.0)

        # Radius ratio larger or equal to unity
        # -------------------------------------
        assert_almost_equal(uniform_z_s(-0.0, 1.0, 1.0), 1.0)
        assert_almost_equal(uniform_z_s( 0.0, 1.0, 1.0), 0.0)
        assert_almost_equal(uniform_z_s( 0.0, 1.1, 1.0), 0.0)
        assert_almost_equal(uniform_z_s( 1.0, 2.0, 1.0), 0.0)

    def test_uniform_z_v_edge_cases(self):
        # Standard edge cases, primary transit
        # ------------------------------------
        assert_almost_equal(uniform_z_v(self.z_edge, self.k, 1.0), self.f_edge)

        # Standard edge cases, secondary eclipse
        # --------------------------------------
        assert_almost_equal(uniform_z_v(-self.z_edge, self.k, -1.0), self.f_edge)


if __name__ == '__main__':
    unittest.main()