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

from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s, eval_quad_z_v

class TestQuadraticModelNB(unittest.TestCase):

    def setUp(self) -> None:
        self.k = k = 0.1
        self.d = d = self.k**2
        self.ldc1d0 = array([0., 0.])
        self.ldc1d1 = array([0.2, 0.1])
        self.ldc2d1 = array([[0.2, 0.1]])
        self.z_edge = array([-0.0, 0.0, k, 1.0-k, 1.0, 1.0+k])
        self.f_edge = array([1.0, 1-d, 1-d, 1-d, 0.9951061298, 1.0])

    def test_quadratic_z_s_basic_cases(self):
        assert_almost_equal(eval_quad_z_s(        -2.0, self.k, self.ldc1d1), 1.0)
        assert_almost_equal(eval_quad_z_s(         2.0, self.k, self.ldc1d1), 1.0)
        assert_almost_equal(eval_quad_z_s(         0.2, self.k, self.ldc1d0), 1.0-self.d)
        assert_almost_equal(eval_quad_z_s(         0.2, self.k, self.ldc1d1), 0.98914137)

    def test_quadratic_z_s_edge_cases(self):
        # Standard edge cases
        # -------------------
        assert_almost_equal(eval_quad_z_s(        -0.0, self.k, self.ldc1d1), 1.0)
        assert_almost_equal(eval_quad_z_s(         0.0, self.k, self.ldc1d0), 1.0-self.d)
        assert_almost_equal(eval_quad_z_s(         0.0, self.k, self.ldc1d1), 0.98909638)
        assert_almost_equal(eval_quad_z_s(      self.k, self.k, self.ldc1d0), 1.0-self.d)
        assert_almost_equal(eval_quad_z_s(      self.k, self.k, self.ldc1d1), 0.98910745)
        assert_almost_equal(eval_quad_z_s(  1 - self.k, self.k, self.ldc1d0), 1.0-self.d)
        assert_almost_equal(eval_quad_z_s(  1 - self.k, self.k, self.ldc1d1), 0.99164875)
        assert_almost_equal(eval_quad_z_s(         1.0, self.k, self.ldc1d0), 0.99510612)
        assert_almost_equal(eval_quad_z_s(         1.0, self.k, self.ldc1d1), 0.99573423)
        assert_almost_equal(eval_quad_z_s(  1 + self.k, self.k, self.ldc1d0), 1.0)
        assert_almost_equal(eval_quad_z_s(  1 + self.k, self.k, self.ldc1d1), 1.0)

        # Radius ratio larger or equal to unity
        # -------------------------------------
        assert_almost_equal(eval_quad_z_s(-0.0, 1.0, self.ldc1d1), 1.0)
        assert_almost_equal(eval_quad_z_s( 0.0, 1.0, self.ldc1d1), 0.0)
        assert_almost_equal(eval_quad_z_s( 0.0, 1.1, self.ldc1d1), 0.0)
        assert_almost_equal(eval_quad_z_s( 1.0, 2.0, self.ldc1d1), 0.0)


if __name__ == '__main__':
    unittest.main()