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
from math import pi
from numpy import array, copysign, inf
from numpy.testing import assert_almost_equal

from pytransit.contamination import true_radius_ratio, apparent_radius_ratio


class TestContamination(unittest.TestCase):
    """Test the routines to calculate the normalized projected distance (z) assuming zero eccentricity.
    """
    def setUp(self):
        pass

    def test_true_radius_ratio(self):
        assert_almost_equal(true_radius_ratio(0.1, 0.0), 0.1)
        assert_almost_equal(true_radius_ratio(0.1, 1.0), inf)

    def test_apparent_radius_ratio(self):
        assert_almost_equal(apparent_radius_ratio(0.1, 0.0), 0.1)
        assert_almost_equal(apparent_radius_ratio(0.1, 1.0), 0.0)


if __name__ == '__main__':
    unittest.main()
