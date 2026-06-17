#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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

from math import pi
from numpy import array, copysign
from numpy.testing import assert_almost_equal

from pytransit.orbits.orbits_py import z_circular, z_newton_p, z_newton_s, z_iter_p, ta_ip_calculate_table, z_ip_s

T0, P, A, I = 0.0, 1.0, 3.0, 0.5 * pi

TIMES = array([0.0, 0.25, 0.4, 0.5, 0.6, 0.75 + 1e-8, 1.0])
WS = [0.0, 0.5 * pi, pi, 1.5 * pi, 2 * pi]
Z_TRUTH = array([0., 3., -1.7634, -0., -1.7634, 3., 0.])
S_TRUTH = array([1., 1., -1., -1., -1, 1., 1.])


class TestCircularZ:
    """Test the routines to calculate the normalized projected distance (z) assuming zero eccentricity.
    """

    def test_circular(self):
        for w in WS:
            pv = array([T0, P, A, I, 0.0, w])
            z = z_circular(TIMES, pv)
            assert_almost_equal(z, Z_TRUTH, 4)
            assert_almost_equal(copysign(1., z), S_TRUTH)

    def test_newton(self):
        for w in WS:
            pv = array([T0, P, A, I, 0.0, w])
            z = z_newton_p(TIMES, pv)
            assert_almost_equal(z, Z_TRUTH, 4)
            assert_almost_equal(copysign(1., z), S_TRUTH)

        for w in WS:
            pv = array([T0, P, A, I, 0.0, w])
            z = array([z_newton_s(t, pv) for t in TIMES])
            assert_almost_equal(z, Z_TRUTH, 4)
            assert_almost_equal(copysign(1., z), S_TRUTH)

    def test_iter(self):
        for w in WS:
            pv = array([T0, P, A, I, 0.0, w])
            z = z_iter_p(TIMES, pv)
            assert_almost_equal(z, Z_TRUTH, 4)
            assert_almost_equal(copysign(1., z), S_TRUTH)

    def test_interpolated(self):
        for w in WS:
            pv = array([T0, P, A, I, 0.0, w])
            tae, es, ms = ta_ip_calculate_table()
            z = array([z_ip_s(t, T0, P, A, I, 0.0, 0.0, es, ms, tae) for t in TIMES])
            assert_almost_equal(z, Z_TRUTH, 4)
            assert_almost_equal(copysign(1., z), S_TRUTH)
