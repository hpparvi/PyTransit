from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from pytransit.orbits_f import orbits as of

class TestCircularZ(unittest.TestCase):
    """Test the routines to calculate the normalized projected distance (z) assuming e=0 and w=0.

    NOTE:
      The normalized projected distance should be negative during an eclipse.
    """
    def setUp(self):
        self.t  = [0, 0.125, 0.25, 0.375, 0.5, 0.75]
        self.t0 = 0
        self.p  = 1
        self.a  = 5
        self.i  = 0.5*np.pi
        self.zref = np.array([0.0 , 3.535533, 5.0 , -3.535533, -0.0, -5.0])

    def test_circular(self):
        z = of.z_circular(self.t, self.t0, self.p, self.a, self.i, nthreads=1)
        npt.assert_array_almost_equal(z, self.zref)

    def test_ip(self):
        z = of.z_eccentric_ip(self.t, self.t0, self.p, self.a, self.i, 0, 0, nthreads=1, update=True)
        npt.assert_array_almost_equal(z, self.zref, decimal=4)

    def test_iter(self):
        z = of.z_eccentric_iter(self.t, self.t0, self.p, self.a, self.i, 0, 0, nthreads=1)
        npt.assert_array_almost_equal(z, self.zref)

    def test_newton(self):
        z = of.z_eccentric_newton(self.t, self.t0, self.p, self.a, self.i, 0, 0, nthreads=1)
        npt.assert_array_almost_equal(z, self.zref)

    def test_ps3(self):
        z = of.z_eccentric_ps3(self.t, self.t0, self.p, self.a, self.i, 0, 0, nthreads=1)
        npt.assert_array_almost_equal(z, self.zref)

    def test_ps5(self):
        z = of.z_eccentric_ps5(self.t, self.t0, self.p, self.a, self.i, 0, 0, nthreads=1)
        npt.assert_array_almost_equal(z, self.zref)

if __name__ == '__main__':
    unittest.main()
