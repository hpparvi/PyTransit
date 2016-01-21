from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt
from math import pi

from pytransit.orbits_f import orbits as of

class TestCircularZ(unittest.TestCase):
    """Test the routines to calculate the normalized projected distance (z) assuming zero eccentricity.
    """
    def setUp(self):
        self.t  = [0, 0.125, 0.25, 0.375, 0.5, 0.75]
        self.t0 = 0
        self.p  = 1
        self.a  = 5
        self.i  = 0.5*np.pi
        self.zref = np.array([0.0 , 3.535533, 5.0 , -3.535533, -0.0, -5.0])
        self.ws = [0, 0.5*pi, pi, 1.5*pi, 2*pi, 0.012, 0.24, -0.53]
        
    def test_circular(self):
        z = of.z_circular(self.t, self.t0, self.p, self.a, self.i, nth=1)
        npt.assert_array_almost_equal(z, self.zref)

    def test_ip(self):
        for w in self.ws:
            z = of.z_eccentric_ip(self.t, self.t0, self.p, self.a, self.i, 0, w, nth=1, update=True)
            npt.assert_array_almost_equal(z, self.zref, decimal=4)
                
    def test_iter(self):
        for w in self.ws:
            z = of.z_eccentric_iter(self.t, self.t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.zref)
 
    def test_newton(self):
        for w in self.ws:
            z = of.z_eccentric_newton(self.t, self.t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.zref)

    def test_ps3(self):
        for w in self.ws:
            z = of.z_eccentric_ps3(self.t, self.t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.zref)

    def test_ps5(self):
        for w in self.ws:
            z = of.z_eccentric_ps5(self.t, self.t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.zref)

if __name__ == '__main__':
    unittest.main()
