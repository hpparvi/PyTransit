from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt
from math import pi
from itertools import product
from numpy import array

from pytransit.orbits_f import orbits as of

class TestCircularZ(unittest.TestCase):
    """Test the routines to calculate the normalized projected distance (z) assuming zero eccentricity.
    """
    def setUp(self):
        self.t  = [-0.2, 0, 0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 1.0]
        self.t0s = [0, 0.213, 0.5456, 1000.945, 100035345354.332]
        self.p  = 1
        self.a  = 5
        self.i  = 0.5*np.pi
        self.zref = np.array([0.0 , 3.535533, 5.0 , -3.535533, -0.0, -5.0])
        self.ws = [0, 0.5*pi, pi, 1.5*pi, 2*pi, 0.012, 0.24, -0.53]
        
        self.ref_zs = [array([ 4.75528258,  0.        ,  3.5355339 ,  5.        , -3.53553391,
                              -0.        , -4.45503262, -5.        ,  0.        ]),
                       array([-2.59908672,  4.86549255,  2.62587315,  1.15194713,  4.25497241,
                              -4.86549256, -1.18249499, -1.15194713,  4.86549256]),
                       array([-4.99808937, -1.41304668, -2.3922333 , -4.79617546,  4.39058308,
                               1.41304668,  3.63191386,  4.79617546, -1.41304669]),
                       array([ 3.95076837,  1.69369986,  4.52413991, -4.70440015, -2.12888659,
                              -1.69369986, -4.96057487,  4.70440015,  1.69369985]),
                       array([ 4.37339681, -3.65650322, -0.17409761, -3.41029256, -4.99696886,
                               3.65647283,  1.37885316,  3.41032513, -3.65644245])]

    def test_circular(self):
        for (i,t0), w in product(enumerate(self.t0s), self.ws):
            z = of.z_circular(self.t, t0, self.p, self.a, self.i, nth=1)
            npt.assert_array_almost_equal(z, self.ref_zs[i])

    def test_ip(self):
        for (i,t0), w in product(enumerate(self.t0s), self.ws):
            z = of.z_eccentric_ip(self.t, t0, self.p, self.a, self.i, 0, w, nth=1, update=True)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)
                
    def test_iter(self):
        for (i,t0), w in product(enumerate(self.t0s), self.ws):
            z = of.z_eccentric_iter(self.t, t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)
 
    def test_newton(self):
        for (i,t0), w in product(enumerate(self.t0s), self.ws):
            z = of.z_eccentric_newton(self.t, t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)

    def test_ps3(self):
        for (i,t0), w in product(enumerate(self.t0s), self.ws):
            z = of.z_eccentric_ps3(self.t, t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)

    def test_ps5(self):
        for (i,t0), w in product(enumerate(self.t0s), self.ws):
            z = of.z_eccentric_ps5(self.t, t0, self.p, self.a, self.i, 0, w, nth=1)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)

if __name__ == '__main__':
    unittest.main()
