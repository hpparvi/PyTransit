from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt
from math import pi
from itertools import product
from numpy import array

from pytransit.orbits_f import orbits as of

class TestEccentricZ(unittest.TestCase):
    """Test the routines to calculate the normalized projected distance (z) assuming zero eccentricity.
    """
    def setUp(self):
        self.p  = 1
        self.a  = 5
        self.i  = 0.5*np.pi
        self.t   = [-0.2, 0.0, 0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 1.0]
        self.t0s = [0.01, 0.213, 0.5456, 1000.945]
        self.es  = [0.05, 0.15, 0.35, 0.5, 0.75]
        self.ws = [0, 0.5*pi, pi, 1.5*pi, 2*pi, 0.012, 0.24, -0.53]

        ## Note: We assume that the routine using the Newton's method is bug free...
        self.ref_zs = [of.z_eccentric_newton(self.t,t0,self.p,self.a,self.i,e,w,nth=1)
                       for t0,e,w in product(self.t0s,self.es,self.ws)]

    def test_ip(self):
        for i,(t0,e,w) in enumerate(product(self.t0s, self.es, self.ws)):
            z = of.z_eccentric_ip(self.t, t0, self.p, self.a, self.i, e, w, nth=1, update=True)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)
                
    def test_iter(self):
        for i,(t0,e,w) in enumerate(product(self.t0s, self.es, self.ws)):
            z = of.z_eccentric_iter(self.t, t0, self.p, self.a, self.i, e, w, nth=1)
            npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=3)
 
    def test_ps3(self):
        for i,(t0,e,w) in enumerate(product(self.t0s, self.es, self.ws)):
            if e < 0.35:
                z = of.z_eccentric_ps3(self.t, t0, self.p, self.a, self.i, e, w, nth=1)
                npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=2)

    def test_ps5(self):
        for i,(t0,e,w) in enumerate(product(self.t0s, self.es, self.ws)):
            if e < 0.35:
                z = of.z_eccentric_ps5(self.t, t0, self.p, self.a, self.i, e, w, nth=1)
                npt.assert_array_almost_equal(z, self.ref_zs[i], decimal=2)

if __name__ == '__main__':
    unittest.main()
