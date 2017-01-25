from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from numpy.random import uniform, seed
from pytransit.orbits_f import orbits as of
import pytransit.utils.orbits as o

class TestOrbits(unittest.TestCase):
    """Tests the orbits module functionality and correctness.
    """

    def setUp(self):
        self.t  = 1.500001
        self.t0 = 1.5
        self.p  = 14.0
        self.a  = 6.0

        seed(0)
        self.bs = uniform(0, 0.90, 50)
        self.ws = uniform(0, 2*np.pi, 50)

        
    def test_lerp_threading(self):
        pass


    def test_b_circular(self):
        """Impact parameter mapping test for circular orbits.
        
        Test impact parameter mapping to ensure that b == z(t0) for circular orbits.
        """
        Is = o.i_from_ba(self.bs, self.a)

        zs = np.concatenate([of.z_circular(self.t, self.t0, self.p, self.a, i, 1) for i in Is])
        npt.assert_array_almost_equal(self.bs, zs)

        zs = np.concatenate([of.z_eccentric_newton(self.t, self.t0, self.p, self.a, i, 0, 0, 1) for i in Is])
        npt.assert_array_almost_equal(self.bs, zs)

        zs = np.concatenate([of.z_eccentric_iter(self.t, self.t0, self.p, self.a, i, 0, 0, 1) for i in Is])
        npt.assert_array_almost_equal(self.bs, zs)

        zs = np.concatenate([of.z_eccentric_ps3(self.t, self.t0, self.p, self.a, i, 0, 0, 1) for i in Is])
        npt.assert_array_almost_equal(self.bs, zs)

        zs = np.concatenate([of.z_eccentric_ps5(self.t, self.t0, self.p, self.a, i, 0, 0, 1) for i in Is])
        npt.assert_array_almost_equal(self.bs, zs)


        
    def test_b_ecc_newton(self):
        """Impact parameter mapping test for z_eccentric_newton.

        Test impact parameter mapping to ensure that b == z(t0) for eccentric orbits
        using the Newton's method.
        """
        
        es = uniform(0, 0.75, 50)
        Is = o.i_from_baew(self.bs, self.a, es, self.ws)
        zs = np.concatenate([of.z_eccentric_newton(self.t, self.t0, self.p, self.a, i, e, w, 1) for i,e,w in zip(Is, es, self.ws)])
        npt.assert_array_almost_equal(self.bs, zs)


    def test_b_ecc_iter(self):
        """Impact parameter mapping test for z_eccentric_iter.

        Test impact parameter mapping to ensure that b == z(t0) for eccentric orbits
        using the iterative method.
        """
        es = uniform(0, 0.75, 50)
        Is = o.i_from_baew(self.bs, self.a, es, self.ws)
        zs = np.concatenate([of.z_eccentric_iter(self.t, self.t0, self.p, self.a, i, e, w, 1) for i,e,w in zip(Is, es, self.ws)])
        npt.assert_array_almost_equal(self.bs, zs, decimal=4)

        
    def test_b_ecc_ps3(self):
        """Impact parameter mapping test for z_eccentric_ps3.

        Test impact parameter mapping to ensure that b == z(t0) for eccentric orbits
        using the ps3 method.
        """
        es = uniform(0, 0.15, 50)
        Is = o.i_from_baew(self.bs, self.a, es, self.ws)
        zs = np.concatenate([of.z_eccentric_ps3(self.t, self.t0, self.p, self.a, i, e, w, 1) for i,e,w in zip(Is, es, self.ws)])
        npt.assert_array_almost_equal(self.bs, zs, decimal=3)

        
    def test_b_ecc_ps5(self):
        """Impact parameter mapping test for z_eccentric_ps5.

        Test impact parameter mapping to ensure that b == z(t0) for eccentric orbits
        using the ps5 method.
        """
        es = uniform(0, 0.20, 50)
        Is = o.i_from_baew(self.bs, self.a, es, self.ws)
        zs = np.concatenate([of.z_eccentric_ps5(self.t, self.t0, self.p, self.a, i, e, w, 1) for i,e,w in zip(Is, es, self.ws)])
        npt.assert_array_almost_equal(self.bs, zs, decimal=3)

        
if __name__ == '__main__':
    unittest.main()
