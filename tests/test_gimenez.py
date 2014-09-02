from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as pl

from pytransit.gimenez import Gimenez

#m1 = Gimenez(npol=500, lerp=False)
#m2 = Gimenez(npol=500, lerp=True)
#np.savez('reference_flux', flux=m1(z, 0.1, u))

class TestGimenezModel(unittest.TestCase):
    """Tests the Gimenez model functionality and correctness.
    """

    def setUp(self):
        self.k = 0.1
        self.z = np.abs(np.linspace(-1.5,1.5,2000))
        self.u = [[0.0, 0.0],[0.2, 0.1],[0.3, 0.2],[0.4,0.3]]
        self.f_ref = np.load('reference_flux.npz')['flux']


    def test_nolerp_basic(self):
        f = Gimenez(npol=500, nthr=1, lerp=False)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f, self.f_ref[:,0])
        for i in range(2,4):
            f = Gimenez(npol=500, nthr=1, lerp=False)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f, self.f_ref[:,0:i])


    def test_lerp_basic(self):
        f = Gimenez(npol=500, nthr=1, lerp=True)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f,self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = Gimenez(npol=500, nthr=1, lerp=True)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f,self.f_ref[:,0:i], decimal=4)


    def test_nolerp_npol(self):
        f = Gimenez(npol=100, nthr=1, lerp=False)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f, self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = Gimenez(npol=100, nthr=1, lerp=False)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f, self.f_ref[:,0:i], decimal=4)


    def test_lerp_npol(self):
        f = Gimenez(npol=100, nthr=1, lerp=True)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f,self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = Gimenez(npol=100, nthr=1, lerp=True)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f,self.f_ref[:,0:i], decimal=4)


    def test_nolerp_threading(self):
        for nthr in range(16):
            f = Gimenez(npol=500, nthr=nthr, lerp=False)(self.z, self.k, self.u[0])
            npt.assert_array_almost_equal(f, self.f_ref[:,0])
            for i in range(2,4):
                f = Gimenez(npol=500, nthr=nthr, lerp=False)(self.z, self.k, self.u[0:i])
                npt.assert_array_almost_equal(f, self.f_ref[:,0:i])


    def test_lerp_threading(self):
        for nthr in range(16):
            f = Gimenez(npol=500, nthr=nthr, lerp=True)(self.z, self.k, self.u[0])
            npt.assert_array_almost_equal(f, self.f_ref[:,0], decimal=4)
            for i in range(2,4):
                f = Gimenez(npol=500, nthr=nthr, lerp=True)(self.z, self.k, self.u[0:i])
                npt.assert_array_almost_equal(f, self.f_ref[:,0:i], decimal=4)


if __name__ == '__main__':
    unittest.main()
