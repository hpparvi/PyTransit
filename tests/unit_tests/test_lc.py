from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from pytransit import Gimenez, MandelAgol

## TODO: Write the tests for supersampled models

class TestCommon(unittest.TestCase):
    def setUp(self):
        self.npt=2000
        self.k = 0.1
        self.z = np.abs(np.linspace(-1.5,1.5,self.npt))
        self.u = [[0.0, 0.0],[0.2, 0.1],[0.3, 0.2],[0.4,0.3]]
        self.npb = len(self.u)


    def sbt(self, tm):
        f  = tm(self.z, self.k, self.u[0])
        assert f.ndim == 1, 'Given one set of ld coefficients, the returned array should be one dimensional.'
        assert f.shape == (self.npt,), 'Returned array shape should be (npt)'

    def mbt(self, tm):
        for i in range(1,self.npb):
            f  = tm(self.z, self.k, self.u[:i+1])
            assert f.ndim == 2, 'Given multiple sets of ld coefficients, the returned array should be two dimensional.'
            assert f.shape == (self.npt,i+1), 'Returned array shape should be (npt,npb)'


    ## Gimenez model tests
    ## -------------------
    def test_gmd_singleband(self):
        self.sbt(Gimenez(interpolate=False))

    def test_gmd_multiband(self):
        self.mbt(Gimenez(interpolate=False))

    def test_gmi_singleband(self):
        self.sbt(Gimenez(interpolate=True))

    def test_gmi_multiband(self):
        self.mbt(Gimenez(interpolate=True))


    ## Mandel & Agol model tests
    ## -------------------------
    def test_mad_singleband(self):
        self.sbt(MandelAgol(interpolate=False))

    def test_mad_multiband(self):
        self.mbt(MandelAgol(interpolate=False))

    def test_mai_singleband(self):
        self.sbt(MandelAgol(interpolate=True))

    def test_mai_multiband(self):
        self.mbt(MandelAgol(interpolate=True))
  

class TestGimenezModel(unittest.TestCase):
    """Tests the Gimenez model functionality and correctness.
    """

    def setUp(self):
        self.k = 0.1
        self.z = np.abs(np.linspace(-1.5,1.5,2000))
        self.u = [[0.0, 0.0],[0.2, 0.1],[0.3, 0.2],[0.4,0.3]]
        self.f_ref = np.load('reference_flux.npz')['flux']


    def tae(self, tm):
        f = tm(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f, self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = tm(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f, self.f_ref[:,0:i], decimal=4)


    def test_noip_np500(self):
        self.tae(Gimenez(npol=500, nthr=1, interpolate=False))
 
    def test_ip_np500(self):
        self.tae(Gimenez(npol=500, nthr=1, interpolate=True))

    def test_noip_np100(self):
        self.tae(Gimenez(npol=100, nthr=1, interpolate=False))
 
    def test_ip_np100(self):
        self.tae(Gimenez(npol=100, nthr=1, interpolate=True))

    def test_noip_threading(self):
        for nthr in range(16):
            tm = Gimenez(npol=500, nthr=nthr, interpolate=False)
            f = tm(self.z, self.k, self.u[0])
            npt.assert_array_almost_equal(f, self.f_ref[:,0])
            for i in range(2,4):
                f = tm(self.z, self.k, self.u[0:i])
                npt.assert_array_almost_equal(f, self.f_ref[:,0:i])


    def test_ip_threading(self):
        for nthr in range(16):
            tm = Gimenez(npol=500, nthr=nthr, interpolate=True)
            f = tm(self.z, self.k, self.u[0])
            npt.assert_array_almost_equal(f, self.f_ref[:,0], decimal=4)
            for i in range(2,4):
                f = tm(self.z, self.k, self.u[0:i])
                npt.assert_array_almost_equal(f, self.f_ref[:,0:i], decimal=4)


if __name__ == '__main__':
    unittest.main()
