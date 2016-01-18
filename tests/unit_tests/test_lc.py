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
  

class TestSpecialCases(unittest.TestCase):
    """Test the models when z is in [0,k,1-k,1,1+k]"""
    def setUp(self):
        self.k = 0.1
        self.u = [0.1, 0.2]
        self.z = np.array([ 0.00,  0.01,  0.02,  0.03,  0.04,  0.09,  0.10,  0.11,  0.12,
                            0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,  0.90,  0.99,
                            1.00,  1.09,  1.10,  1.11,  1.12])
        self.f = np.array([ 0.98928841,  0.98928846,  0.98928863,  0.98928890,  0.98928929,
                            0.98929290,  0.98929398,  0.98929516,  0.98929647,  0.98931150,
                            0.98934366,  0.98939543,  0.98947512,  0.98959659,  0.98978450,
                            0.99008997,  0.99067950,  0.99512386,  0.99571273,  0.99985831,
                            0.99995404,  1.00000000,  1.00000000])

    def test_mad_special_z(self):
        """Test the direct Mandel & Agol model with z in [0,k,1-k,1,1+k]"""
        tm = MandelAgol(interpolate=False)
        npt.assert_array_almost_equal(tm(self.z, self.k, self.u), self.f, decimal=4)

    def test_mai_special_z(self):
        """Test the interpolated Mandel & Agol model with z in [0,k,1-k,1,1+k]"""
        tm = MandelAgol(interpolate=True, klims=[0.09,1.11])
        npt.assert_array_almost_equal(tm(self.z, self.k, self.u), self.f, decimal=4)

    def test_gmd_special_z(self):
        """Test the direct Gimenez model with z in [0,k,1-k,1,1+k]"""
        tm = Gimenez(interpolate=False)
        npt.assert_array_almost_equal(tm(self.z, self.k, self.u), self.f, decimal=4)

    def test_gmd_special_z(self):
        """Test the interpolated Gimenez model with z in [0,k,1-k,1,1+k]"""
        tm = Gimenez(interpolate=True)
        npt.assert_array_almost_equal(tm(self.z, self.k, self.u), self.f, decimal=4)


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
