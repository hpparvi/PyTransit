from __future__ import division
import unittest
import numpy as np
import numpy.testing as npt

from pytransit import Gimenez, MandelAgol

class TestCommon(unittest.TestCase):
    def setUp(self):
        self.npt=2000
        self.k = 0.1
        self.z = np.abs(np.linspace(-1.5,1.5,self.npt))
        self.u = [[0.0, 0.0],[0.2, 0.1],[0.3, 0.2],[0.4,0.3]]
        self.npb = len(self.u)


    def test_gmd_singleband(self):
        tm = Gimenez(interpolate=False)
        f  = tm(self.z, self.k, self.u[0])
        assert f.ndim == 1
        assert f.shape == (self.npt,)


    def test_gmd_multiband(self):
        tm = Gimenez(interpolate=False)
        for i in range(1,self.npb):
            f  = tm(self.z, self.k, self.u[:i+1])
            assert f.ndim == 2
            assert f.shape == (self.npt,i+1)


    def test_mad_singleband(self):
        tm = MandelAgol(interpolate=False)
        f  = tm(self.z, self.k, self.u[0])
        assert f.ndim == 1
        assert f.shape == (self.npt,)


    def test_mad_multiband(self):
        tm = MandelAgol(interpolate=False)
        for i in range(1,self.npb):
            f  = tm(self.z, self.k, self.u[:i+1])
            assert f.ndim == 2
            assert f.shape == (self.npt,i+1)
      

    def test_gmi_singleband(self):
        tm = Gimenez(interpolate=True)
        f  = tm(self.z, self.k, self.u[0])
        assert f.ndim == 1
        assert f.shape == (self.npt,)


    def test_gmi_multiband(self):
        tm = Gimenez(interpolate=True)
        for i in range(1,self.npb):
            f  = tm(self.z, self.k, self.u[:i+1])
            assert f.ndim == 2
            assert f.shape == (self.npt,i+1)


    def test_mai_singleband(self):
        tm = MandelAgol(interpolate=True)
        f  = tm(self.z, self.k, self.u[0])
        assert f.ndim == 1
        assert f.shape == (self.npt,)


    def test_mai_multiband(self):
        tm = MandelAgol(interpolate=True)
        for i in range(1,self.npb):
            f  = tm(self.z, self.k, self.u[:i+1])
            assert f.ndim == 2
            assert f.shape == (self.npt,i+1)
  

class TestGimenezModel(unittest.TestCase):
    """Tests the Gimenez model functionality and correctness.
    """

    def setUp(self):
        self.k = 0.1
        self.z = np.abs(np.linspace(-1.5,1.5,2000))
        self.u = [[0.0, 0.0],[0.2, 0.1],[0.3, 0.2],[0.4,0.3]]
        self.f_ref = np.load('reference_flux.npz')['flux']


    def test_noip_np500(self):
        f = Gimenez(npol=500, nthr=1, interpolate=False)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f, self.f_ref[:,0])
        for i in range(2,4):
            f = Gimenez(npol=500, nthr=1, interpolate=False)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f, self.f_ref[:,0:i])


    def test_ip_np500(self):
        f = Gimenez(npol=500, nthr=1, interpolate=True)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f,self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = Gimenez(npol=500, nthr=1, interpolate=True)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f,self.f_ref[:,0:i], decimal=4)


    def test_noip_np100(self):
        f = Gimenez(npol=100, nthr=1, interpolate=False)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f, self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = Gimenez(npol=100, nthr=1, interpolate=False)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f, self.f_ref[:,0:i], decimal=4)


    def test_ip_np100(self):
        f = Gimenez(npol=100, nthr=1, interpolate=True)(self.z, self.k, self.u[0])
        npt.assert_array_almost_equal(f,self.f_ref[:,0], decimal=4)
        for i in range(2,4):
            f = Gimenez(npol=100, nthr=1, interpolate=True)(self.z, self.k, self.u[0:i])
            npt.assert_array_almost_equal(f,self.f_ref[:,0:i], decimal=4)


    def test_noip_threading(self):
        for nthr in range(16):
            f = Gimenez(npol=500, nthr=nthr, interpolate=False)(self.z, self.k, self.u[0])
            npt.assert_array_almost_equal(f, self.f_ref[:,0])
            for i in range(2,4):
                f = Gimenez(npol=500, nthr=nthr, interpolate=False)(self.z, self.k, self.u[0:i])
                npt.assert_array_almost_equal(f, self.f_ref[:,0:i])


    def test_ip_threading(self):
        for nthr in range(16):
            f = Gimenez(npol=500, nthr=nthr, interpolate=True)(self.z, self.k, self.u[0])
            npt.assert_array_almost_equal(f, self.f_ref[:,0], decimal=4)
            for i in range(2,4):
                f = Gimenez(npol=500, nthr=nthr, interpolate=True)(self.z, self.k, self.u[0:i])
                npt.assert_array_almost_equal(f, self.f_ref[:,0:i], decimal=4)


if __name__ == '__main__':
    unittest.main()
