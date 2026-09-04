"""The library-wide flux convention: models return the deficit dF = F - 1.

Out-of-transit flux is exactly 0.0 -- an equality, not a tolerance, because 0.0 is
representable and being exactly zero is the point of the convention.

SecondaryEclipseModel is deliberately excluded from the deficit assertion: it returns
the planet's visible flux on the planet's own surface-brightness scale (out-of-eclipse
pi*k**2, zero at total eclipse), which is already a contribution rather than a
normalised flux. It gets its own test here so the exception is recorded rather than
silently tolerated.

TransmissionSpectroscopyModel is covered at kernel level in test_tsmodel_grad.py
instead: nothing in the repo constructs it through the model class, and it takes npb
from `ldc` inside evaluate() rather than from set_data().
"""
import unittest

import numpy as np
from numpy import linspace, array

from pytransit.models.rrmodel import RoadRunnerModel
from pytransit.models.udmodel import UniformDiskModel

try:
    # Match the other JAX test files: the JAX backend defaults to float32, which is
    # not precise enough for this fixture and produces NaN deep in transit (verified
    # by reproducing it with x64 disabled). This only configures JAX if it is
    # importable; the JAX subtests below are skipped by _backends() otherwise.
    import jax
    jax.config.update('jax_enable_x64', True)
except ImportError:
    pass

P, A, INC, K = 3.0, 6.0, 1.45, 0.10
LDC = array([0.30, 0.10])
TIMES = linspace(-0.5, 0.5, 500)
OOT = np.abs(TIMES) > 0.2


def _backends():
    """Every backend the model supports on this machine."""
    available = ['numba', 'jax']
    try:
        import pyopencl as cl
        if cl.get_platforms():
            available.append('opencl')
    except Exception:
        pass
    return available


class TestFluxDeficitConvention(unittest.TestCase):
    def _check(self, flux, name):
        flux = np.asarray(flux, float)
        self.assertTrue(OOT.any(), 'test setup should have out-of-transit points')
        self.assertTrue((flux[OOT] == 0.0).all(),
                        f'{name}: out-of-transit flux must be exactly 0.0, '
                        f'got max |flux| = {np.abs(flux[OOT]).max()}')
        self.assertLess(flux.min(), -1e-3, f'{name}: in-transit deficit should be negative')
        self.assertLessEqual(flux.max(), 0.0, f'{name}: deficit must never be positive')

    def test_roadrunner(self):
        for backend in _backends():
            with self.subTest(backend=backend):
                m = RoadRunnerModel(backend=backend)
                m.set_data(TIMES)
                self._check(m.evaluate(K, 0.0, P, A, INC, 0.0, 0.0, ldc=LDC),
                            f'RoadRunnerModel/{backend}')

    def test_uniform_disk_already_conformed(self):
        """UniformDiskModel used this convention before the change; it must still hold."""
        for backend in _backends():
            with self.subTest(backend=backend):
                m = UniformDiskModel(backend=backend)
                m.set_data(TIMES)
                self._check(m.evaluate(K, 0.0, P, A, INC, 0.0, 0.0),
                            f'UniformDiskModel/{backend}')

    def test_secondary_eclipse_keeps_its_own_scale(self):
        """Documents the deliberate exception rather than silently tolerating it."""
        from pytransit.models.semodel import SecondaryEclipseModel
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = np.asarray(m.evaluate(K, 0.0, P, A, INC, 0.0, 0.0), float)
        self.assertAlmostEqual(flux[OOT].max(), np.pi * K ** 2, places=10)


if __name__ == '__main__':
    unittest.main()
