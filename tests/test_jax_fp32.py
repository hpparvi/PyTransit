"""The JAX RoadRunner model in single precision.

Two things break a transit model in float32, and both are guarded here.

**Quadrature-node margin.** `_ldm_nodes` keeps the impact parameter strictly inside the
limb with `min(gs, 1 - margin)`. The margin has to exceed the working precision's epsilon
or it rounds away entirely: a hard-coded 1e-9 is far below float32 eps (1.2e-7), so in
single precision the guard vanished, `b` landed exactly on `1 + k`, the quadrature weights
underflowed to zero, and the last table row became `0/0 = NaN`. The split-cubic build then
spread that NaN across every interval touching it, so all samples with `g` near 1 -- the
in-window, out-of-transit band -- came back NaN.

**Time origin.** A float32 ULP at BJD 2458325 is about 0.25 d, so folding an absolute
timestamp in single precision quantises a whole transit onto a couple of representable
values. The model subtracts a reference epoch in float64 before the kernel ever sees a
time, exactly as the OpenCL backend does in `CLModel.set_data`.

This module deliberately does NOT enable x64: it is the only test file that exercises the
single-precision path, and `jax_enable_x64` is global. If another module has already
switched it on in the same interpreter, the tests skip rather than silently pass in double
precision.
"""
import unittest

import numpy as np
import jax
import jax.numpy as jnp
from numpy import linspace, array

from pytransit.models.rrmodel import RoadRunnerModel

P, A, INC, K = 3.0, 6.0, 1.45, 0.10
LDC = array([0.30, 0.10])
BJD = 2458325.0


def _is_single_precision() -> bool:
    """True when JAX is in its default 32-bit mode."""
    return jnp.zeros(1).dtype == jnp.float32


class TestSinglePrecision(unittest.TestCase):
    def setUp(self):
        if not _is_single_precision():
            self.skipTest("jax_enable_x64 is on in this interpreter; run this file on its own")

    def _evaluate(self, times, t0):
        model = RoadRunnerModel(backend='jax')
        model.set_data(times)
        return np.asarray(model.evaluate(K, t0, P, A, INC, 0.0, 0.0, ldc=LDC), float)

    def test_flux_is_finite_across_the_whole_window(self):
        """No NaN anywhere -- including the in-window, out-of-transit band.

        The failure this pins was confined to `g` near 1, i.e. |t| between the end of the
        transit and the edge of the bounding window, so a test spanning only the transit
        would miss it entirely.
        """
        times = linspace(-0.2, 0.2, 5000)
        flux = self._evaluate(times, 0.0)
        self.assertEqual(flux.dtype, np.float64)      # asarray upcast; the kernel ran fp32
        n_bad = int((~np.isfinite(flux)).sum())
        self.assertEqual(n_bad, 0, f'{n_bad} of {flux.size} samples are not finite')

    def test_transit_is_actually_present(self):
        """Guards the test above from passing on an all-zero flux."""
        times = linspace(-0.2, 0.2, 5000)
        flux = self._evaluate(times, 0.0)
        self.assertLess(flux.min(), -1e-3)
        self.assertLessEqual(flux.max(), 0.0)

    def test_realistic_timestamps_match_a_zero_origin(self):
        """A BJD time origin must give the same light curve as a zero-centred one.

        Without the reference-epoch shift the fp32 fold quantises the transit away and the
        error is the full transit depth.
        """
        offsets = linspace(-0.2, 0.2, 2000)
        zero = self._evaluate(offsets, 0.0)
        bjd = self._evaluate(offsets + BJD, BJD)
        depth = -zero.min()
        self.assertGreater(depth, 1e-3, 'test setup should produce a real transit')
        np.testing.assert_allclose(bjd, zero, atol=1e-4 * depth)

    def test_single_precision_tracks_double_precision(self):
        """fp32 must agree with the fp64 reference to a small fraction of the depth."""
        times = linspace(-0.2, 0.2, 2000)
        single = self._evaluate(times, 0.0)
        reference = RoadRunnerModel(backend='numba')
        reference.set_data(times)
        double = np.asarray(reference.evaluate(K, 0.0, P, A, INC, 0.0, 0.0, ldc=LDC), float)
        depth = -double.min()
        np.testing.assert_allclose(single, double, atol=1e-3 * depth)


if __name__ == '__main__':
    unittest.main()
