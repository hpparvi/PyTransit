"""The annulus weight table used by the OpenCL RoadRunner model and the transmission spectroscopy model."""
import unittest

import numpy as np
from numpy import linspace

from pytransit.backends.numba.rrmodel import create_z_grid, calculate_weights_2d, calculate_weights_3d


class TestWeightTable(unittest.TestCase):
    def test_k_spacing_matches_the_table_nodes(self):
        ze, _ = create_z_grid(0.7, 20, 20)
        nk, kmin, kmax, ng = 16, 0.05, 0.35, 40
        dk, _, weights = calculate_weights_3d(nk, kmin, kmax, ze, ng)
        ks = linspace(kmin, kmax, nk)
        self.assertAlmostEqual(dk, ks[1] - ks[0])
        for ik in (0, 7, nk - 1):
            _, _, wg = calculate_weights_2d(kmin + ik * dk, ze, ng)
            np.testing.assert_allclose(weights[ik], wg, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
