#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The annulus grid used by the OpenCL RoadRunner model: uniform-gamma annuli sampled at their mean mu."""

import pytest
from numpy import pi, arcsin, diff, sqrt, allclose

from pytransit.models.roadrunner.common import create_z_grid


class TestGridGeometry:
    def test_edges_are_uniform_in_gamma(self):
        """z = sin(gamma) with gamma equally spaced on (0, pi/2]: dense at the limb, finite at centre."""
        ze, _ = create_z_grid(40)
        assert ze.size == 40
        assert ze[-1] == pytest.approx(1.0)
        assert allclose(diff(arcsin(ze)), 0.5 * pi / 40)

    def test_samples_are_the_area_weighted_mean_mu_of_each_annulus(self):
        """Sampling at the area-weighted mean mu is exact for any profile linear in mu."""
        ze, zm = create_z_grid(40)
        z0 = ze.copy(); z0[1:] = ze[:-1]; z0[0] = 0.0
        mu0, mu1 = sqrt(1 - z0 ** 2), sqrt(1 - ze ** 2)
        mubar = (2.0 / 3.0) * (mu0 ** 3 - mu1 ** 3) / (mu0 ** 2 - mu1 ** 2)
        assert allclose(zm, sqrt(1 - mubar ** 2))

    def test_samples_lie_inside_their_annuli(self):
        ze, zm = create_z_grid(40)
        z0 = ze.copy(); z0[1:] = ze[:-1]; z0[0] = 0.0
        assert (zm > z0).all() and (zm < ze).all()
        assert (diff(zm) > 0).all()
