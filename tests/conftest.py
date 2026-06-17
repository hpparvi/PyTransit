#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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

from types import SimpleNamespace

import pytest
from numpy import linspace, pi, tile
from numpy.random import randint, seed, uniform, normal


@pytest.fixture
def model_data():
    """Deterministic test data shared by the transit-model tests.

    The random draws are seeded and kept in a fixed order so the
    generated values are reproducible across test runs.
    """
    seed(0)
    npt = 20
    npv = 5
    time = linspace(-0.1, 0.1, npt)
    lcids = randint(0, 3, size=npt)
    pbids = [0, 1, 1]
    ldc = tile([[0.01, 0.3]], (npv, 2))
    radius_ratios = uniform(0.09, 0.11, size=(npv, 2))
    zero_epochs = normal(0.0, 0.01, size=npv)
    periods = normal(1.0, 0.01, size=npv)
    smas = normal(3.0, 0.01, size=npv)
    inclinations = uniform(0.49 * pi, 0.5 * pi, size=npv)
    eccentricities = uniform(0.0, 0.9, size=npv)
    omegas = uniform(0, 2 * pi, size=npv)

    return SimpleNamespace(
        npt=npt, npv=npv, time=time, lcids=lcids, pbids=pbids, ldc=ldc,
        radius_ratios=radius_ratios, zero_epochs=zero_epochs, periods=periods,
        smas=smas, inclinations=inclinations, eccentricities=eccentricities,
        omegas=omegas,
    )
