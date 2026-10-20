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

"""`GravityDarkenedModel.evaluate` is the supported replacement for the deprecated `evaluate_ps`."""

import warnings
from math import sqrt

import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy import pi, linspace, array, isfinite, arccos
from scipy.integrate import quad

from pytransit import GravityDarkenedModel
from pytransit.contamination import BoxcarFilter
from pytransit.models.numba.gdmodel import star_planet_overlap

TIMES = linspace(-0.12, 0.12, 300)
PV = dict(rho=0.5, rperiod=0.5, tpole=8000.0, phi=0.3, beta=0.25, t0=0.0, p=2.0, a=5.0, i=0.5 * pi, l=0.4)


@pytest.fixture(scope='module')
def model():
    tm = GravityDarkenedModel(filters=BoxcarFilter('TESS', 600, 1000), model='blackbody', tmin=6000, tmax=9000)
    tm.set_data(TIMES)
    return tm


def test_evaluate_does_not_warn(model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.evaluate(k=array([0.1]), ldc=array([0.2, 0.1]), **PV)
    assert not [w for w in caught if issubclass(w.category, AstropyDeprecationWarning)]


def test_evaluate_matches_evaluate_ps(model):
    flux = model.evaluate(k=array([0.1]), ldc=array([0.2, 0.1]), **PV)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyDeprecationWarning)
        legacy = model.evaluate_ps(k=array([0.1]), ldc=array([0.2, 0.1]), **PV)
    assert isfinite(flux).all() and flux.min() < 0.99
    assert (flux == legacy).all()


def test_evaluate_accepts_a_scalar_radius_ratio_and_list_coefficients(model):
    """`evaluate_ps` needs `k` as a 1D array and `ldc` as an ndarray; `evaluate` converts them."""
    reference = model.evaluate(k=array([0.1]), ldc=array([0.2, 0.1]), **PV)
    assert (model.evaluate(k=0.1, ldc=[0.2, 0.1], **PV) == reference).all()


def test_evaluate_ps_still_warns(model):
    with pytest.warns(AstropyDeprecationWarning):
        model.evaluate_ps(k=array([0.1]), ldc=array([0.2, 0.1]), **PV)


@pytest.mark.parametrize('b', [0.0, 0.3, 0.7, 0.9])
def test_default_resolution_reproduces_a_non_rotating_star(b):
    """Without rotation the star is a limb-darkened sphere, and Mandel & Agol is exact.

    The default resolution reaches about 30 ppm; the earlier defaults (sres=80, pres=5) gave 165 ppm.
    """
    from pytransit import QuadraticModel
    time = linspace(-0.13, 0.13, 300)
    tm = GravityDarkenedModel(filters=BoxcarFilter('blue', 400, 500), model='blackbody', tmin=5000, tmax=9500)
    tm.set_data(time)
    qm = QuadraticModel(interpolate=False)
    qm.set_data(time)
    inc = arccos(b / 6.0)
    flux = tm.evaluate(0.1, 0.3, 1e4, 8500.0, 0.0, 0.22, [0.3, 0.2], 0.0, 3.0, 6.0, inc, 0.0)
    assert abs(flux - qm.evaluate(0.1, [0.3, 0.2], 0.0, 3.0, 6.0, inc)).max() < 4e-5


def _reference_overlap(x, y, k, f):
    """The area of a planet disk inside the projected stellar ellipse, by scanline quadrature."""
    b = 1.0 - f

    def chord(yy):
        dc = k * k - (yy - y) ** 2
        de = 1.0 - (yy / b) ** 2
        if dc <= 0.0 or de <= 0.0:
            return 0.0
        lo, hi = max(x - sqrt(dc), -sqrt(de)), min(x + sqrt(dc), sqrt(de))
        return max(hi - lo, 0.0)

    lo, hi = max(y - k, -b), min(y + k, b)
    return quad(chord, lo, hi, epsabs=1e-13, epsrel=1e-12, limit=200)[0] if hi > lo else 0.0


@pytest.mark.parametrize('x,y', [(0.0, 0.88), (0.0, 0.95), (0.97, 0.0), (0.6, 0.65), (0.2, 0.1), (0.0, 1.2)],
                         ids=['polar-edge', 'polar-limb', 'equatorial-limb', 'diagonal', 'inside', 'outside'])
def test_star_planet_overlap(x, y):
    """The planet must be clipped by the stellar ellipse itself, not by a stretched circle."""
    k, f = 0.1, 0.09
    assert abs(star_planet_overlap(x, y, k, f) - _reference_overlap(x, y, k, f)) < 1e-12


def test_misaligned_transit_over_a_flattened_star_matches_brute_force():
    """A path crossing the stellar poles must make contact where the flattened limb really is.

    Both models carry their own resolution error (tens of ppm); the stretched-circle overlap the
    fast model used before missed by about 300 ppm.
    """
    tm = GravityDarkenedModel(filters=BoxcarFilter('blue', 400, 500), model='blackbody', tmin=5000, tmax=9500)
    tm.set_data(linspace(-0.13, 0.13, 300))
    args = (0.1, 0.3, 0.6, 8500.0, 0.0, 0.22, [0.3, 0.2], 0.0, 3.0, 6.0, arccos(0.3 / 6.0), 0.5 * pi)
    assert abs(tm.evaluate(*args) - tm.evaluate_brute(*args, res=800)).max() < 6e-5
