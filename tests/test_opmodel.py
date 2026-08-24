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

from math import cos, sin, sqrt

from numpy import pi, linspace, zeros, array, abs, isfinite, nanmin
from scipy.integrate import quad

from meepmeep.backends.numba.point2d import solve2d, pos_c

from pytransit import OblatePlanetModel, QuadraticModel


def quadratic_intensity(x, y, u1, u2):
    r2 = x * x + y * y
    if r2 >= 1.0:
        return 0.0
    mu = sqrt(1.0 - r2)
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


def reference_flux(cx, cy, k, f, al, u1, u2):
    """Numerically secure oblate-planet transit flux for quadratic limb darkening.

    Integrates the quadratic limb darkening intensity over the exact planet-star overlap
    region with nested adaptive quadrature.
    """
    b = 1.0 - f
    ca, sa = cos(al), sin(al)
    hh = k * sqrt(sa * sa + b * b * ca * ca)

    def scanline(y):
        if abs(y) >= 1.0:
            return 0.0
        yp = (y - cy) / k
        d = b * b * ca * ca + sa * sa - yp * yp
        if d <= 0.0:
            return 0.0
        sd = sqrt(d) / b
        u = yp * ca * sa * (1.0 / (b * b) - 1.0)
        v = sa * sa / (b * b) + ca * ca
        w = sqrt(1.0 - y * y)
        lo = max(cx + k * (u - sd) / v, -w)
        hi = min(cx + k * (u + sd) / v, w)
        if hi <= lo:
            return 0.0
        return quad(quadratic_intensity, lo, hi, args=(y, u1, u2), epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    ylo, yhi = max(cy - hh, -1.0), min(cy + hh, 1.0)
    blocked = quad(scanline, ylo, yhi, epsabs=1e-12, epsrel=1e-11, limit=500)[0] if yhi > ylo else 0.0
    return 1.0 - blocked / (pi * (1.0 - u1 / 3.0 - u2 / 6.0))


class TestOblatePlanetModel:
    time = linspace(-0.06, 0.06, 500)
    pv = dict(k=0.1, f=0.3, alpha=0.7, ldc=[0.4, 0.3], t0=0.0, p=2.0, a=6.0, i=0.5 * pi)

    def test_theta_matches_exact(self):
        """The default θ-grid scanline areas should match the exact areas to well below ppm level."""
        tm = OblatePlanetModel()
        tm.set_data(self.time)
        for inc in (0.5 * pi, 1.45, 1.41):
            pv = dict(self.pv, i=inc)
            ft = tm.evaluate(**pv)
            fx = tm.evaluate(**pv, exact_areas=True)
            assert isfinite(ft).all() and isfinite(fx).all()
            assert abs(ft - fx).max() < 1e-5

    def test_exact_areas_flag(self):
        """The initializer default and the per-call override must select the intersection area algorithm."""
        tmt = OblatePlanetModel()
        tmx = OblatePlanetModel(exact_areas=True)
        tmt.set_data(self.time)
        tmx.set_data(self.time)
        ft = tmt.evaluate(**self.pv)
        fx = tmx.evaluate(**self.pv)
        assert (fx == tmt.evaluate(**self.pv, exact_areas=True)).all()
        assert (ft == tmx.evaluate(**self.pv, exact_areas=False)).all()
        assert abs(ft - fx).max() > 0.0

    def test_per_passband_radius_ratios(self):
        """Each passband must use its own ellipse geometry when the radius ratios differ."""
        tm = OblatePlanetModel()
        lcids = zeros(self.time.size, int)
        lcids[self.time.size // 2:] = 1
        tm.set_data(self.time, lcids=lcids, pbids=[0, 1])
        kw = dict(k=array([[0.10, 0.15]]), f=array([0.3]), alpha=array([0.7]),
                  ldc=array([[0.4, 0.3, 0.35, 0.25]]),
                  t0=array([0.0]), p=array([2.0]), a=array([6.0]), i=array([1.48]))
        ft = tm.evaluate(**kw)
        fx = tm.evaluate(**kw, exact_areas=True)
        assert abs(ft - fx).max() < 1e-5
        n = self.time.size // 2
        assert nanmin(fx[n:]) < nanmin(fx[:n])  # the larger radius ratio must give a deeper transit


class TestOblatePlanetModelAccuracy:
    """Compare the limb-darkened model fluxes against nested adaptive quadrature."""
    u1, u2 = 0.4, 0.3
    p, a = 2.0, 6.0
    times = linspace(0.0, 0.058, 8)

    def _model_and_reference(self, k, f, alpha, inc, **kwargs):
        tm = OblatePlanetModel()
        tm.set_data(self.times)
        kwargs.setdefault('exact_areas', True)
        flux = tm.evaluate(k=k, f=f, alpha=alpha, ldc=[self.u1, self.u2],
                           t0=0.0, p=self.p, a=self.a, i=inc, **kwargs)
        xyc = solve2d(0.0, self.p, self.a, inc, 0.0, 0.0)
        ref = array([reference_flux(*pos_c(t, xyc), k, f, alpha, self.u1, self.u2) for t in self.times])
        return flux, ref

    def test_reference_against_mandel_agol(self):
        """The quadrature reference must reproduce the analytic circular-planet model."""
        qm = QuadraticModel(interpolate=False)
        qm.set_data(self.times)
        fq = qm.evaluate(k=0.1, ldc=[self.u1, self.u2], t0=0.0, p=self.p, a=self.a, i=0.5 * pi)
        xyc = solve2d(0.0, self.p, self.a, 0.5 * pi, 0.0, 0.0)
        ref = array([reference_flux(*pos_c(t, xyc), 0.1, 0.0, 0.0, self.u1, self.u2) for t in self.times])
        assert abs(fq - ref).max() < 1e-7

    def test_spherical_planet(self):
        """For f=0 the model must reach the RoadRunner baseline accuracy (~4 ppm)."""
        flux, ref = self._model_and_reference(0.1, 0.0, 0.0, 0.5 * pi)
        assert abs(flux - ref).max() < 1e-5

    def test_oblate_planet(self):
        """With the area-equivalent LD radius the error must stay below ~30 ppm for f=0.3."""
        flux, ref = self._model_and_reference(0.1, 0.3, 0.7, 0.5 * pi)
        assert abs(flux - ref).max() < 3.5e-5

    def test_oblate_planet_grazing(self):
        flux, ref = self._model_and_reference(0.1, 0.3, 0.7, 1.41)
        assert abs(flux - ref).max() < 7e-5

    def test_exact_ld(self):
        """Exact-footprint limb darkening must reach sub-ppm accuracy for strong oblateness."""
        flux, ref = self._model_and_reference(0.1, 0.3, 0.7, 0.5 * pi, exact_ld=True, exact_areas=False)
        assert abs(flux - ref).max() < 2e-6

    def test_exact_ld_grazing(self):
        flux, ref = self._model_and_reference(0.1, 0.3, 0.7, 1.41, exact_ld=True, exact_areas=False)
        assert abs(flux - ref).max() < 2.5e-6

    def test_exact_ld_with_exact_areas(self):
        """The exact-area annuli remove the scanline resolution floor in grazing geometries."""
        flux, ref = self._model_and_reference(0.1, 0.3, 0.7, 1.41, exact_ld=True, exact_areas=True)
        assert abs(flux - ref).max() < 1.5e-6

    def test_exact_ld_flag(self):
        """The initializer default and the per-call override must select the limb darkening mode."""
        tm0 = OblatePlanetModel()
        tm1 = OblatePlanetModel(exact_ld=True)
        tm0.set_data(self.times)
        tm1.set_data(self.times)
        kw = dict(k=0.1, f=0.3, alpha=0.7, ldc=[self.u1, self.u2], t0=0.0, p=self.p, a=self.a, i=0.5 * pi)
        f0 = tm0.evaluate(**kw)
        f1 = tm1.evaluate(**kw)
        assert (f1 == tm0.evaluate(**kw, exact_ld=True)).all()
        assert (f0 == tm1.evaluate(**kw, exact_ld=False)).all()
        assert abs(f0 - f1).max() > 0.0
