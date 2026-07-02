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

from math import acos, cos, hypot, pi, sin, sqrt

from numpy import abs, arange, clip, cos as ncos, linspace, maximum, minimum, sin as nsin, sqrt as nsqrt

from pytransit.models.roadrunner.ecintersection import create_ellipse, ellipse_circle_intersection_area


def circle_circle_intersection_area(z: float, k: float) -> float:
    """Exact area of intersection between a unit circle and a circle with radius k at distance z."""
    if z >= 1.0 + k:
        return 0.0
    if z <= 1.0 - k:
        return pi * k * k
    return (k * k * acos((z * z + k * k - 1.0) / (2 * z * k))
            + acos((z * z + 1.0 - k * k) / (2 * z))
            - 0.5 * sqrt((1 + k - z) * (z + k - 1) * (z - k + 1) * (z + k + 1)))


def reference_area(cx: float, cy: float, k: float, f: float, a: float, n: int = 500_000) -> float:
    """High-resolution scanline reference for the ellipse-circle intersection area."""
    b = 1.0 - f
    hh = k * sqrt(sin(a) ** 2 + b * b * cos(a) ** 2)
    ys = (arange(n) + 0.5) / n * 2 * hh - hh
    dy = 2 * hh / n
    y = ys / k
    ca, sa = cos(a), sin(a)
    d = b * b * ca * ca + sa * sa - y * y
    m = d > 0.0
    sd = nsqrt(d[m]) / b
    u = (ca * sa * y[m]) / b ** 2 - ca * sa * y[m]
    v = sa ** 2 / b ** 2 + ca ** 2
    x0, x1 = k * (u - sd) / v, k * (u + sd) / v
    yy = ys[m] + cy
    inside = abs(yy) <= 1.0
    w = nsqrt(clip(1.0 - yy[inside] ** 2, 0.0, None))
    lo = maximum(-w - cx, x0[inside])
    hi = minimum(w - cx, x1[inside])
    return (clip(hi - lo, 0.0, None)).sum() * dy


class TestEllipseCircleIntersectionArea:
    k, f, a = 0.1, 0.3, 0.7
    ny = 10_000

    def test_circular_planet(self):
        """A non-flattened, non-rotated ellipse should reproduce the exact circle-circle intersection area."""
        ys, xs = create_ellipse(self.ny, self.k, 0.0, 0.0)
        for z in linspace(0.905, 1.095, 20):
            area = ellipse_circle_intersection_area(z, 0.0, z, self.k, 0.0, xs, ys)
            assert abs(area - circle_circle_intersection_area(z, self.k)) < 1e-7

    def test_side_ingress_egress(self):
        """Partial overlap with the ellipse center outside the |cx| <= k band."""
        ys, xs = create_ellipse(self.ny, self.k, self.f, self.a)
        for cx in (-0.7, -0.5, 0.5, 0.7):
            for cy in (0.0, 0.6, 0.75, 0.85):
                z = hypot(cx, cy)
                if 1.0 - self.k < z < 1.0 + self.k:
                    area = ellipse_circle_intersection_area(cx, cy, z, self.k, self.f, xs, ys)
                    assert abs(area - reference_area(cx, cy, self.k, self.f, self.a)) < 1e-7

    def test_grazing_center_band(self):
        """Grazing geometries with |cx| <= k, where the unclamped scanline overlap used to bias the area low."""
        ys, xs = create_ellipse(self.ny, self.k, self.f, self.a)
        for cx in linspace(-0.09, 0.09, 7):
            for cy in (0.95, 1.0, 1.05):
                z = hypot(cx, cy)
                if 1.0 - self.k < z < 1.0 + self.k:
                    area = ellipse_circle_intersection_area(cx, cy, z, self.k, self.f, xs, ys)
                    assert abs(area - reference_area(cx, cy, self.k, self.f, self.a)) < 1e-7

    def test_limits(self):
        ys, xs = create_ellipse(self.ny, self.k, self.f, self.a)
        assert ellipse_circle_intersection_area(0.0, 0.0, 0.0, self.k, self.f, xs, ys) == pi * self.k ** 2 * (1.0 - self.f)
        assert ellipse_circle_intersection_area(1.2, 0.0, 1.2, self.k, self.f, xs, ys) == 0.0
