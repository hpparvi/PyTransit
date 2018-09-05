## PyTransit
## Copyright (C) 2010--2017  Hannu Parviainen
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from numpy import pi, array
import pytransit.orbits_py as orbits

TWO_PI = 2 * pi


def not_implemented(*nargs, **kwargs):
    raise NotImplementedError


class Orbit(object):
    methods = 'iteration newton ps3 ps5 interpolation'.split()

    ea_functions = dict(iteration=orbits.ea_iter_v,
                        newton=orbits.ea_newton_v,
                        ps3=not_implemented,
                        ps5=not_implemented,
                        interpolation=not_implemented)

    ta_functions = dict(iteration=orbits.ta_iter_v,
                        newton=orbits.ta_newton_v,
                        ps3=orbits.ta_ps3,
                        ps5=orbits.ta_ps5,
                        interpolation=orbits.ta_ip)

    z_functions = dict(iteration=orbits.z_iter_v,
                       newton=orbits.z_newton_v,
                       ps3=orbits.z_ps3,
                       ps5=orbits.z_ps5,
                       interpolation=orbits.z_iter_v)

    def __init__(self, method='iteration', nthr=0, circular_e_threshold=1e-5):
        assert method in self.methods
        self.nthr = nthr
        self.method = method
        self._mine = circular_e_threshold
        self._ea_function = self.ea_functions[method]
        self._ta_function = self.ta_functions[method]
        self._z_function = self.z_functions[method]

    def mean_anomaly(self, time, t0, p, e=0., w=0.):
        return orbits.mean_anomaly(time, t0, p, e, w)

    def eccentric_anomaly(self, time, t0, p, e=0., w=0.):
        if e > self._mine:
            return self._ea_function(time, t0, p, e, w)
        else:
            return self.mean_anomaly(time, t0, p)

    def true_anomaly(self, time, t0, p, e=0., w=0.):
        if e > self._mine:
            return self._ta_function(time, t0, p, e, w)
        else:
            return self.mean_anomaly(time, t0, p)

    def projected_distance(self, time, t0, p, a, i, e=0., w=0.):
        if e > self._mine:
            return self._z_function(time, array([t0, p, a, i, e, w]))
        else:
            return orbits.z_circular(time, array([t0, p, a, i, e, w]))

    def phase(self, time, t0, p, a, i, e=0., w=0.):
        raise NotImplementedError


class CircularOrbit(Orbit):
    def mean_anomaly(self, time, t0, p):
        return ((time - t0) / p) * TWO_PI

    def eccentric_anomaly(self, time, t0, p):
        return self.mean_anomaly(time, t0, p)

    def true_anomaly(self, time, t0, p):
        return self.mean_anomaly(time, t0, p)

    def projected_distance(self, time, t0, p, a, i):
        return orbits.z_circular(time, t0, p, a, i, self.nthr)


#eclipse_shift = orbits.eclipse_shift_ex
#duration_circular = orbits.duration_circular
#duration_eccentric = orbits.duration_eccentric_w
#z_circular = orbits.z_circular
#z_eccentric_newton = orbits.z_eccentric_newton
