#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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

from numpy import pi, array
from .orbits_py import mean_anomaly, duration_eccentric, eclipse_phase, ea_iter_v, ea_newton_v, z_ps5, z_newton_v, \
    z_ps3, z_iter_v, ta_ip, ta_ps5, ta_ps3, ta_newton_v, ta_iter_v, z_circular

TWO_PI = 2 * pi


def extract_time_transit(time, k, tc, p, a, i, e, w, td_factor=1.1):
    """ Extracts the in-transit times from a light curve.

    Parameters
    ----------
    time
    k
    tc
    p
    a
    i
    e
    w
    td_factor

    Returns
    -------

    """
    td = duration_eccentric(p, k, a, i, e, w, 1)
    folded_time = (time - tc + 0.5*p) % p - 0.5*p
    mask = abs(folded_time) < td_factor*0.5*td
    return time[mask], mask


def extract_time_eclipse(time, k, tc, p, a, i, e, w, td_factor=1.1):
    """ Extracts the in-eclipse times from a light curve.

    Parameters
    ----------
    time
    k
    tc
    p
    a
    i
    e
    w
    td_factor

    Returns
    -------

    """
    td  = duration_eccentric(p, k, a, i, e, w, 1)
    tc += eclipse_phase(p, i, e, w)
    folded_time = (time - tc + 0.5*p) % p - 0.5*p
    mask = abs(folded_time) < td_factor*0.5*td
    return time[mask], mask


def not_implemented(*nargs, **kwargs):
    raise NotImplementedError


class Orbit(object):
    methods = 'iteration newton ps3 ps5 interpolation'.split()

    ea_functions = dict(iteration=ea_iter_v,
                        newton=ea_newton_v,
                        ps3=not_implemented,
                        ps5=not_implemented,
                        interpolation=not_implemented)

    ta_functions = dict(iteration=ta_iter_v,
                        newton=ta_newton_v,
                        ps3=ta_ps3,
                        ps5=ta_ps5,
                        interpolation=ta_ip)

    z_functions = dict(iteration=z_iter_v,
                       newton=z_newton_v,
                       ps3=z_ps3,
                       ps5=z_ps5,
                       interpolation=z_iter_v)

    def __init__(self, method='iteration', circular_e_threshold=1e-2):
        assert method in self.methods
        self.method = method
        self._mine = circular_e_threshold
        self._ea_function = self.ea_functions[method]
        self._ta_function = self.ta_functions[method]
        self._z_function = self.z_functions[method]

    def mean_anomaly(self, time, t0, p, e=0., w=0.):
        return mean_anomaly(time, t0, p, e, w)

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
            return z_circular(time, array([t0, p, a, i, e, w]))

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
        return z_circular(time, t0, p, a, i)
