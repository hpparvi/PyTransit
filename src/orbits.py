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

from numpy import pi, array
import pytransit.orbits_py as o
from .orbits_py import i_from_ba, i_from_baew, as_from_rhop

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
    td = o.duration_eccentric(p, k, a, i, e, w, 1)
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
    td  = o.duration_eccentric(p, k, a, i, e, w, 1)
    tc += o.eclipse_phase(p, i, e, w)
    folded_time = (time - tc + 0.5*p) % p - 0.5*p
    mask = abs(folded_time) < td_factor*0.5*td
    return time[mask], mask


def not_implemented(*nargs, **kwargs):
    raise NotImplementedError


class Orbit(object):
    methods = 'iteration newton ps3 ps5 interpolation'.split()

    ea_functions = dict(iteration=o.ea_iter_v,
                        newton=o.ea_newton_v,
                        ps3=not_implemented,
                        ps5=not_implemented,
                        interpolation=not_implemented)

    ta_functions = dict(iteration=o.ta_iter_v,
                        newton=o.ta_newton_v,
                        ps3=o.ta_ps3,
                        ps5=o.ta_ps5,
                        interpolation=o.ta_ip)

    z_functions = dict(iteration=o.z_iter_v,
                       newton=o.z_newton_v,
                       ps3=o.z_ps3,
                       ps5=o.z_ps5,
                       interpolation=o.z_iter_v)

    def __init__(self, method='iteration', nthr=0, circular_e_threshold=1e-5):
        assert method in self.methods
        self.nthr = nthr
        self.method = method
        self._mine = circular_e_threshold
        self._ea_function = self.ea_functions[method]
        self._ta_function = self.ta_functions[method]
        self._z_function = self.z_functions[method]

    def mean_anomaly(self, time, t0, p, e=0., w=0.):
        return o.mean_anomaly(time, t0, p, e, w)

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
            return o.z_circular(time, array([t0, p, a, i, e, w]))

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
        return o.z_circular(time, t0, p, a, i)

