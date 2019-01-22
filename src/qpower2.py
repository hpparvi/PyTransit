# -*- coding: utf-8 -*-
#
# Copyright (C) 2010--2019  Hannu Parviainen
#
# This file is part of PyTransit
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""QPower2 transit model.

This module offers the QPower2 class implementing the transit model by Maxted & Gill (ArXIV:1812.01606).
"""

from pytransit.qpower2_py import qpower2
from pytransit.tm import TransitModel

from numpy import select, pi

class QPower2(TransitModel):
    """QPower2 transit model (Maxted & Gill, ArXIV:1812.01606).
    """

    def __init__(self, nthr=1, interpolate=False, supersampling=1, exptime=0.020433598, eclipse=False, **kwargs):
        """Initialise the model.

        Args:
            nldc: Number of limb darkening coefficients, can be either 0 (no limb darkening) or 2 (quadratic limb darkening).
            nthr: Number of threads (default = 0).
            interpolate: If True, evaluates the model using interpolation (default = False).
            supersampling: Number of subsamples to calculate for each light curve point (default=0).
            exptime: Integration time for a single exposure, used in supersampling default=(0.02).
            eclipse: If True, evaluates the model for eclipses. If false, eclipses are filtered out (default = False).

        """
        super().__init__(2, nthr, interpolate, supersampling, exptime, eclipse, **kwargs)
        self.interpolate = interpolate or kwargs.get('lerp', False)

    def _eval(self, z, k, u, c=0.0):
        """Wraps the Numba implementation of a transit over a chromosphere

            Args:
                z: Array of normalised projected distances.
                k: Planet to star radius ratio.
                u: Array of limb darkening coefficients
                c: Array of contamination values as [c1, c2, ... c_npb].

            Returns:
                An array of model flux values for each z.
        """
        return qpower2(z, k, u, c)


    def __call__(self, z, k, u, c=0.0):
        """Evaluates the model for the given z, k, and u.

        Evaluates the transit model given an array of normalised distances, a radius ratio, and
        a set of limb darkening coefficients.

        Args:
            z: Array of normalised projected distances.
            k: Planet to star radius ratio.
            u: Array of limb darkening coefficients.
            c: Contamination factor (fraction of third light), optional.

        Returns:
            An array of model flux values for each z.
        """
        return qpower2(z, k, u, c)

