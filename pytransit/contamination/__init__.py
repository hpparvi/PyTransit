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

"""Module to model flux contamination in transit light curves.
"""

from numpy import sqrt
from .instrument import Instrument
from .contamination import SMContamination, BBContamination, contaminate_light_curve
from .filter import ClearFilter, BoxcarFilter, TabulatedFilter, sdss_z, sdss_i, sdss_r, sdss_g


def true_radius_ratio(apparent_k: float, contamination: float) -> float:
    return apparent_k / sqrt(1 - contamination)


def apparent_radius_ratio(true_k: float, contamination: float) -> float:
    return true_k * sqrt(1 - contamination)

sdss_griz = (sdss_g, sdss_r, sdss_i, sdss_z)

__all__ = ("SMContamination BBContamination Instrument ClearFilter BoxcarFilter TabulatedFilter".split() +
           "true_radius_ratio apparent_radius_ratio sdss_z sdss_i sdss_r sdss_g sdss_griz".split() +
           ["contaminate_light_curve"])
