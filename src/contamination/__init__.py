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

from numpy import sqrt
from .instrument import Instrument
from .contamination import SMContamination, BBContamination
from .filter import ClearFilter, BoxcarFilter, TabulatedFilter


def true_radius_ratio(apparent_k, contamination):
    return apparent_k / sqrt(1 - contamination)

def apparent_radius_ratio(true_k, contamination):
    return true_k * sqrt(1 - contamination)

__all__ = "SMContamination BBContamination Instrument ClearFilter BoxcarFilter TabulatedFilter true_radius_ratio apparent_radius_ratio".split()