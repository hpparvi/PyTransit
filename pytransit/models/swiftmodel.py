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
from typing import Union, Callable, Tuple

from deprecated import deprecated

from .rrmodel import RoadRunnerModel

__all__ = ['SwiftModel']

class SwiftModel(RoadRunnerModel):

    @deprecated(version='2.3.0', reason='SwiftModel has been renamed to RoadRunnnerModel, please use it instead.')
    def __init__(self, ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 interpolate: bool = False, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut: float = 0.7, ng: int = 50,
                 parallel: bool = False, small_planet_limit: float = 0.01):

        super().__init__(ldmodel, interpolate, klims, nk, nzin, nzlimb, zcut, ng, parallel,
                 small_planet_limit)
    pass