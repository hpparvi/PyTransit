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

from numpy import ndarray
from .filter import Filter, ClearFilter


class Instrument:
    def __init__(self, name, filters, qes=None):
        self.name = name

        assert all([isinstance(f, Filter) for f in filters]), "All filters must be Filter instances."
        self.filters = filters
        self.pb_n = npb = len(filters)
        self.pb_names = [f.name for f in self.filters]

        if qes is not None:
            if isinstance(qes, (tuple, list, ndarray)):
                assert len(filters) == len(qes), "Number of QE profiles differs from the number of passbands."
                assert all([isinstance(qe, Filter) for qe in qes]), "All QE profiles must be Filter instances."
                self.qes = qes
            elif isinstance(qes, Filter):
                self.qes = npb * [qes]
        else:
            self.qes = npb * [ClearFilter('QE')]