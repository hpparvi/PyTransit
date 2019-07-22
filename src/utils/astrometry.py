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

from ..orbits.orbits_py import a_from_mp
from .constants import msun, mjup

as_c = {'as':1., 'mas':1e3, 'muas':1e6}

def angular_signal(mass, period, distance, mstar=1., units='as'):
    """Angular semi-amplitude of the astrometric signal caused by a planet.
    
    Parameters
    ----------

      mass     : planet mass            [M_Jup]
      period   : orbital period         [d]
      distance : distance to the system [pc]
      mstar    : stellar mass           [M_Sun]
      units    : 'as', 'mas', or 'muas'

    Returns
    -------

      angular semi-amplitude of the astrometric signal by the planet in given units

    """
    return as_c[units] * (mass*mjup)/(mstar*msun) * a_from_mp(mstar, period) / distance
