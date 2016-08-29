## PyTransit
## Copyright (C) 2010--2015  Hannu Parviainen
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

"""Fast and easy exoplanet transit modelling in Python

This package aims to offer a simple-to-use Python interfaces for the exoplanet
transit models by Mandel & Agol (2002) and A. Gimenez (2006) implemented in
Fortran. 

Author
  Hannu Parviainen  <hannu.parviainen@physics.ox.ac.uk>

Date
  28.05.2015

"""

from .gimenez import Gimenez
from .mandelagol import MandelAgol
from .orbits import *

__all__ = 'Gimenez MandelAgol orbits z_circular z_eccentric_newton'.split()
