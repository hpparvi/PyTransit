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

"""Astrophysical quantities

 Astrophysical quantities with their error estimates
 ===================================================

 Notes
 -----
  Error estimates follow the form Q_e where Q is the
  quantity in question, and are available only for a
  subset of quantities.

 =============  ============================== ======
 ``au``         astronomical unit                   m
 ``msun``       solar mass                         kg
 ``rsun``       solar radius                        m
 ``dsun``       mean solar mass density        g/cm^3
 ``mjup``       Jupiter mass                       kg
 ``rjup``       volumetric mean Jupiter radius      m
 ``rjup_eq``    equatorial Jupiter radius           m
 ``djup``       mean Jupiter mass density      g/cm^3
 ``mnep``       Neptune mass            radius     kg
 ``rnep``       volumetric mean Neptune radius      m
 ``rnep_eq``    equatorial Neptune radius           m
 ``djup``       mean Neptune mass density      g/cm^3
 ``rearth``     volumetric mean Earth radius        m
 ``rearth_eq``  equatorial Earth radius             m
 =============  ============================== ======
"""

from __future__ import division

d_h = 24.
d_m = 60 * d_h
d_s = 60 * d_m

au,   au_e             = 1.496e11, 0.0       
msun, msun_e           = 1.9891e30, 0.0      
rsun, rsun_e           = 0.5*1.392684e9, 0.0
dsun, dsun_e           = 1.408, 0.0
mjup, mjup_e           = 1.89896e27, 0.0     
rjup, rjup_e           = 6.9911e7, 0.0       
rjup_eq, rjup_eq_e     = 7.1492e7, 0.0
djup, djup_e           = 1.326, 0.0       
mnep, mnep_e           = 1.0242e26, 0.0
rnep, mnep_e           = 2.4622e7, 0.0
rnep_eq, mnep_eq_e     = 2.4764e7, 0.0
dnep, dnep_e           = 1.638, 0.0       
mearth, mearth_e       = 5.9726e24, 0.0
rearth, rearth_e       = 6.371e6, 0.0
rearth_eq, rearth_eq_e = 6.3781e6, 0.0
dearth, dearth_e       = 5.514, 0.0       
