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

import numpy as np
from scipy.constants import c, k, h
from scipy.optimize import fmin

def fold(time, period, origo=0.0, shift=0.0, normalize=True,  clip_range=None):
    """Folds the given data over a given period.

    Parameters
    ----------
    
      time        
      period      
      origo       
      shift       
      normalize   
      clip_range  

    Returns
    -------

      phase       
    """
    tf = ((time - origo)/period + shift) % 1.

    if not normalize:
        tf *= period
        
    if clip_range is not None:
        mask = np.logical_and(clip_range[0]<tf, tf<clip_range[1])
        tf = tf[mask], mask
    return tf


def planck(T, wl):
    """Radiance as a function or black-body temperature and wavelength.

    Parameters
    ----------

      T   : Temperature  [K]
      wl  : Wavelength   [m]

    Returns
    -------

      B   : Radiance
    """
    return 2*h*c**2/wl**5 / (np.exp(h*c/(wl*k*T))-1)


def contamination_bb(c1, T, wl1, wl2):
    """Contamination from a third object radiating as a black-body given a contamination estimate in a reference wavelength. 

    Parameters
    ----------

      c1   : Contamination in the reference wavelength [-]
      T    : Temperature                               [K]
      wl1  : Reference wavelength                      [m]
      wl2  : Target wavelength                         [m]

    Returns
    -------

      c2  : Contamination in the given wavelength      [-]
    """
    B1   = planck(T, wl1)
    B2   = planck(T, wl2)

    return  c1*(B2/B1)


def nonlinear_ld_to_general_ld(ldc):
    def nl(mu,ld):
        return 1. - np.sum([ld[i-1]*(1.-mu**(0.5*i)) for i in range(1,5)], axis=0)
    def gn(mu,ld):
        return 1. - np.sum([ld[i-1]*(1.-mu**i) for i in range(1,5)], axis=0)

    mu = np.linspace(0,1,200)

    pvg = fmin(lambda pv:sum((Inl-gn(mu,pv))**2), ld[::2])
