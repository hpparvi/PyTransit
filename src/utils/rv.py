from __future__ import division

from numpy import pi, sin, sqrt
from scipy.constants import G

def mp_from_kiepms(K, i, e, p, Ms):
    "Calculates the planet's mass from the fitted parameters"
    return K * ((p*24*3600)/(2*pi*G))**(1/3) * Ms**(2/3)/sin(i) * sqrt((1-e**2))
