"""Gimenez transit model

   Includes a class offering an easy access to the Fortran implementation of the
   transit model by A. Gimenez (A&A 450, 1231--1237, 2006).

   Author
     Hannu Parviainen <hpparvi@gmail.com>
"""
from gimenez_f import gimenez as g

class Gimenez(object):
    """Gimenez transit model

    Examples
    --------
    Basic case:
      m = Gimenez() # Initialize the model, use quadratic limb darkening law and all available cores
      I = m(z,k,u)  # Evaluate the model for projected distance z, radius ratio k, and limb darkening coefficients u
      
    Use linear interpolation:
      m = Gimenez(lerp=True) # Initialize the model
      I = m(z,k,u)           # Evaluate the model

    Use linear interpolation, two different sets of z:
      m  = Gimenez(lerp=True)      # Initialize the model
      I1 = m(z1,k,u)               # Evaluate the model for z1, update the interpolation table
      I2 = m(z2,k,u, update=False) # Evaluate the model for z2, don't update the interpolation table
    """

    def __init__(self, npol=100, nldc=2, nthr=0, lerp=False):
        """
        Parameters
          npol  int   number of 
          nldc  int   number of limb darkening coefficients (2 == quadratic)
          nthr  int   number of threads
          lerp  bool  use linear interpolation, useful for large light curves
        """
        self._coeff_arr = g.init_arrays(npol, nldc)
        self.npol = npol
        self.nldc = nldc
        self.nthr = nthr

        self._eval = self._eval_lerp if lerp else self._eval_nolerp


    def __call__(self, z, k, u, c=0., b=1e-8, update=True):
        """Evaluate the model

        Parameters
          z    real   projected distance
          k    real   planet to star radius ratio
          u  n*real   limb darkening coefficients
          c    real   contamination (fraction of third light)
          b    real   minimum impact parameter (for the interpolating model)
          update bool update the interpolation table (for the interpolating model)
        """
        return self._eval(z, k, u, c, b, update)


    def _eval_nolerp(self, z, k, u, c, b, update):
        return g.eval(z, k, u, c, self.nthr, *self._coeff_arr)


    def _eval_lerp(self, z, k, u, c, b, update):
        return g.eval_lerp(z, k, u, b, c, self.nthr, update, *self._coeff_arr)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as pl

    z = np.linspace(1e-7,1.3,1000)
    for nldc,ldc,ls in zip([0,1,2], [[], [0], [0.3,0]], ['-','--',':']):
        pl.plot(Gimenez(nldc=nldc, lerp=True)(z, 0.1, ldc), ls=ls,  c='0.0')
    pl.ylim(0.988, 1.001)
    pl.show()
