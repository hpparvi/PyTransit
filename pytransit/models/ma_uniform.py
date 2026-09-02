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

from typing import Union
from numpy import ndarray, squeeze, zeros, asarray, isscalar
from .numba.ma_uniform_nb import uniform_model_v, uniform_model_s
from ._deprecation import deprecated_evaluation_method
from .transitmodel import TransitModel

__all__ = ['UniformModel']

npfloat = Union[float, ndarray]

class UniformModel(TransitModel):
    """Transit over a uniform (unlimb-darkened) stellar disk (Mandel & Agol, ApJ 580, L171-L175, 2002).

    The uniform model gives the flux blocked by an opaque circular planet crossing a disk of
    constant surface brightness. Because there is no limb darkening, the model takes no limb
    darkening coefficients and reduces to the analytic overlap area of two circles, which makes
    it the fastest model in PyTransit.

    The model is used for two things in practice:

    - **Secondary eclipses**, where the planet's dayside is occulted by the star. The planet
      disk really is close to uniform in this geometry, so the model is physically correct
      rather than merely convenient. Pass ``eclipse=True`` to shift the modelled event to the
      secondary eclipse.
    - **Transits where limb darkening is negligible or irrelevant**, such as very low-precision
      photometry or transit-search applications where speed matters more than a ppm-accurate
      shape.

    Examples
    --------
    ::

        from pytransit import UniformModel

        tm = UniformModel()
        tm.set_data(time)
        flux = tm.evaluate(k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi)

    See Also
    --------
    EclipseModel : secondary eclipse model with an explicit planet-star flux ratio.
    UniformModelCL : OpenCL implementation of the same model.
    """

    def __init__(self, eclipse: bool = False) -> None:
        """
        Parameters
        ----------
        eclipse : bool, optional
            If `True`, the model reproduces a secondary eclipse (an occultation of the planet
            by the star) rather than a transit. Defaults to `False`.
        """
        super().__init__()
        self.is_eclipse = eclipse
        self._zsign = -1.0 if self.is_eclipse else 1.0

    def evaluate(self, k: npfloat, t0: npfloat, p: npfloat, a: npfloat, i: npfloat, e: npfloat = None, w: npfloat = None,
                 copy: bool = True) -> ndarray:
        """Evaluates the uniform transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float, 1D vector, or 2D array.
        t0
            Transit center(s) as a float or a 1D vector.
        p
            Orbital period(s) as a float or a 1D vector.
        a
            Orbital semi-major axis (axes) divided by the stellar radius as a float or a 1D vector.
        i
            Orbital inclination(s) as a float or a 1D vector.
        e
            Orbital eccentricity as a float or a 1D vector.
        w
            Argument of periastron as a float or a 1D vector.
        copy

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters simultaneously.
        The orbital parameters can be given either as a float or a 1D array-like (preferably ndarray for optimal speed.)

        Returns
        -------
        Transit model
        """

        k = asarray(k)

        # Scalar parameters branch
        # ------------------------
        if isscalar(p):
            e = 0. if e is None else e
            w = 0. if w is None else w
            flux = uniform_model_s(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples,
                                   self.exptimes, zsign=self._zsign)

        # Parameter population branch
        # ---------------------------
        else:
            npv = t0.size
            e = zeros(npv) if e is None else asarray(e)
            w = zeros(npv) if w is None else asarray(w)

            if k.ndim == 1:
                k = k.reshape((k.size,1))

            flux = uniform_model_v(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples, self.exptimes, zsign=self._zsign)
        return squeeze(flux)

    @deprecated_evaluation_method()
    def evaluate_ps(self, k: float, t0: float, p: float, a: float, i: float, e: float = 0., w: float = 0.) -> ndarray:
        """Evaluate the transit model for a set of scalar parameters.

         Parameters
         ----------
         k : array-like
             Radius ratio(s) either as a single float or an 1D array.
         t0 : float
             Transit center as a float.
         p : float
             Orbital period as a float.
         a : float
             Orbital semi-major axis divided by the stellar radius as a float.
         i : float
             Orbital inclination(s) as a float.
         e : float, optional
             Orbital eccentricity as a float.
         w : float, optional
             Argument of periastron as a float.

         Notes
         -----
         This version of the `evaluate` method is optimized for calculating a single transit model (such as when using a
         local optimizer). If you want to evaluate the model for a large number of parameters simultaneously, use either
         `evaluate` or `evaluate_pv`.

         Returns
         -------
         ndarray
             Modelled flux as a 1D ndarray.
         """
        if self.time is None:
            raise ValueError("Need to set the data before calling the transit model.")

        k = asarray(k)
        flux = uniform_model_s(self.time, k, t0, p, a, i, e, w, self.lcids, self.pbids, self.nsamples, self.exptimes, zsign=self._zsign)
        return squeeze(flux)
