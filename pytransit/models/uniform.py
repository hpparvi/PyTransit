from math import nan, isfinite
from typing import Union, Optional, Tuple

from numba import njit
from numpy import ndarray, squeeze, zeros, asarray, isscalar, atleast_1d, ones_like, arctan2, array, sqrt

from meepmeep.xy.par_fitting import coeffs as coeffs_fit
from meepmeep.xy.par_direct import coeffs as coeffs_dir

from .numba.udmodel import uniform_model_v
from .transitmodel import TransitModel


__all__ = ['UniformDiskModel']

sov = Union[float, ndarray]


@njit
def umdir(time, k, t0, p, a, i, e, w, derivatives, lcids, pbids, epids, nsamples, exptimes):
    k = atleast_1d(array(k))
    dkdp = ones_like(k)
    cfs, dcfs = coeffs_dir(0.0, p, a, i, e, w, derivatives)
    flux, dflux = uniform_model_v(time, k, t0, p, dkdp, cfs, dcfs, derivatives,
                                  lcids, pbids, epids, nsamples, exptimes)
    return flux, dflux


@njit
def umfit(time, k2, t0, p, rho, b, secw, sesw, derivatives, lcids, pbids, epids, nsamples, exptimes):
    k = sqrt(k2)
    dkdp = 0.5/k
    cfs, dcfs = coeffs_fit(0.0, p, rho, b, secw, sesw, derivatives)
    flux, dflux = uniform_model_v(time, k, t0, p, dkdp, cfs, dcfs, derivatives,
                                  lcids, pbids, epids, nsamples, exptimes)
    return flux, dflux


class UniformDiskModel(TransitModel):
    parametrizations = ('direct', 'fitting')

    def __init__(self, parametrization: str = 'direct') -> None:
        if parametrization not in self.parametrizations:
            raise ValueError(f'Parametrization needs to be one of {self.parametrizations}.')

        super().__init__()

        if parametrization == 'direct':
            self.evaluate = self.evaluate_direct
        else:
            self.evaluate = self.evaluate_fitting

    def evaluate_direct(self, k: sov, t0: sov, p: sov, a: sov, i: sov,
                        e: Optional[sov] = None, w: Optional[sov] = None,
                        return_derivatives: bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Evaluates the uniform transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Planet-star radius ratio either as a single float, 1D vector, or 2D array.
        t0
            Transit center [d] as a float or a 1D vector.
        p
            Orbital period [d] as a float or a 1D vector.
        a
            Orbital semi-major axis [R_Star] as a float or a 1D vector.
        i
            Orbital inclination [rad] as a float or a 1D vector.
        e
            Orbital eccentricity as a float or a 1D vector.
        w
            Argument of periastron [rad] as a float or a 1D vector.
        return_derivatives
            Returns the flux derivatives if `True`

        Returns
        -------
        Normalized flux
        """

        e = 0. if e is None else e
        w = 0. if w is None else w

        flux, dflux = umdir(self.time, k, t0, p, a, i, e, w, return_derivatives,
                            self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes)

        if return_derivatives:
            return squeeze(flux), squeeze(dflux)
        else:
            return squeeze(flux)

    def evaluate_fitting(self, k2: sov, t0: sov, p: sov, rho: sov, b: sov,
                         secw: Optional[sov] = None, sesw: Optional[sov] = None,
                         return_derivatives: bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Evaluates the uniform transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k2
            Planet-star area ratio as a single float, 1D vector, or 2D array.
        t0
            Transit center [d] as a float or a 1D vector.
        p
            Orbital period [d] as a float or a 1D vector.
        rho
            Stellar density [g/cm^3] as a float or a 1D vector.
        b
            Impact parameter as a float or a 1D vector.
        secw
            sqrt(e) cos(w) as a float or a 1D vector.
        sesw
            sqrt(e) sin(w) as a float or a 1D vector.
        return_derivatives
            Returns the flux derivatives if `True`

        Returns
        -------
        Normalized flux
        """

        secw = 0. if secw is None else secw
        sesw = 0. if sesw is None else sesw

        flux, dflux = umfit(self.time, k2, t0, p, rho, b, secw, sesw, return_derivatives,
                            self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes)

        if return_derivatives:
            return squeeze(flux), squeeze(dflux)
        else:
            return squeeze(flux)
