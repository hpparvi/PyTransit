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
from typing import Tuple, Callable, Union, List, Optional

from numba import njit
from numpy import ndarray, array, squeeze, atleast_2d, atleast_1d, zeros, asarray, linspace, sqrt, pi, ones, log, exp
from scipy.integrate import trapz

from pytransit.models.numba.ptmodel import pt_model_direct_s, pt_model_direct_s_noparallel

from .numba.ptmodel import create_z_grid, calculate_weights_3d
from .transitmodel import TransitModel

__all__ = ['PTModel']

@njit(fastmath=True)
def ld_uniform(mu, pv):
    return ones(mu.size)

@njit(fastmath=True)
def ldi_uniform(pv):
    return pi

@njit(fastmath=True)
def ld_linear(mu, pv):
    return 1. - pv[0]*(1.-mu)

@njit(fastmath=True)
def ldi_linear(pv):
    return 2*pi * 1/6*(3-2*pv[0])

@njit(fastmath=True)
def ld_quadratic(mu, pv):
    return 1. - pv[0]*(1.-mu) - pv[1]*(1.-mu)**2

@njit(fastmath=True)
def ldi_quadratic(pv):
    return 2*pi * 1/12*(-2*pv[0]-pv[1]+6)

@njit(fastmath=True)
def ld_quadratic_tri(mu, pv):
    a, b = sqrt(pv[0]), 2*pv[1]
    u, v = a * b, a * (1. - b)
    return 1. - u*(1.-mu) - v*(1.-mu)**2

@njit(fastmath=True)
def ld_nonlinear(mu, pv):
    return 1. - pv[0]*(1.-sqrt(mu)) - pv[1]*(1.-mu) - pv[2]*(1.-pow(mu, 1.5)) - pv[3]*(1.-mu**2)

@njit(fastmath=True)
def ld_general(mu, pv):
    ldp = zeros(mu.size)
    for i in range(pv.size):
        ldp += pv[i]*(1.0-mu**(i+1))
    return ldp

@njit(fastmath=True)
def ld_square_root(mu, pv):
    return 1. - pv[0]*(1.-mu) - pv[1]*(1.-sqrt(mu))

@njit(fastmath=True)
def ld_logarithmic(mu, pv):
    return 1. - pv[0]*(1.-mu) - pv[1]*mu*log(mu)

@njit(fastmath=True)
def ld_exponential(mu, pv):
    return 1. - pv[0]*(1.-mu) - pv[1]/(1.-exp(mu))

@njit(fastmath=True)
def ld_power2(mu, pv):
    return 1. - pv[0]*(1.-mu**pv[1])

class PTModel(TransitModel):
    ldmodels = {'uniform': (ld_uniform, ldi_uniform),
                'linear': (ld_linear, ldi_linear),
                'quadratic': (ld_quadratic, ldi_quadratic),
                'quadratic_tri': ld_quadratic_tri,
                'nonlinear': ld_nonlinear,
                'general': ld_general,
                'square_root': ld_square_root,
                'logarithmic': ld_logarithmic,
                'exponential': ld_exponential,
                'power2': ld_power2}

    def __init__(self, ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 interpolate: bool = True, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut=0.7, ng: int = 50, parallel: bool = True):
        """The ridiculously fast transit model by Parviainen (2020).

        Parameters
        ----------
        interpolate : bool, optional
            Use the interpolation method presented in Parviainen (2015) if true.
        klims : tuple, optional
            Radius ratio limits (kmin, kmax) for the interpolated model.
        nk : int, optional
            Radius ratio grid size for the interpolated model.
        nz : int, optional
            Normalized distance grid size for the interpolated model.
        """
        super().__init__()
        self.interpolate = interpolate

        self.parallel = parallel
        if self.parallel:
            self._m_direct_s = pt_model_direct_s
        else:
            self._m_direct_s = pt_model_direct_s_noparallel

        if isinstance(ldmodel, str):
            try:
                if isinstance(self.ldmodels[ldmodel], tuple):
                    self.ldmodel = self.ldmodels[ldmodel][0]
                    self.ldmmean = self.ldmodels[ldmodel][1]
                else:
                    self.ldmodel = self.ldmodels[ldmodel]
                    self.ldmmean = None
            except KeyError:
                print(
                    f"Unknown limb darkening model: {ldmodel}. Choose from [{', '.join(self.ldmodels.keys())}] or supply a callable function.")
                raise

        self.klims = klims
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut

        self.ze = None
        self.zm = None
        self.mu = None
        self.dk = None
        self.dg = None
        self.weights = None

        self._ldmu = linspace(1, 0, 200)
        self._ldz = sqrt(1 - self._ldmu ** 2)

        self.init_integration(nzin, nzlimb, zcut, ng, nk)

    def init_integration(self, nzin, nzlimb, zcut, ng, nk=None):
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut
        self.ze, self.zm = create_z_grid(zcut, nzin, nzlimb)
        self.mu = sqrt(1 - self.zm ** 2)
        if self.interpolate:
            self.dk, self.dg, self.weights = calculate_weights_3d(nk, self.klims[0], self.klims[1], self.ze, ng)

    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, List], t0: Union[float, ndarray],
                 p: Union[float, ndarray],
                 a: Union[float, ndarray], i: Union[float, ndarray], e: Optional[Union[float, ndarray]] = None,
                 w: Optional[Union[float, ndarray]] = None, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float, 1D vector, or 2D array.
        ldc
            Limb darkening coefficients as a 1D or 2D array.
        t0
            Transit center(s) as a float or a 1D vector.
        p
            Orbital period(s) as a float or a 1D vector.
        a
            Orbital semi-major axis (axes) divided by the stellar radius as a float or a 1D vector.
        i
            Orbital inclination(s) as a float or a 1D vector.
        e : optional
            Orbital eccentricity as a float or a 1D vector.
        w : optional
            Argument of periastron as a float or a 1D vector.

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters simultaneously. In
        the first case, the orbital parameters should all be given as floats. In the second case, the orbital parameters
        should be given as a 1D array-like.

        Returns
        -------
        ndarray
            Modelled flux either as a 1D or 2D ndarray.
        """

        # Scalar parameters branch
        # ------------------------
        if isinstance(t0, float):

            ldp = self.ldmodel(self.mu, ldc)
            if self.ldmmean is not None:
                istar = self.ldmmean(ldc)
            else:
                istar = 2 * trapz(self._ldz * self.ldmodel(self._ldmu, ldc), self._ldz)

            if e is None:
                e, w = 0.0, 0.0

            return self._m_direct_s(self.time, asarray(k), t0, p, a, i, e, w, ldp, istar, self.ze, self.ng, self.lcids,
                                    self.pbids,
                                    self.nsamples, self.exptimes, self._es, self._ms, self._tae)
