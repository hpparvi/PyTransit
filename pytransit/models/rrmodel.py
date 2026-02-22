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
from typing import Tuple, Callable, Union, List, Literal

import jax
import numba
from numpy import ndarray, linspace, isscalar, atleast_1d, sqrt, pi, zeros
from scipy.integrate import trapezoid

from .ldmodel import LDModel
from .transitmodel import TransitModel
from ..backends.numba.limb_darkening import *
from ..backends.numba.limb_darkening.uniform import ldd_uniform
from ..backends.numba.rrmodel import create_z_grid, calculate_weights_3d, rr_simple, rr_simple_and_grad
from ..backends.jax.rrmodel import rr_simple as jax_model

__all__ = ['RoadRunnerModel']


class RoadRunnerModel(TransitModel):
    ldmodels = {'uniform': (ld_uniform, ldd_uniform, ldi_uniform, ldig_uniform),
                'linear': (ld_linear, ldd_linear, ldi_linear, ldig_linear),
                'quadratic': (ld_quadratic, ldd_quadratic, ldi_quadratic, ldig_quadratic),
                'quadratic-tri': (ld_quadratic_tri, ldd_quadratic_tri, ldi_quadratic_tri, ldig_quadratic_tri),
                'nonlinear': (ld_nonlinear, ldd_nonlinear, ldi_nonlinear, ldig_nonlinear),
                'general': (ld_general, ldd_general, ldi_general, ldig_general),
                'power-2': (ld_power_2, ldd_power_2, ldi_power_2, ldig_power_2)}

    def __init__(self, backend: Literal["numba", "jax"] = "numba", return_grad: bool = False,
                 parallel: bool = False, n_threads: int | None = None,
                 ldmodel: Union[str, Callable, Tuple[Callable, Callable]] = 'quadratic',
                 precompute_weights: bool = False, klims: tuple = (0.005, 0.5), nk: int = 256,
                 nzin: int = 20, nzlimb: int = 20, zcut: float = 0.7, ng: int = 100,
                 small_planet_limit: float = 0.05, **kwargs):
        """The RoadRunner transit model by Parviainen (2020).

        Parameters
        ----------
        precompute_weights : bool, optional
            Precompute a 3D weight table for radius ratio values set by `klims`.
        klims : tuple, optional
            Radius ratio limits (kmin, kmax) for the precomputed weight table.
        nk : int, optional
            Radius ratio grid size for the precomputed weight table.
        nzin : int, optional
            Normalized distance grid size for the inner disk.
        nzlimb : int, optional
            Normalized distance grid size for the limb.
        zcut: float, optional
            Normalized distance that separates the stellar disk into an inner disk and limb.
        ng : int, optional
            Size of the grazing value table.
        n_threads: int, optional
            Number of threads to use for the model computation.
        small_planet_limit: float, optional
            The radius ratio limit below which to use a small planet approximation.
        """
        super().__init__(backend, return_grad, parallel, n_threads)

        if backend == "jax":
            self._model = jax.jit(jax_model)
        elif backend == "numba":
            if return_grad:
                self._model = numba.njit(rr_simple_and_grad, parallel=parallel)

            else:
                self._model = numba.njit(rr_simple, parallel=parallel)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.interpolate: bool = precompute_weights
        self.splimit: float = small_planet_limit

        # Set up the limb darkening model
        # --------------------------------
        if isinstance(ldmodel, str):
            try:
                if isinstance(self.ldmodels[ldmodel], tuple):
                    self.ldmodel = self.ldmodels[ldmodel][0]
                    self.ldgrad  = self.ldmodels[ldmodel][1]
                    self.ldmmean = self.ldmodels[ldmodel][2]
                    self.ldigmean = self.ldmodels[ldmodel][3]
            except KeyError:
                print(
                    f"Unknown limb darkening model: {ldmodel}. Choose from [{', '.join(self.ldmodels.keys())}] or supply a callable function.")
                raise
        elif isinstance(ldmodel, LDModel):
            self.ldmodel = ldmodel
            self.ldmmean = ldmodel._integrate
            self.ldgrad = ldmodel._gradient
        elif callable(ldmodel):
            self.ldmodel = ldmodel
            self.ldmmean = None
        elif isinstance(ldmodel, tuple) and callable(ldmodel[0]) and callable(ldmodel[1]):
            self.ldmodel = ldmodel[0]
            self.ldmmean = ldmodel[1]
        else:
            raise NotImplementedError

        # Set the basic variable
        # ----------------------
        self.klims = klims
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut

        # Declare the basic arrays
        # ------------------------
        self.ze = None
        self.zm = None
        self.mu = None
        self.dk = None
        self.dg = None
        self.weights = None

        self._ldmu = linspace(1, 0, 200)
        self._ldz = sqrt(1 - self._ldmu ** 2)

        self.init_integration(nzin, nzlimb, zcut, ng, nk)

    def _init_model(self):
        pass

    def init_integration(self, nzin, nzlimb, zcut, ng, nk):
        self.nk = nk
        self.ng = ng
        self.nzin = nzin
        self.nzlimb = nzlimb
        self.zcut = zcut
        self.ze, self.zm = create_z_grid(zcut, nzin, nzlimb)
        self.mu = sqrt(1 - self.zm ** 2)
        self.dk, self.dg, self.weights = calculate_weights_3d(nk, self.klims[0], self.klims[1], self.ze, ng)

    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, List],
                 t0: Union[float, ndarray], p: Union[float, ndarray], a: Union[float, ndarray],
                 i: Union[float, ndarray], e: Union[float, ndarray] = 0.0, w: Union[float, ndarray] = 0.0) -> ndarray | tuple[ndarray, ndarray]:
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

        npv = 1 if isscalar(p) else p.size
        ldc = atleast_1d(ldc)

        if isinstance(self.ldmodel, LDModel):
            ldp, ldg, ldi = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.return_grad:
                ldg = self.ldgrad(self.mu, atleast_1d(ldc))
            else:
                ldg = None

            if self.ldmmean is not None:
                ldi = evaluate_ldi(self.ldmmean, ldc)
            else:
                ldi = zeros((npv, self.npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(npv):
                    for ipb in range(self.npb):
                        ldi[ipv, ipb] = 2 * pi * trapezoid(self._ldz * ldpi[ipv, ipb], self._ldz)

        if self.return_grad:
            dldi = evaluate_ldig(self.ldigmean, ldc)
            return self._model(self.times, k, t0, p, a, i, e, w,
                                      self.nsamples, self.exptimes, ldp, ldg, ldi, dldi,
                                      self.weights, self.dk, self.klims[0], self.klims[1], self.dg, self.ze)
        else:
            dldi = None
            return self._model(self.times, k, t0, p, a, i, e, w,
                               self.nsamples, self.exptimes, ldp, ldg, ldi, dldi,
                               self.weights, self.dk, self.klims[0], self.klims[1], self.dg, self.ze)