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
from typing import Union, List, Literal, Callable, Tuple

import numba
from numpy import ndarray, linspace, isscalar, atleast_1d, atleast_2d, sqrt, pi, zeros, array
from scipy.integrate import trapezoid

from .ldmodel import LDModel
from .transitmodel import TransitModel
from ..backends.numba.limb_darkening import *
from ..backends.numba.limb_darkening.uniform import ldd_uniform
from ..backends.numba.tsmodel import tsmodel, tsmodel_and_grad
from ..backends.numba.rrmodel import create_z_grid, calculate_weights_3d

__all__ = ['TransmissionSpectroscopyModel']


def evaluate_ldg(ldgrad_fn, mu, ldc):
    """Evaluate LD gradient profiles for all parameter vectors and passbands.

    Parameters
    ----------
    ldgrad_fn : callable
        LD gradient function (e.g. ldd_quadratic), signature (mu, pv) -> (1+nldc, nmu).
    mu : ndarray, shape (nmu,)
    ldc : ndarray, shape (npv, npb, nldc)

    Returns
    -------
    ldg : ndarray, shape (npv, npb, 1+nldc, nmu)
    """
    npv, npb, nldc = ldc.shape
    nmu = mu.size
    sample = ldgrad_fn(mu, ldc[0, 0])
    nrows = sample.shape[0]
    ldg = zeros((npv, npb, nrows, nmu))
    for ipv in range(npv):
        for ipb in range(npb):
            ldg[ipv, ipb, :, :] = ldgrad_fn(mu, ldc[ipv, ipb])
    return ldg


def evaluate_distar(ldig_fn, ldc):
    """Evaluate disk-integrated intensity derivatives for all parameter vectors and passbands.

    Parameters
    ----------
    ldig_fn : callable
        LD disk-integrated gradient function (e.g. ldig_quadratic), signature (pv) -> (nldc,).
    ldc : ndarray, shape (npv, npb, nldc)

    Returns
    -------
    distar : ndarray, shape (npv, npb, nldc)
    """
    npv, npb, nldc = ldc.shape
    distar = zeros((npv, npb, nldc))
    for ipv in range(npv):
        for ipb in range(npb):
            distar[ipv, ipb, :] = ldig_fn(ldc[ipv, ipb])
    return distar


class TransmissionSpectroscopyModel(TransitModel):
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
                 nzin: int = 20, nzlimb: int = 20, zcut: float = 0.7, ng: int = 100, **kwargs):
        super().__init__(backend, return_grad, parallel, n_threads)

        if backend == "jax":
            raise NotImplementedError("JAX backend not yet implemented for TransmissionSpectroscopyModel")
        elif backend == "numba":
            if parallel:
                if return_grad:
                    self._model = numba.njit(tsmodel_and_grad.py_func, parallel=True)
                else:
                    self._model = numba.njit(tsmodel.py_func, parallel=True)
            else:
                if return_grad:
                    self._model = tsmodel_and_grad
                else:
                    self._model = tsmodel
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.interpolate: bool = precompute_weights

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

        # Set the basic variables
        # -----------------------
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
                 i: Union[float, ndarray], e: Union[float, ndarray] = 0.0,
                 w: Union[float, ndarray] = 0.0) -> ndarray | tuple[ndarray, ndarray]:
        """Evaluate the transmission spectroscopy model.

        Parameters
        ----------
        k
            Radius ratio(s) as a float, 1D, or 2D array with shape (npv, npb).
        ldc
            Limb darkening coefficients as a 1D, 2D, or 3D array with shape (npv, npb, nldc).
        t0
            Transit center(s).
        p
            Orbital period(s).
        a
            Scaled semi-major axis (a/Rs).
        i
            Orbital inclination(s) in radians.
        e : optional
            Orbital eccentricity.
        w : optional
            Argument of periastron.

        Returns
        -------
        ndarray or tuple
            Model flux (npv, npb, npt), or (flux, dflux) if return_grad=True.
        """
        k = atleast_2d(k)
        t0, p, a, i, e, w = map(atleast_1d, (t0, p, a, i, e, w))
        npv = k.shape[0]

        # Reshape ldc to 3D (npv, npb, nldc)
        # -----------------------------------
        ldc = array(ldc)
        if ldc.ndim == 1:
            ldc = ldc.reshape((1, 1, ldc.shape[0]))
        elif ldc.ndim == 2:
            ldc = ldc.reshape((1, ldc.shape[0], ldc.shape[1]))
        elif ldc.ndim == 3:
            pass
        else:
            raise ValueError("ldc must be 1D, 2D, or 3D")

        self.npb = npb = ldc.shape[1]

        # Limb darkening profiles
        # -----------------------
        if isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((npv, npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(npv):
                    for ipb in range(npb):
                        istar[ipv, ipb] = 2 * pi * trapezoid(self._ldz * ldpi[ipv, ipb], self._ldz)

        # Weight table
        # ------------
        if self.interpolate:
            dk, dg, weights = self.dk, self.dg, self.weights
        else:
            dk, dg, weights = None, None, None

        if self.return_grad:
            ldg = evaluate_ldg(self.ldgrad, self.mu, ldc)
            distar = evaluate_distar(self.ldigmean, ldc)
            return self._model(self.times, k, t0, p, a, i, e, w,
                               self.nsamples, self.exptimes,
                               ldp, ldg, istar, distar,
                               weights, dk, self.klims[0], self.klims[1], self.ng, dg, self.ze)
        else:
            return self._model(self.times, k, t0, p, a, i, e, w,
                               self.nsamples, self.exptimes,
                               ldp, istar,
                               weights, dk, self.klims[0], self.klims[1], self.ng, dg, self.ze)

    def __call__(self, k, ldc, t0, p, a, i, e=0.0, w=0.0):
        return self.evaluate(k, ldc, t0, p, a, i, e, w)
