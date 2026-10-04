#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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

from numpy import asarray, ndarray, pi, sqrt
from numpy.polynomial.legendre import leggauss

__all__ = ['LDModel']


class LDModel:
    """Base class for user-defined limb darkening models.

    A subclass implements `_evaluate` and, to use the model with ``return_grad=True``,
    `_gradient`. The disk-integrated intensity and its gradient are integrated
    numerically from those two by default; override `_integrate` and
    `_integrate_gradient` when they are known analytically. A model that computes
    everything from shared intermediate results can override `__call__` instead.

    The methods receive the limb darkening coefficients `x` as an array of shape
    ``(npv, npb, nldc)``: parameter sets, passbands, and coefficients. `__call__`
    brings the coefficients to that shape, reading a 1D array as one parameter set
    for one passband and a 2D array as ``(npb, nldc)``.

    Parameters
    ----------
    niz
        Number of Gauss-Legendre nodes used by the numerical disk integrals.
    """

    def __init__(self, niz: int = 200):
        # The disk integral 2 pi int_0^1 I z dz equals 2 pi int_0^1 I mu dmu. The
        # integrand is smooth in mu, but has a square-root endpoint in z.
        x, w = leggauss(niz)
        self._int_mu = 0.5 * (x + 1.0)
        self._int_w = 0.5 * w * self._int_mu
        self._int_z = sqrt(1.0 - self._int_mu ** 2)

    def __call__(self, mu: ndarray, x: ndarray, return_grad: bool = False
                 ) -> tuple[ndarray, ndarray | None, ndarray, ndarray | None]:
        """Evaluate the limb darkening model.

        Parameters
        ----------
        mu
            Array of mu (= cos(theta)) values.
        x
            Limb darkening coefficients as a 1D, 2D, or 3D array.
        return_grad
            Whether to also evaluate the derivatives.

        Returns
        -------
        ldp : ndarray
            Intensity profiles, shape ``(npv, npb, nmu)``.
        ldg : ndarray or None
            Profile derivatives, shape ``(npv, npb, 1 + nldc, nmu)``: row 0 is dI/dmu
            and the rest are dI/dc_j. None unless `return_grad` is set.
        ldi : ndarray
            Disk-integrated intensities, shape ``(npv, npb)``.
        dldi : ndarray or None
            Derivatives of the disk-integrated intensities with respect to the
            coefficients, shape ``(npv, npb, nldc)``. None unless `return_grad` is set.
        """
        x = asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape((1, 1, -1))
        elif x.ndim == 2:
            x = x.reshape((1, x.shape[0], -1))
        elif x.ndim != 3:
            raise ValueError("The limb darkening coefficient array must be 1D, 2D, or 3D.")

        ldp = self._evaluate(mu, x)
        ldi = self._integrate(x)
        if return_grad:
            return ldp, self._gradient(mu, x), ldi, self._integrate_gradient(x)
        return ldp, None, ldi, None

    def _evaluate(self, mu: ndarray, x: ndarray) -> ndarray:
        """Intensity profile I(mu), shape ``(npv, npb, nmu)``."""
        raise NotImplementedError

    def _gradient(self, mu: ndarray, x: ndarray) -> ndarray:
        """Profile derivatives ``[dI/dmu, dI/dc_0, ...]``, shape ``(npv, npb, 1 + nldc, nmu)``."""
        raise NotImplementedError

    def _integrate(self, x: ndarray) -> ndarray:
        """Disk-integrated intensity, shape ``(npv, npb)``."""
        return 2.0 * pi * (self._evaluate(self._int_mu, x) * self._int_w).sum(-1)

    def _integrate_gradient(self, x: ndarray) -> ndarray:
        """Derivatives of the disk-integrated intensity, shape ``(npv, npb, nldc)``."""
        return 2.0 * pi * (self._gradient(self._int_mu, x)[:, :, 1:, :] * self._int_w).sum(-1)
