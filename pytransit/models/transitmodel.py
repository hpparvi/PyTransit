#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
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
import jax

from typing import Literal, Callable

from numpy import (ones, ndarray, asarray, concatenate, log, ndim, pi, squeeze, zeros, unique,
                   atleast_1d, issubdtype, integer, float64, array, full, isscalar)

from ._utils import _npv_from_k, _param_is_expanded


class TransitModel:
    """Exoplanet transit light curve model.
    """
    def __init__(self, backend: Literal["numba", "jax", "opencl"] = "numba", return_grad: bool = False,
                 parallel: bool = False, n_threads: int | None = None, **kwargs):
        self.backend = backend
        self.return_grad = return_grad
        self.n_threads = n_threads
        self.parallel = parallel

        self.times = None
        self.lcids = None
        self.pbids = None
        self.epids = None
        self.nsamples = None
        self.exptimes = None

        self.nlc: int = 0
        self.npt: int = 0
        self.npb: int = 0
        self.ntc: int = 0
        self.nor: int = 0

        self.simple = True
        self._time_id: int | None = None
        self._err_broadcast: tuple | None = None
        self._kwargs = kwargs

        self._init_model()

    def _init_model(self):
        raise NotImplementedError

    def set_data(self,
                 times: ndarray | list,
                 lcids: ndarray | list | None = None,
                 pbids: ndarray | list | None = None,
                 epids: ndarray | list | None = None,
                 nsamples: ndarray | list | None  = None,
                 exptimes: ndarray | list | None = None,
                 include_orbit_variations: bool = False) -> None:

        if id(times) == self._time_id and lcids is None and pbids is None and nsamples is None and exptimes is None and epids is None:
            return

        self._time_id = id(times)
        self.times    = array(times, float64)
        self.npt      = self.times.size

        # Light curve indices
        # -------------------
        # The light curve a datapoint belongs to.
        self.lcids    = asarray(lcids) if lcids is not None else zeros(self.npt, 'int')
        self.nlc = unique(self.lcids).size

        if not issubdtype(self.lcids.dtype, integer):
            raise ValueError(f"The light curve indices must be given as integers instead of {self.lcids.dtype}.")

        if self.lcids.size != self.npt:
            raise ValueError(f"Light curve index array size ({self.lcids.size}) should equal to the number of datapoints ({self.npt}).")

        # Passband indices
        # ----------------
        self.pbids = asarray(pbids) if pbids is not None else zeros(self.nlc, 'int')
        self.npb = unique(self.pbids).size

        if not issubdtype(self.pbids.dtype, integer):
            raise ValueError(f"The passband indices must be given as integers instead of {self.pbids.dtype}.")

        if self.pbids.size != self.nlc:
            raise ValueError(f"Passband index array size ({self.pbids.size}) should equal to the number of light curves ({self.nlc}).")

        if not (self.pbids.max() == (self.npb-1) and self.pbids.min() == 0):
            raise ValueError(f"Passband indices (`pbids`) for {self.npb} unique passbands should be given as integers between 0 and {self.npb - 1}.")

        # Epoch indices
        # -------------
        self.epids = asarray(epids) if epids is not None else zeros(self.nlc, 'int')
        self.ntc = unique(self.epids).size

        if include_orbit_variations:
            self.nor = self.ntc
        else:
            self.nor = 1

        if self.epids.size != self.nlc:
            raise ValueError(f"Epoch index array size ({self.epids.size}) should equal to the number of light curves ({self.nlc}).")

        if not (self.epids.max() == (self.ntc-1) and self.epids.min() == 0):
            raise ValueError(f"Epoch indices (`epids`) for {self.ntc} unique epochs should be given as integers between 0 and {self.ntc - 1}.")

        # Supersampling
        # -------------
        # A number of samples and the exposure time for each light curve.
        if nsamples is None:
            self.nsamples = ones(self.nlc, 'int')
        else:
            if isscalar(nsamples):
                self.nsamples = full(self.nlc, nsamples, 'int')
            else:
                self.nsamples = asarray(nsamples, 'int')
                if self.nsamples.min() < 1:
                    raise ValueError(f"The number of samples must be at least 1, but got {self.nsamples.min()}.")
                if self.nsamples.size != self.nlc:
                    raise ValueError(f"Number of samples array size ({self.nsamples.size}) should equal to the number of light curves ({self.nlc}).")

        if exptimes is None:
            self.exptimes = zeros(self.nlc, 'd')
        else:
            if isscalar(exptimes):
                self.exptimes = full(self.nlc, exptimes, 'd')
            else:
                self.exptimes = asarray(exptimes, 'd')
                if self.exptimes.min() < 0.0:
                    raise ValueError(f"The exposure time must be positive, but got {self.exptimes.min()}.")
                if self.exptimes.size != self.nlc:
                    raise ValueError(f"Exposure time array size ({self.exptimes.size}) should equal to the number of light curves ({self.nlc}).")

        if self.npb > 1 or self.ntc > 1:
            self.simple = False

        if self.backend == "jax":
            self.times = jax.device_put(self.times)
            self.lcids = jax.device_put(self.lcids)
            self.epids = jax.device_put(self.epids)
            self.nsamples = jax.device_put(self.nsamples)
            self.exptimes = jax.device_put(self.exptimes)

        self._err_broadcast = None
        self._on_data_set()

    def _on_data_set(self) -> None:
        """Hook called at the end of `set_data`, after all the shape attributes are set.

        Backends that keep a private copy of the data, such as OpenCL with its
        device buffers, override this to (re)upload it. The default does nothing.
        """
        pass

    def evaluate(self,
                 k: float | ndarray,
                 t0: float | ndarray,
                 p: float | ndarray,
                 a: float | ndarray,
                 i: float | ndarray,
                 e: float | ndarray = 0.0,
                 w: float | ndarray = 0.0,
                 ldp: ndarray | None = None) -> ndarray | tuple[ndarray, ndarray]:
        raise NotImplementedError

    def loglikelihood(self, obs, err, jitter, k, t0, p, a, i, e=0.0, w=0.0,
                       *, baseline: float = 1.0, **kwargs):
        """Log-likelihood and its gradient, reduced on the OpenCL device.

        Evaluates the model with ``copy=False`` and reduces the device-resident
        flux and Jacobian in place, so the Jacobian -- the largest array in a fit
        -- never crosses the bus. Only a handful of doubles are read back.

        The noise is independent Gaussian with a fixed per-point `err` and a
        sampled `jitter`, combined as ``e2 = err**2 + jitter**2``; a constant
        `err` gives the i.i.d. case. `evaluate` returns a flux deficit
        ``dF = F - 1``, not normalised flux, so the residual is
        ``(obs - baseline) - dF``: `baseline` shifts the deficit back onto the
        scale of normalised photometry.

        Parameters
        ----------
        obs : ndarray
            Observed flux, shape ``(npt,)``.
        err : ndarray
            Per-point uncertainties, shape ``(npt,)``. May be a scalar.
        jitter : float
            Additional white noise added in quadrature.
        k, t0, p, a, i, e, w : float or ndarray
            Model parameters, exactly as for `evaluate`.
        baseline : float, optional
            Flux baseline; default 1.0 for normalised photometry. A scalar
            shared across the whole light curve.
        **kwargs
            Passed to `evaluate`, so limb-darkening coefficients go here as
            ``ldc=`` (RoadRunner) or ``ldp=`` (uniform disk).

        Returns
        -------
        lnl : float or ndarray
            Log-likelihood; a plain float for a single parameter vector,
            otherwise shape ``(npv,)``.
        grad : ndarray
            Gradient with the same columns `evaluate` returns for the Jacobian,
            in the order ``[k, t0, p, a, i, e, w, ldc]``, followed by
            ``d lnL / d baseline`` and then a trailing ``d lnL / d jitter``.
        """
        if self.backend != 'opencl':
            raise NotImplementedError(
                f"The device-side log-likelihood reduction needs backend='opencl', not "
                f"'{self.backend}'. Compute it from evaluate()'s flux and Jacobian instead.")
        if not self.return_grad:
            raise ValueError("loglikelihood() needs a model built with return_grad=True.")
        if 'copy' in kwargs:
            raise TypeError("loglikelihood() does not take a 'copy' argument; it always "
                            "reduces on the device and returns the result.")

        obs = asarray(obs, float64)
        # Broadcast a scalar `err` once and reuse the array: minting a new one per
        # call would defeat the backend's identity-keyed upload cache, re-sending
        # the uncertainties on every likelihood evaluation.
        if isscalar(err) or ndim(err) == 0:
            value = float(err)
            if self._err_broadcast is None or self._err_broadcast[0] != value:
                self._err_broadcast = (value, full(self.npt, value, float64))
            err = self._err_broadcast[1]
        else:
            err = asarray(err, float64)

        npv = _npv_from_k(asarray(k), self.npb)
        slots, ncol = self._gradient_slots((k, t0, p, a, i, e, w), npv, **kwargs)

        self._model.set_observations(obs, err)
        self.evaluate(k, t0, p, a, i, e, w, **kwargs, copy=False)
        sums = self._model.reduce_loglikelihood(jitter, baseline, slots, ncol, npv)

        # Slot order matches reduction.cl: chi2, log_e2, chi2_e2, inv_e2, r_e2,
        # then one reduced column per gradient slot.
        chi2, log_e2, chi2_e2, inv_e2, r_e2 = (sums[:, j] for j in range(5))
        lnl = -0.5 * self.npt * log(2.0 * pi) - 0.5 * log_e2 - 0.5 * chi2
        djitter = jitter * (chi2_e2 - inv_e2)
        dbaseline = r_e2                      # d lnL/d baseline = sum(r/e2)
        # baseline goes before jitter so that grad[..., -1] is still d lnL/d jitter.
        grad = concatenate([sums[:, 5:], dbaseline[:, None], djitter[:, None]], axis=1)
        if lnl.size == 1:
            return float(lnl[0]), grad[0]
        return lnl, grad

    def _gradient_slots(self, orig_params, npv, **kwargs):
        """Describe the reduction's output slots, and the compact Jacobian width.

        Each slot names a compact Jacobian column and the data points it accepts,
        which makes the reduction land directly in `evaluate`'s expanded column
        layout: a parameter shared across passbands or epochs accepts every point
        and collapses to one column, while a per-passband or per-epoch parameter
        gets one slot per group. See `backends/opencl/reduction.cl`.
        """
        MATCH_ALL, MATCH_PASSBAND, MATCH_EPOCH = 0, 1, 2
        col, kind, grp = [], [], []
        # (compact column, number of groups, match rule, parameter as passed)
        specs = [(0, self.npb, MATCH_PASSBAND, orig_params[0]),
                 (1, self.ntc, MATCH_EPOCH, orig_params[1])]
        specs += [(2 + n, self.nor, MATCH_EPOCH, orig_params[2 + n]) for n in range(5)]

        for ccol, nd2, rule, original in specs:
            if _param_is_expanded(original, npv, nd2):
                col += nd2 * [ccol]
                kind += nd2 * [rule]
                grp += list(range(nd2))
            else:
                col.append(ccol)
                kind.append(MATCH_ALL)
                grp.append(0)

        nldc = self._ld_column_count(**kwargs)
        for j in range(self.npb * nldc):
            col.append(7 + j)
            kind.append(MATCH_ALL)      # the compact LD block is already per passband
            grp.append(0)
        return (array(col, 'int32'), array(kind, 'int32'), array(grp, 'int32')), 7 + self.npb * nldc

    def _ld_column_count(self, **kwargs) -> int:
        """Number of limb-darkening coefficients in the compact Jacobian."""
        return 0

    def get_callable(self) -> Callable:
        raise NotImplementedError

