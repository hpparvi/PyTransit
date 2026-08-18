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

from typing import Optional

from numpy import (asarray, atleast_2d, ones, zeros, hstack, unique, isfinite, nan, ndarray,
                   errstate, where)
from numpy.linalg import pinv, LinAlgError, cond


class LSTSQBaseline:
    """A parameterless multiplicative baseline fitted by linear least squares.

    ``LSTSQBaseline`` models the same linear-in-covariates baseline as
    :class:`~pytransit.lpf.baselines.linearbaseline.LinearModelBaseline`, but adds *no*
    free parameters to the LPF's parameter set. Instead, the baseline coefficients are
    solved for directly at every evaluation by least-squares fitting the relative
    residuals ``flux_obs / flux_model``.

    This works because the baseline is linear in its coefficients, so conditional on the
    transit model they have a closed-form maximum-likelihood solution. The coefficients
    are *profiled out* rather than sampled.

    .. note::
       Profiling is not marginalisation. The posterior conditions on the best-fit
       baseline rather than integrating over it, so parameter uncertainties are somewhat
       underestimated relative to a fully marginalised treatment.

    The design matrix is fixed (it depends only on the covariates, never on the parameter
    vector), so its pseudoinverse is computed once in :meth:`init_data`. Each call is then
    two small matrix products, fully vectorised over the parameter vector population.

    Usage
    -----
    ``LSTSQBaseline`` does not join the ``BaseLPF._baseline_models`` chain: those models
    are called as ``blm(pv, bl)`` and are evaluated *before* the transit model, an ordering
    a least-squares baseline cannot satisfy. An LPF opts in by overriding ``flux_model``::

        class MyLPF(BaseLPF):
            def _init_baseline(self):
                self.lstsq_baseline = LSTSQBaseline(self)

            def flux_model(self, pv):
                fmod = self.transit_model(pv)
                return fmod * self.lstsq_baseline(fmod) + self.trends(pv)

    Because the model is not registered with ``_add_baseline_model``, ``lpf.baseline(pv)``
    will *not* return this baseline. Plotting helpers that rely on ``lpf.baseline(pv)``
    should call ``lpf.lstsq_baseline(lpf.transit_model(pv))`` instead.

    Parameters
    ----------
    lpf
        The log posterior function the baseline belongs to. Its data must already be
        initialised, and it must have been given covariates.
    name
        Name of the baseline model.
    lcids
        Indices of the light curves the baseline is applied to. Defaults to all of them.
        Points belonging to light curves not covered keep a baseline of exactly 1.0.
    """

    def __init__(self, lpf, name: str = 'lstsq', lcids=None):
        self.name = name
        self.lpf = lpf

        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising LSTSQBaseline.')

        if lpf.covariates is None:
            raise ValueError('LSTSQBaseline requires the LPF to be initialised with covariates.')

        self._bl: Optional[ndarray] = None
        self.init_data(lcids)

    @staticmethod
    def normalize_covariates(covariates: ndarray) -> ndarray:
        """Normalise a light curve's covariate matrix before the intercept column is added.

        Each column is standardised to zero mean and unit standard deviation. Mixing a
        column on an arbitrary scale (raw BJD ~2.46e6, pixel positions ~1e3) with an
        intercept column of ones makes the design matrix badly conditioned, and ``pinv``
        returns a numerically poor solution rather than raising.

        ``BaseLPF._init_data`` already standardises the covariates it stores, so this is
        normally a no-op. It is applied anyway because standardisation is idempotent and
        this class may be handed covariates that did not come through that path.

        Constant columns are only centred: they carry no information, and scaling them
        would divide by zero. A design matrix left rank deficient by such a column is
        caught by :meth:`_check_conditioning`.

        Parameters
        ----------
        covariates
            Covariate matrix of shape ``(npt, ncov)``.

        Returns
        -------
        The standardised covariate matrix, same shape as the input.
        """
        cs = covariates.std(0)
        return (covariates - covariates.mean(0)) / where(cs > 0.0, cs, 1.0)

    def init_data(self, lcids=None):
        """Build the per-light-curve design matrices and their pseudoinverses."""
        self.lcids = lcids if lcids is not None else unique(self.lpf.lcids)
        self.nlc = self.lcids.size
        self.slices = [self.lpf.lcslices[lcid] for lcid in self.lcids]

        self.covs = []
        self.pinvs = []
        for i, lcid in enumerate(self.lcids):
            cv = self.normalize_covariates(asarray(self.lpf.covariates[lcid], 'd'))
            if cv.ndim == 1:
                cv = cv[:, None]
            x = hstack([ones((cv.shape[0], 1)), cv])
            self._check_conditioning(x, lcid)
            try:
                self.pinvs.append(pinv(x))
            except LinAlgError as e:
                raise LinAlgError(f'Failed to invert the {self.name} design matrix for light curve {lcid}: {e}')
            self.covs.append(x)

        self.ncoef = [x.shape[1] for x in self.covs]

        # Flat mask of the points this baseline covers. Points outside it stay at 1.0.
        self.mask = zeros(self.lpf.lcids.size, bool)
        for sl in self.slices:
            self.mask[sl] = 1

    @staticmethod
    def _check_conditioning(x: ndarray, lcid: int, limit: float = 1e10) -> None:
        with errstate(all='ignore'):
            c = cond(x)
        if not isfinite(c) or c > limit:
            raise ValueError(
                f'The design matrix for light curve {lcid} is singular or badly conditioned '
                f'(condition number {c:.3e}). The covariates are standardised automatically, so '
                f'this points to duplicate, constant, or linearly dependent covariate columns.')

    def _baseline_array(self, npv: int) -> ndarray:
        """Return the cached output buffer, reallocating it only when `npv` changes."""
        if self._bl is None or self._bl.shape[0] != npv:
            self._bl = ones((npv, self.lpf.timea.size))
        return self._bl

    def _observed_flux(self, oflux) -> ndarray:
        """Validate the observed flux and return it as a 2D array broadcastable over `npv`."""
        of = atleast_2d(self.lpf.ofluxa if oflux is None else asarray(oflux, 'd'))
        if of.shape[-1] != self.lpf.timea.size:
            raise ValueError(f'The observed flux has {of.shape[-1]} points, but the LPF data has '
                             f'{self.lpf.timea.size}.')
        return of

    def _fit(self, fmod: ndarray, oflux: ndarray, i: int) -> ndarray:
        """Least-squares coefficients for light curve `i`, for all parameter vectors."""
        sl = self.slices[i]
        with errstate(divide='ignore', invalid='ignore'):
            res = oflux[:, sl] / fmod[:, sl]            # (npv, npt_lc)
        return res @ self.pinvs[i].T                    # (npv, ncoef)

    def coefficients(self, fmod, oflux=None):
        """Fit the baseline and return its coefficients rather than its evaluation.

        Parameters
        ----------
        fmod
            Model fluxes, either a 1D ``(npt,)`` array or a 2D ``(npv, npt)`` array.
        oflux
            Observed fluxes to fit against, either a 1D ``(npt,)`` array or a 2D
            ``(npv, npt)`` array. Defaults to the LPF's own ``ofluxa``.

        Returns
        -------
        A list with one entry per covered light curve. Each entry has shape
        ``(npv, ncoef)`` for a 2D `fmod` and ``(ncoef,)`` for a 1D one, where the first
        coefficient is the intercept. The coefficient count may differ between light
        curves, which is why this is a list rather than an array.
        """
        fmod = asarray(fmod, 'd')
        fm = atleast_2d(fmod)
        of = self._observed_flux(oflux)
        cs = [self._fit(fm, of, i) for i in range(self.nlc)]
        return cs if fmod.ndim > 1 else [c[0] for c in cs]

    def __call__(self, fmod, oflux=None):
        """Fit and evaluate the baseline for a set of model fluxes.

        Parameters
        ----------
        fmod
            Model fluxes, either a 1D ``(npt,)`` array or a 2D ``(npv, npt)`` array,
            where `npt` is the total number of datapoints in the LPF.
        oflux
            Observed fluxes to fit against, either a 1D ``(npt,)`` array or a 2D
            ``(npv, npt)`` array. Defaults to the LPF's own ``ofluxa``, which is what a
            likelihood evaluation wants; pass it explicitly to fit the baseline against
            simulated or otherwise modified data.

        Returns
        -------
        The multiplicative baseline, shaped like `fmod`. A parameter vector whose fit
        fails, or whose residuals are not finite, gets an all-NaN baseline, which
        propagates to a non-finite log likelihood and is rejected by the sampler.

        .. warning::
           The returned array is a reused internal buffer that the next call overwrites.
           Copy it if you need to keep it.
        """
        fmod = asarray(fmod, 'd')
        fm = atleast_2d(fmod)
        of = self._observed_flux(oflux)
        bl = self._baseline_array(fm.shape[0])

        for i, sl in enumerate(self.slices):
            b = self._fit(fm, of, i) @ self.covs[i].T   # (npv, npt_lc)
            bad = ~isfinite(b).all(axis=1)
            if bad.any():
                b[bad] = nan
            bl[:, sl] = b

        return bl if fmod.ndim > 1 else bl[0]
