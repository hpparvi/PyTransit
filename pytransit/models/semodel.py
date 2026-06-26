import numba

from numpy import ndarray, squeeze, asarray

from ..backends.numba.semodel import semodel as nbmodel
from ..backends.numba.semodel_grad import semodel_grad as nbmodel_grad
from .transitmodel import TransitModel
from ._utils import _normalize_parameter_shapes, _npv_from_k, _expand_gradient, PType

__all__ = ['SecondaryEclipseModel']


class SecondaryEclipseModel(TransitModel):
    """Secondary-eclipse (occultation) model.

    Models the occultation light curve produced when the planet passes behind
    the star. The planet is treated as a uniformly bright disk, so the
    out-of-eclipse flux equals ``pi * k**2`` (the planet's projected area in
    units of the stellar disk area) and the eclipse removes the fraction of the
    planet's disk hidden by the star. Host-star limb darkening is not modelled
    because only the geometric overlap matters.

    The eclipse centre is offset from the mid-transit time ``t0`` by the
    eccentricity-dependent eclipse-time offset plus the transit-to-eclipse
    light-travel delay, which is why a stellar radius ``rstar`` is required.

    Notes
    -----
    The returned flux is on the planet's *absolute* surface-brightness scale
    (out-of-eclipse baseline ``pi * k**2``, dropping to zero at mid-eclipse),
    not the normalised-deficit scale used by the transit models. Multiply by
    the planet-to-star surface-brightness (or flux) ratio to place it on the
    same normalised scale as a transit light curve.
    """

    def _init_model(self):
        if self.backend == 'numba':
            if self.return_grad:
                self._model = numba.njit(nbmodel_grad, parallel=self.parallel)
            else:
                self._model = numba.njit(nbmodel, parallel=self.parallel)
        else:
            raise ValueError(f"The SecondaryEclipseModel supports only the 'numba' backend, got '{self.backend}'.")

    def evaluate(self,
                 k: PType, t0: PType, p: PType, a: PType, i: PType, e: PType = 0.0, w: PType = 0.0,
                 rstar: float = 1.0) -> ndarray | tuple[ndarray, ndarray]:
        """Evaluate the secondary-eclipse model.

        Parameters
        ----------
        k
            Planet-to-star radius ratio as a float, 1D, or 2D array (npv, npb).
        t0
            Mid-transit time(s) (the eclipse is located relative to this).
        p
            Orbital period(s).
        a
            Scaled semi-major axis (a/R_star).
        i
            Orbital inclination(s) [rad].
        e : optional
            Orbital eccentricity.
        w : optional
            Argument of periastron [rad].
        rstar : optional
            Stellar radius [R_sun], used for the light-travel-time correction.

        Returns
        -------
        ndarray or tuple
            Eclipse flux (npv, npt), or (flux, dflux) if ``return_grad=True``.
            The gradient columns match the parameters as passed, in the order
            ``[k, t0, p, a, i, e, w]``: a scalar parameter contributes a single
            shared column, while a passband-dependent radius ratio or an
            epoch-dependent transit centre / orbital parameter is expanded into
            one column per passband or epoch.
        """
        orig_params = (k, t0, p, a, i, e, w)
        npv = _npv_from_k(asarray(k), self.npb)
        k, t0, p, a, i, e, w = _normalize_parameter_shapes(k, t0, p, a, i, e, w, self.npb, self.ntc, self.nor)
        result = self._model(self.times, k, t0, p, a, i, e, w, float(rstar),
                             self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes, self.npb, self.nor)
        if not self.return_grad:
            return squeeze(result)

        flux, dflux_compact = result[0], result[1]
        dflux = _expand_gradient(dflux_compact, flux, npv, orig_params,
                                 self.lcids, self.pbids, self.epids,
                                 self.npb, self.ntc, self.nor)
        return squeeze(flux), squeeze(dflux)

    def get_callable(self):
        return self._model
