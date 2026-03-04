import jax
import numba

from numpy import array, ndarray, squeeze

from ..backends.numba.udmodel import udmodel as nbmodel
from ..backends.numba.udmodel_grad import udmodel_grad as nbmodel_grad
from ..backends.jax.udmodel import udmodel as jaxmodel, udmodel_grad as jaxmodel_grad
from .transitmodel import TransitModel
from ._utils import _normalize_parameter_shapes, _npv_from_k

__all__ = ['UniformDiskModel']

class UniformDiskModel(TransitModel):
    def _init_model(self):
        if self.backend == 'numba':
            if self.return_grad:
                self._model = numba.njit(nbmodel_grad, parallel=self.parallel)
            else:
                self._model = numba.njit(nbmodel, parallel=self.parallel)
        elif self.backend == 'jax':
            if self.return_grad:
                self._model = jax.jit(jaxmodel_grad, static_argnums=(13, 15))
            else:
                self._model = jax.jit(jaxmodel, static_argnums=(13, 15))

    def evaluate(self,
                 k: float | ndarray,
                 t0: float | ndarray,
                 p: float | ndarray,
                 a: float | ndarray,
                 i: float | ndarray,
                 e: float | ndarray = 0.0,
                 w: float | ndarray = 0.0,
                 ldp: ndarray | None = None) -> ndarray | tuple[ndarray, ndarray]:
        npv = _npv_from_k(k, self.npb)
        k, t0, p, a, i, e, w = _normalize_parameter_shapes(k, t0, p, a, i, e, w, self.npb, self.ntc, self.nor)
        flux = self._model(self.times,
                           k, t0, p, a, i, e, w,
                           self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes,
                           npv, self.npb, self.nor)

        if self.return_grad:
            return squeeze(flux[0]), squeeze(flux[1])
        else:
            return squeeze(flux)
