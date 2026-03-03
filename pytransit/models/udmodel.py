import jax
import numba

from numpy import array, ndarray

from pytransit.backends.numba.udmodel import udmodel as nbmodel
from pytransit.backends.numba.udmodel_grad import udmodel_grad as nbmodel_grad
from pytransit.backends.jax.udmodel import uniform_model as jaxmodel, _uniform_model_fwd as jaxmodel_grad
from pytransit.models.transitmodel import TransitModel

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
                self._model = jax.jit(jaxmodel_grad)
            else:
                self._model = jax.jit(jaxmodel)
    def evaluate(self,
                 k: float | ndarray,
                 t0: float | ndarray,
                 p: float | ndarray,
                 a: float | ndarray,
                 i: float | ndarray,
                 e: float | ndarray = 0.0,
                 w: float | ndarray = 0.0,
                 ldp: ndarray | None = None) -> ndarray | tuple[ndarray, ndarray]:
        k, t0, p, a, i, e, w = self._normalize_parameter_shapes(k, t0, p, a, i, e, w)
        return self._model(self.times, k, t0, p, a, i, e, w)

