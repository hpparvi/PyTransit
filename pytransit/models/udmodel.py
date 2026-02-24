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

    def set_data(self, times) -> None:
        if self.backend == 'jax':
            self.times = jax.device_put(times)
        else:
            self.times = array(times)

    def evaluate(self, k, t0, p, a, i, e, w) -> ndarray | tuple[ndarray, ndarray]:
        return self._model(self.times, k, t0, p, a, i, e, w)

