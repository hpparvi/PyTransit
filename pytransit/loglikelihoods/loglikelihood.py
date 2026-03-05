import numpy as np
from jax import numpy as jnp
from numpy import ndarray

from ..models.transitmodel import TransitModel

__all__ = ["LogLikelihood"]


class LogLikelihood():
    def __init__(self, tm: TransitModel, obs: ndarray, err: ndarray, *nargs, **kwargs):
        self.tm: TransitModel = tm

        if self.tm.backend == 'jax':
            self.obs = jnp.array(obs)
            self.err = jnp.array(err)
        else:
            self.obs = np.array(obs)
            self.err = np.array(err)

    def __call__(self, *nargs):
        return self._loglikelihood(*nargs)
