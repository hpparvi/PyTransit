import numpy as np
from jax import numpy as jnp
from numpy import ndarray

from ..models.transitmodel import TransitModel

__all__ = ["LogLikelihood"]


class LogLikelihood():
    def __init__(self, tm: TransitModel, obs: ndarray, err: ndarray, *nargs, **kwargs):
        self.tm: TransitModel = tm

        obs = np.asarray(obs)
        # A scalar `err` is the i.i.d. case; broadcast it once so every backend
        # sees one uncertainty per point.
        err = np.broadcast_to(np.asarray(err, float), obs.shape).copy()

        if self.tm.backend == 'jax':
            self.obs = jnp.array(obs)
            self.err = jnp.array(err)
        else:
            self.obs = np.array(obs)
            self.err = err

    def __call__(self, *nargs, **kwargs):
        return self._loglikelihood(*nargs, **kwargs)
