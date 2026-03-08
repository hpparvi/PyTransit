import jax
from jax import numpy as jnp

from .loglikelihood import LogLikelihood

__all__ = ["WNLogLikelihood"]


class WNLogLikelihood(LogLikelihood):
    def __init__(self, tm, obs, err):
        super().__init__(tm, obs, err)

        if self.tm.backend == 'jax':
            model = self.tm.get_callable()
            def loglike(k, t0, p, a, i, e, w, jitter):
                fmod = model(k, t0, p, a, i, e, w)
                e2 = self.err**2 + jitter**2
                return - 0.5*self.obs.size*jnp.log(2*jnp.pi) - 0.5*jnp.sum(jnp.log(e2)) - 0.5*jnp.sum((self.obs - fmod)**2 / e2)

            if self.tm.return_grad:
                self._loglikelihood = jax.jit(jax.value_and_grad(loglike, argnums=tuple(range(8))))
            else:
                self._loglikelihood = jax.jit(loglike)

        else:
            pass