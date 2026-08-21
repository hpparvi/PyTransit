import jax
from jax import numpy as jnp

from .loglikelihood import LogLikelihood

__all__ = ["WNLogLikelihood"]


class WNLogLikelihood(LogLikelihood):
    """White-noise log-likelihood for a transit model.

    The noise is independent Gaussian with the fixed per-point uncertainties
    `err` and a sampled `jitter`, combined as ``e2 = err**2 + jitter**2``. Pass a
    constant `err` for the i.i.d. case.

    With ``tm.return_grad``, calling the object returns ``(lnl, grad)``, where
    `grad` holds one column per model parameter as passed -- the same layout
    `tm.evaluate` uses for its Jacobian -- plus a trailing ``d lnL / d jitter``.

    The OpenCL backend reduces the log-likelihood and its gradient on the device,
    so the Jacobian never crosses the bus; see
    `pytransit.backends.opencl.reduction.cl`.
    """

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

        elif self.tm.backend == 'opencl':
            def loglike(k, t0, p, a, i, e=0.0, w=0.0, jitter=0.0, **kwargs):
                return self.tm.loglikelihood(self.obs, self.err, jitter,
                                             k, t0, p, a, i, e, w, **kwargs)
            self._loglikelihood = loglike

        else:
            raise NotImplementedError(
                f"WNLogLikelihood does not support the '{self.tm.backend}' backend yet. "
                f"Use backend='jax' or 'opencl', or compute it from evaluate()'s output.")
