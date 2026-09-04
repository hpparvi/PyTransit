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
    `tm.evaluate` uses for its Jacobian -- then ``d lnL / d baseline``, then a
    trailing ``d lnL / d jitter``.

    `tm.evaluate` returns a flux deficit ``dF = F - 1``, not normalised flux, so
    the residual is ``(obs - baseline) - dF``; `baseline` (default 1.0) shifts
    the deficit back onto the scale of normalised photometry.

    The OpenCL backend reduces the log-likelihood and its gradient on the device,
    so the Jacobian never crosses the bus; see
    `pytransit.backends.opencl.reduction.cl`.
    """

    def __init__(self, tm, obs, err):
        super().__init__(tm, obs, err)

        if self.tm.backend == 'jax':
            model = self.tm.get_callable()
            def loglike(k, t0, p, a, i, e, w, baseline, jitter):
                fmod = model(k, t0, p, a, i, e, w)
                e2 = self.err**2 + jitter**2
                # (obs - baseline) before subtracting fmod: both operands are near
                # 1.0, so this subtraction is exact (Sterbenz), and everything
                # after stays on the small deficit scale. Do not reorder to
                # obs - (baseline + fmod); that discards the numerical benefit.
                r = (self.obs - baseline) - fmod
                return - 0.5*self.obs.size*jnp.log(2*jnp.pi) - 0.5*jnp.sum(jnp.log(e2)) - 0.5*jnp.sum(r**2 / e2)

            if self.tm.return_grad:
                self._loglikelihood = jax.jit(jax.value_and_grad(loglike, argnums=tuple(range(9))))
            else:
                self._loglikelihood = jax.jit(loglike)

        elif self.tm.backend == 'opencl':
            def loglike(k, t0, p, a, i, e=0.0, w=0.0, baseline=1.0, jitter=0.0, **kwargs):
                return self.tm.loglikelihood(self.obs, self.err, jitter,
                                             k, t0, p, a, i, e, w,
                                             baseline=baseline, **kwargs)
            self._loglikelihood = loglike

        else:
            raise NotImplementedError(
                f"WNLogLikelihood does not support the '{self.tm.backend}' backend yet. "
                f"Use backend='jax' or 'opencl', or compute it from evaluate()'s output.")
