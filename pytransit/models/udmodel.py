import jax
import jax.numpy as jnp
import numba

from numpy import array, ndarray, squeeze, asarray

from ..backends.numba.udmodel import udmodel as nbmodel
from ..backends.numba.udmodel_grad import udmodel_grad as nbmodel_grad
from ..backends.jax.udmodel import udmodel as jaxmodel, udmodel_grad as jaxmodel_grad
from .transitmodel import TransitModel
from ._utils import _normalize_parameter_shapes, _npv_from_k, _expand_gradient

__all__ = ['UniformDiskModel']

class UniformDiskModel(TransitModel):
    def _init_model(self):
        if self.backend == 'numba':
            if self.return_grad:
                self._model = numba.njit(nbmodel_grad, parallel=self.parallel)
            else:
                self._model = numba.njit(nbmodel, parallel=self.parallel)
        elif self.backend == 'jax':
            vmap_axes = (None, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None)
            if self.return_grad:
                self._model = jax.jit(jax.vmap(jaxmodel_grad, in_axes=vmap_axes), static_argnums=(13, 14))
            else:
                self._model = jax.jit(jax.vmap(jaxmodel, in_axes=vmap_axes), static_argnums=(13, 14))

    def evaluate(self,
                 k: PType, t0: PType, p: PType, a: PType, i: PType, e: PType = 0.0, w: PType = 0.0,
                 ldp: ArrayLike | None = None) -> ndarray | tuple[ndarray, ndarray] | jax.Array | tuple[jax.Array, jax.Array]:

        orig_params = (k, t0, p, a, i, e, w)
        npv = _npv_from_k(asarray(k), self.npb)
        k, t0, p, a, i, e, w = _normalize_parameter_shapes(k, t0, p, a, i, e, w, self.npb, self.ntc, self.nor)
        result = self._model(self.times, k, t0, p, a, i, e, w,
                             self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes, self.npb, self.nor)
        sq = jnp.squeeze if self.backend == 'jax' else squeeze
        if not self.return_grad:
            return sq(result)

        flux, dflux_compact = result[0], result[1]
        dflux = _expand_gradient(dflux_compact, flux, npv, orig_params,
                                 self.lcids, self.pbids, self.epids,
                                 self.npb, self.ntc, self.nor, use_jax=(self.backend == 'jax'))
        return sq(flux), sq(dflux)

    def get_callable(self):
        if self.backend == 'jax':
            def model(k, t0, p, a, i, e, w):
                return jaxmodel(self.times, k, t0, p, a, i, e, w,
                          self.lcids, self.pbids, self.epids,
                          self.nsamples, self.exptimes,
                          self.npb, self.nor)
            return model
        else:
            return self._model
