import jax
import jax.numpy as jnp
import numba

from numpy import array, ndarray, squeeze, asarray
from numpy.typing import ArrayLike

from ..backends.numba.udmodel import udmodel as nbmodel
from ..backends.numba.udmodel_grad import udmodel_grad as nbmodel_grad
from ..backends.jax.udmodel import udmodel as jaxmodel, udmodel_grad as jaxmodel_grad
from .transitmodel import TransitModel

try:
    from ..backends.opencl.udmodel import UDModelCL
except ImportError as error:          # pragma: no cover - depends on the local install
    # Only a genuinely missing dependency is tolerated here. The backend touches no
    # device at import time (the context is created in `UDModelCL.__init__`), so a
    # machine without an OpenCL driver still imports fine and fails later, loudly.
    # Catching more than ImportError would turn a typo in the backend into a
    # misleading "no OpenCL driver" message.
    UDModelCL = None
    _CL_IMPORT_ERROR = error
from ._utils import _normalize_parameter_shapes, _npv_from_k, _expand_gradient, PType

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
        elif self.backend == 'opencl':
            if UDModelCL is None:
                raise RuntimeError("The OpenCL backend is unavailable because pyopencl could not "
                                   "be imported.") from _CL_IMPORT_ERROR
            unknown = set(self._kwargs) - {'precision', 'cl_ctx', 'cl_queue'}
            if unknown:
                raise ValueError(f"Unknown OpenCL backend options: {sorted(unknown)}. "
                                 f"Expected 'precision', 'cl_ctx', or 'cl_queue'.")
            self._model = UDModelCL(return_grad=self.return_grad,
                                    precision=self._kwargs.get('precision', 'double'),
                                    cl_ctx=self._kwargs.get('cl_ctx'),
                                    cl_queue=self._kwargs.get('cl_queue'))
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _on_data_set(self):
        if self.backend == 'opencl':
            self._model.set_data(self.times, self.lcids, self.pbids, self.epids,
                                 self.nsamples, self.exptimes)

    def evaluate(self,
                 k: PType, t0: PType, p: PType, a: PType, i: PType, e: PType = 0.0, w: PType = 0.0,
                 ldp: ArrayLike | None = None, copy: bool = True
                 ) -> ndarray | tuple[ndarray, ndarray] | jax.Array | tuple[jax.Array, jax.Array] | None:
        """Evaluate the model.

        Parameters
        ----------
        copy
            OpenCL backend only. If `False`, the results are left in device memory
            and `None` is returned, skipping a readback that dominates the runtime
            for a large parameter-vector population. The flux and the *compact*
            gradient are then available as `model._model._b_flux` and, when
            `return_grad=True`, `_b_dflux`. Two caveats: the gradient is the
            *compact* `(npv, npt, 7)` array, not expanded into per-passband and
            per-epoch columns, because that expansion happens on the host; and a
            later read of those buffers must go through the model's own queue
            (`model._model.queue`) or follow a `queue.finish()`, since the kernel
            is still in flight when this returns.
        """
        if not copy and self.backend != 'opencl':
            raise ValueError(f"The 'copy' argument is supported only by the OpenCL backend, "
                             f"not by '{self.backend}'.")

        orig_params = (k, t0, p, a, i, e, w)
        npv = _npv_from_k(asarray(k), self.npb)
        k, t0, p, a, i, e, w = _normalize_parameter_shapes(k, t0, p, a, i, e, w, self.npb, self.ntc, self.nor)
        kwargs = {} if self.backend != 'opencl' else {'copy': copy}
        result = self._model(self.times, k, t0, p, a, i, e, w,
                             self.lcids, self.pbids, self.epids, self.nsamples, self.exptimes,
                             self.npb, self.nor, **kwargs)
        if result is None:
            return None
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
