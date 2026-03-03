import numpy as np
from numba import types
from numba.core.extending import overload
from numpy import asarray, atleast_2d, full


def _normalize_parameter_shape(p, npv, nd2):
    """Reshape a scalar, 1D, or 2D parameter array into a 2D (npv, nd2) shape."""
    p = asarray(p)
    if p.ndim == 0:
        if npv == 1 and nd2 == 1:
            return atleast_2d(p)
        else:
            return full((npv, nd2), p)
    elif p.ndim == 1:
        if p.size == nd2:
            if npv > 1:
                raise ValueError("Cannot cast 1D parameter array to the required 2D shape.")
            return atleast_2d(p)
        else:
            if p.size != npv:
                raise ValueError("Cannot cast 1D parameter array to the required 2D shape.")
            return atleast_2d(p).T
    elif p.ndim == 2:
        if p.shape[1] != nd2:
            raise ValueError("The 2D parameter array has an incompatible shape.")
        return p
    else:
        raise ValueError("The parameter array has too many dimensions.")


@overload(_normalize_parameter_shape)
def _normalize_parameter_shape_ovld(p, npv, nd2):
    if isinstance(p, types.Float):
        def impl(p, npv, nd2):
            return np.full((npv, nd2), p)
        return impl
    elif isinstance(p, types.Array) and p.ndim == 1:
        def impl(p, npv, nd2):
            if p.size == nd2:
                if npv > 1:
                    raise ValueError("Cannot cast 1D parameter array to the required 2D shape.")
                return p.reshape(1, nd2)
            else:
                if p.size != npv:
                    raise ValueError("Cannot cast 1D parameter array to the required 2D shape.")
                return p.reshape(npv, 1)
        return impl
    elif isinstance(p, types.Array) and p.ndim == 2:
        def impl(p, npv, nd2):
            if p.shape[1] != nd2:
                raise ValueError("The 2D parameter array has an incompatible shape.")
            return p
        return impl


def _npv_from_k(k, npb):
    """Determine the number of parameter vectors from the radius ratio array."""
    k = asarray(k)
    if k.ndim == 0:
        return 1
    elif k.ndim == 1:
        if k.size == npb:
            return 1
        else:
            return k.size
    elif k.ndim == 2:
        if k.shape[1] != npb:
            raise ValueError("The radius ratio array should have a shape (npv, npb).")
        return k.shape[0]
    else:
        raise ValueError("The radius ratio array should have a shape (npv, npb).")


@overload(_npv_from_k)
def _npv_from_k_ovld(k, npb):
    if isinstance(k, types.Float):
        def impl(k, npb):
            return 1
        return impl
    elif isinstance(k, types.Array) and k.ndim == 1:
        def impl(k, npb):
            if k.size == npb:
                return 1
            else:
                return k.size
        return impl
    elif isinstance(k, types.Array) and k.ndim == 2:
        def impl(k, npb):
            if k.shape[1] != npb:
                raise ValueError("The radius ratio array should have a shape (npv, npb).")
            return k.shape[0]
        return impl


def _normalize_parameter_shapes(k, t0, p, a, i, e, w, npb, ntc, nor):
    k = asarray(k)
    npv = _npv_from_k(k, npb)

    ks = _normalize_parameter_shape(k, npv, npb)
    t0s = _normalize_parameter_shape(t0, npv, ntc)
    ps = _normalize_parameter_shape(p, npv, nor)
    smas = _normalize_parameter_shape(a, npv, nor)
    incs = _normalize_parameter_shape(i, npv, nor)
    eccs = _normalize_parameter_shape(e, npv, nor)
    ws = _normalize_parameter_shape(w, npv, nor)
    return ks, t0s, ps, smas, incs, eccs, ws
