import numpy as np
from numba import types
from numba.core.extending import overload
from numpy import asarray, atleast_2d, full, floating
from numpy.typing import ArrayLike, NDArray


PType = float | floating | NDArray[floating]


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


def _param_is_expanded(orig, npv, nd2):
    """Whether a parameter was supplied with one value per expansion-axis element.

    Mirrors `_normalize_parameter_shape`: a scalar, or a 1D per-parameter-vector
    array of length ``npv``, is shared along the expansion axis (and collapses to
    a single gradient column), whereas a length-``nd2`` 1D array (with ``npv == 1``)
    or an ``(npv, nd2)`` 2D array provides a distinct value per axis element (and
    expands to ``nd2`` gradient columns).
    """
    if nd2 <= 1:
        return False
    o = asarray(orig)
    if o.ndim == 0:
        return False
    if o.ndim == 1:
        return o.size == nd2 and npv == 1
    return o.shape[1] == nd2


def _expand_gradient(dflux_compact, flux, npv, orig_params, lcids, pbids, epids,
                     npb, ntc, nor, ld_block=None, use_jax=False):
    """Scatter a compact per-point gradient into a per-input-parameter Jacobian.

    The Numba/JAX kernels return a compact gradient whose first seven columns
    hold, for each data point, the derivative w.r.t. its own passband's ``k`` and
    its own epoch's ``t0, p, a, i, e, w``. This expands those columns to match how
    the parameters were passed to ``evaluate``: a parameter shared across
    passbands/epochs keeps a single column, while a passband-dependent radius ratio
    or an epoch-dependent transit centre / orbital parameter is scattered into one
    column per passband (``k``) or epoch (``t0`` and the orbital parameters), with
    each point's derivative landing only in its own column. An optional ``ld_block``
    (already per-passband) is appended unchanged. The output column order is
    ``[k, t0, p, a, i, e, w, ldc]``.

    Parameters
    ----------
    dflux_compact : ndarray
        Compact gradient, shape ``(npv, npt, 7 [+ ld columns])``.
    flux : ndarray
        Model flux, shape ``(npv, npt)``; used to propagate invalid-PV NaNs.
    npv : int
        Number of parameter vectors.
    orig_params : tuple
        The ``(k, t0, p, a, i, e, w)`` arguments *as passed* to ``evaluate``
        (before shape normalisation), used to decide per-parameter expansion.
    lcids, pbids, epids : ndarray
        Light-curve, passband, and epoch index arrays.
    npb, ntc, nor : int
        Number of passbands, transit centres, and orbit variations.
    ld_block : ndarray, optional
        Per-passband limb-darkening derivative columns to append unchanged.
    use_jax : bool, optional
        Use ``jax.numpy`` (functional scatter) instead of NumPy.
    """
    xp = __import__('jax.numpy', fromlist=['']) if use_jax else np

    pb_pt = np.asarray(pbids)[np.asarray(lcids)]   # passband index per data point
    ep_pt = np.asarray(epids)[np.asarray(lcids)]   # epoch index per data point
    npt = pb_pt.size

    def block(ccol, nd2, idx, expanded):
        col = dflux_compact[:, :, ccol]                  # (npv, npt)
        if not expanded:
            return col[:, :, None]                       # shared -> single column
        out = xp.zeros((npv, npt, nd2), dtype=dflux_compact.dtype)
        if use_jax:
            return out.at[:, xp.arange(npt), idx].set(col)
        out[:, np.arange(npt), idx] = col                # scatter to own passband/epoch
        return out

    k, t0, p, a, i, e, w = orig_params
    specs = ((0, npb, pb_pt, k), (1, ntc, ep_pt, t0), (2, nor, ep_pt, p),
             (3, nor, ep_pt, a), (4, nor, ep_pt, i), (5, nor, ep_pt, e), (6, nor, ep_pt, w))
    blocks = [block(c, nd2, idx, _param_is_expanded(o, npv, nd2)) for c, nd2, idx, o in specs]
    if ld_block is not None:
        blocks.append(ld_block)
    dflux = xp.concatenate(blocks, axis=2)

    mask = xp.isnan(flux)                                # invalid-PV rows are all NaN
    if use_jax:
        dflux = xp.where(mask[:, :, None], xp.nan, dflux)
    else:
        dflux[mask] = np.nan
    return dflux


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
