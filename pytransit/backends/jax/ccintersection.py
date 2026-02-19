"""Circle-circle intersection area for JAX.

JAX-compatible (jit + grad) versions of the circle-circle intersection
functions from pytransit.backends.numba.ccintersection. All control flow
uses jnp.where so that every code path is traced and the functions remain
differentiable.

Gradient safety: jnp.where(cond, a, b) evaluates gradients of BOTH a and b,
so even "dead" branches must have finite gradients. We use a small epsilon
guard on the sqrt argument to avoid inf/NaN gradients at the boundary
(sqrt(0) -> inf grad).
"""

import jax.numpy as jnp

_EPS_SQRT = 1e-12   # Guard for sqrt(x): must be > 0


def _safe_sqrt(x):
    """sqrt that returns finite gradients even at x=0."""
    return jnp.sqrt(jnp.maximum(x, _EPS_SQRT))


def ccia(r1, r2, b):
    """Circle-circle intersection area and kite angle (Agol et al. 2020).

    Branchless JAX version. Safe for jax.jit and jax.grad.
    Returns (area, k0).
    """
    r1sq = r1 * r1
    r2sq = r2 * r2
    bsq = b * b

    # The kite area argument is permutation-symmetric in (r1, r2, b),
    # so no sorting is needed.
    kite_arg = (r1 + r2 + b) * (-r1 + r2 + b) * (r1 - r2 + b) * (r1 + r2 - b)
    a_kite = 0.5 * _safe_sqrt(kite_arg)

    k0 = jnp.arctan2(2.0 * a_kite, r2sq - r1sq + bsq)
    k1 = jnp.arctan2(2.0 * a_kite, r1sq - r2sq + bsq)
    a_lens = r1sq * k1 + r2sq * k0 - a_kite

    no_overlap = r1 + r2 <= b
    r2_inside_r1 = b <= r1 - r2
    r1_inside_r2 = b <= r2 - r1

    area = jnp.where(no_overlap, 0.0,
           jnp.where(r2_inside_r1, jnp.pi * r2sq,
           jnp.where(r1_inside_r2, jnp.pi * r1sq,
           a_lens)))

    kite_k0 = jnp.where(no_overlap, 0.0,
              jnp.where(r2_inside_r1, jnp.pi,
              jnp.where(r1_inside_r2, 0.0,
              k0)))

    return area
