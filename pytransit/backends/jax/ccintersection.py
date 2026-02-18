"""Circle-circle intersection area for JAX.

JAX-compatible (jit + grad) versions of the circle-circle intersection
functions from pytransit.backends.numba.ccintersection. All control flow
uses jnp.where so that every code path is traced and the functions remain
differentiable.

Gradient safety: jnp.where(cond, a, b) evaluates gradients of BOTH a and b,
so even "dead" branches must have finite gradients. We use small epsilon
guards on sqrt and arccos arguments to avoid inf/NaN gradients at boundaries
(sqrt(0) -> inf grad, arccos(±1) -> inf grad).
"""

import jax.numpy as jnp

_EPS_SQRT = 1e-12   # Guard for sqrt(x): must be > 0


def _safe_sqrt(x):
    """sqrt that returns finite gradients even at x=0."""
    return jnp.sqrt(jnp.maximum(x, _EPS_SQRT))


def tsort(r1, r2, b):
    """Sort three values into descending order (x >= y >= z) without branches."""
    x = jnp.maximum(r1, jnp.maximum(r2, b))
    z = jnp.minimum(r1, jnp.minimum(r2, b))
    y = (r1 + r2 + b) - x - z
    return x, y, z


def circle_circle_intersection_area(r1, r2, b):
    """Circle-circle intersection area and kite angle (Agol et al. 2020).

    Branchless JAX version. Safe for jax.jit and jax.grad.
    Returns (area, k0).
    """
    x, y, z = tsort(r1, r2, b)

    kite_arg = (x + (y + z)) * (z - (x - y)) * (z + (x - y)) * (x + (y - z))
    a_kite = 0.5 * _safe_sqrt(kite_arg)

    k0 = jnp.arctan2(2.0 * a_kite, (r2 - r1) * (r2 + r1) + b * b)
    k1 = jnp.arctan2(2.0 * a_kite, (r1 - r2) * (r1 + r2) + b * b)
    a_lens = r1 * r1 * k1 + r2 * r2 * k0 - a_kite

    no_overlap = r1 + r2 <= b
    r2_inside_r1 = b <= r1 - r2
    r1_inside_r2 = b <= r2 - r1

    area = jnp.where(no_overlap, 0.0,
           jnp.where(r2_inside_r1, jnp.pi * r2**2,
           jnp.where(r1_inside_r2, jnp.pi * r1**2,
           a_lens)))

    kite_k0 = jnp.where(no_overlap, 0.0,
              jnp.where(r2_inside_r1, jnp.pi,
              jnp.where(r1_inside_r2, 0.0,
              k0)))

    return area, kite_k0
