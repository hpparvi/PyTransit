"""Example: JAX-jittable chi2 with analytic gradients via the uniform disk model.

Uses jax.value_and_grad through udmodel's @custom_jvp to get both chi2 and
dchi2/dk in a single JIT-compiled call. The key pattern: call jax_udmodel
directly (no inner jax.jit) inside the chi2 closure, then wrap everything
with jax.jit(jax.value_and_grad(chi2)).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy import pi

from pytransit.models.udmodel import UniformDiskModel
from pytransit.backends.jax.udmodel import udmodel as jax_udmodel

# --- Setup: observation times for 3 light curves across 2 epochs ---
times = jnp.linspace(-0.5, 0.5, 1000)
lcids = jnp.repeat(jnp.array([0, 1, 2]), times.size)
times = jnp.concatenate([times, times + 3.75, times + 2 * 3.75])

k, t0, p, a, i, e, w = 0.2, [-0.3, 0.1], 3.75, 6.2, 0.49 * pi, 0.12, 1.54

# Use the model wrapper to set up observation metadata (lcids, pbids, etc.)
tm = UniformDiskModel('jax')
tm.set_data(times, lcids=lcids, epids=[0, 0, 1],
            nsamples=[1, 10, 1], exptimes=[0.05, 0.05, 0.0],
            include_orbit_variations=True)

# Generate synthetic observations with noise
tm_g = UniformDiskModel('jax', return_grad=True)
tm_g.set_data(times, lcids=lcids, epids=[0, 0, 1],
              nsamples=[1, 10, 1], exptimes=[0.05, 0.05, 0.0],
              include_orbit_variations=True)
flux_true, _ = tm_g.evaluate(k, t0, p, a, [0.49 * pi, 0.455 * pi], e, w)
np.random.seed(42)
fobs = jnp.array(flux_true + np.random.normal(0.0, 0.001, size=flux_true.size))

# --- Build the chi2 function ---
PARAM_NAMES = ['k', 't0', 'p', 'a', 'i', 'e', 'w']


def chi2(k, t0, p, a, i, e, w):
    """Chi-squared as a function of all transit parameters."""
    flux = jax_udmodel(tm.times, k, t0, p, a, i, e, w,
                       tm.lcids, tm.pbids, tm.epids,
                       tm.nsamples, tm.exptimes,
                       tm.npb, tm.nor)
    return jnp.sum((fobs - flux) ** 2)


# Differentiate w.r.t. all 7 parameters, then JIT the whole thing
jchi2 = jax.jit(jax.value_and_grad(chi2, argnums=tuple(range(7))))

# --- Test parameters ---
k_test = jnp.array([0.1])
t0_test = jnp.array(t0)
p_test = jnp.array([p, p])
a_test = jnp.array([a, a])
i_test = jnp.array([i, i])
e_test = jnp.array([e, e])
w_test = jnp.array([w, w])
params = (k_test, t0_test, p_test, a_test, i_test, e_test, w_test)

# --- Run it ---
print("First call (includes JIT compilation)...")
value, grads = jchi2(*params)
print(f"  chi2 = {value:.6f}")
for name, g in zip(PARAM_NAMES, grads):
    print(f"  dchi2/d{name:2s} = {g}")

print("\nSecond call (JIT-cached, should be fast)...")
params2 = (jnp.array([0.2]),) + params[1:]
value2, grads2 = jchi2(*params2)
print(f"  chi2 = {value2:.6f}")

# Verify each gradient element against finite differences
print("\nFinite-difference check (element-wise)...")
eps = 1e-7
for idx, name in enumerate(PARAM_NAMES):
    analytic = grads[idx]
    for j in range(analytic.size):
        delta = jnp.zeros_like(params[idx]).at[j].set(eps)
        params_plus = list(params)
        params_minus = list(params)
        params_plus[idx] = params[idx] + delta
        params_minus[idx] = params[idx] - delta
        fd = float((chi2(*params_plus) - chi2(*params_minus)) / (2 * eps))
        ana = float(analytic[j])
        denom = max(abs(fd), 1e-30)
        err = abs(ana - fd) / denom
        print(f"  d/d{name}[{j}]: analytic={ana: .8e}  fd={fd: .8e}  rel_err={err:.2e}")
