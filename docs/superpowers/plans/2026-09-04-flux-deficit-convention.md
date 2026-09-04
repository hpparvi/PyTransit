# Flux Deficit Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PyTransit's transit models return the flux deficit `dF = F - 1` (out-of-transit exactly `0.0`) instead of normalised absolute flux, and give the white-noise log-likelihood a fitted `baseline` parameter.

**Architecture:** Every forward kernel currently accumulates `(I_star - I_planet*A_planet)/I_star`, which is `1 - blocked/I_star`. Each becomes `-(I_planet*A_planet)/I_star`, and each out-of-window initialiser becomes `0.0`. Because `dF` and `F` differ by a *constant*, no gradient column changes sign and no gradient arithmetic changes — the only gradient edits are two limb-darkening terms that currently reconstruct the deficit as `(1.0 - flux)` and now read `-flux` directly. The log-likelihood forms its residual as `(obs - baseline) - dF` and gains a fifth scalar reduction slot for `d lnL/d baseline`.

**Tech Stack:** Python 3.10+, NumPy, Numba (`@njit`, `prange`), JAX (`jnp`, `lax.fori_loop`), OpenCL C (`.cl` kernels via pyopencl), pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-09-04-flux-deficit-convention-design.md`

## Global Constraints

- Sign convention is `dF = F - 1`: **negative** in transit, mid-transit approximately `-k^2`. Never `1 - F`.
- Out-of-transit flux must be **exactly** `0.0`, not approximately.
- `UniformDiskModel` already conforms on all three backends. **Do not modify it.**
- `SecondaryEclipseModel` keeps its own planet-surface-brightness convention. **Do not modify it.**
- Gradients w.r.t. `k, t0, p, a, i, e, w, ldc` are **unchanged** by this work. If a gradient test starts failing, that is a bug in the change, not an expected update.
- In `.cl` sources every literal must be cast `(REAL)...` so the fp32 build does not silently promote.
- Do not add `-cl-fast-relaxed-math`; it breaks agreement with the Numba reference and defeats the Kahan compensation in `reduction.cl`.
- Residual order is `(obs - baseline) - dF`, never `obs - (baseline + dF)`. The second form re-creates a value near 1.0 and silently discards the entire numerical benefit while passing every correctness test.
- Run tests file by file (`python -m pytest tests/test_x.py`), matching this repo's existing guidance.

---

### Task 1: RoadRunner returns the flux deficit

All three backends change together. This is deliberate: `tests/test_jax_rrmodel.py` and `tests/test_opencl_rrmodel*.py` compare against the Numba backend, so converting one backend alone leaves the suite red for reasons unrelated to correctness.

**Files:**
- Modify: `pytransit/backends/numba/rrmodel.py:228`, `:234-235`, `:239`
- Modify: `pytransit/backends/numba/rrmodel_grad.py:135`, `:147`, `:152`, `:193`
- Modify: `pytransit/backends/jax/rrmodel.py:271`, `:276`
- Modify: `pytransit/backends/opencl/rrmodel.cl:31`, `:128`, `:142`, `:145`, `:206`, `:236`, `:268`
- Test: `tests/test_numba_rrmodel.py`, `tests/test_numba_rrmodel_grad.py`, `tests/test_jax_rrmodel.py`, `tests/test_opencl_rrmodel.py`, `tests/test_opencl_rrmodel_grad.py`, `tests/test_opencl_rrmodel_class.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RoadRunnerModel.evaluate(...)` returning `dF` with out-of-transit exactly `0.0`; mid-transit approximately `-k^2`. Gradient shape and column order are unchanged: `[k, t0, p, a, i, e, w, ldc...]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_numba_rrmodel.py`:

```python
def test_out_of_transit_flux_is_exactly_zero(self):
    """The deficit convention: out of transit the model contributes nothing at all."""
    from pytransit.models.rrmodel import RoadRunnerModel
    times = linspace(-0.5, 0.5, 500)          # far wider than the transit
    model = RoadRunnerModel()
    model.set_data(times)
    flux = model.evaluate(0.1, 0.0, P, A, INC, 0.0, 0.0, ldc=LDC)
    oot = npabs(times) > 0.2
    self.assertTrue(oot.any())
    # Exact equality, not a tolerance: 0.0 is representable and is the point of the convention.
    self.assertTrue((flux[oot] == 0.0).all())
    self.assertLess(flux.min(), -1e-3)         # in transit the deficit is negative
    self.assertAlmostEqual(flux.max(), 0.0, places=15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_numba_rrmodel.py -k out_of_transit -x -q`
Expected: FAIL — out-of-transit flux is `1.0`, so `(flux[oot] == 0.0).all()` is False.

- [ ] **Step 3: Convert the numba forward kernel**

In `pytransit/backends/numba/rrmodel.py`, replace lines 227-240:

```python
            if not ((bt1 - exptimes[ilc]) <= t <= (bt4 + exptimes[ilc])):
                flux[ipv, ipt] = 0.0
            else:
                kpb = k[ipv, ipb]
                for isample in range(1, nsamples[ilc] + 1):
                    time_offset = exptimes[ilc] * ((isample - 0.5) / nsamples[ilc] - 0.5)
                    z = sep_c(t + time_offset, xyc[iep])
                    if z >= 1.0 + kpb:
                        # An untransiting sample contributes no deficit. The branch is
                        # kept because it skips the lookup and the area computation.
                        continue
                    iplanet = ldm_lookup(z / (1.0 + kpb), gcs[ipb], n1, coef[ipb])
                    aplanet = ccia(1.0, kpb, z)
                    flux[ipv, ipt] += -(iplanet * aplanet) / ldi[ipv, ipb]
                flux[ipv, ipt] /= nsamples[ilc]
```

- [ ] **Step 4: Convert the numba gradient kernel**

In `pytransit/backends/numba/rrmodel_grad.py`:

Line 135 — `flux[ipv, ipt] = 1.0` becomes `flux[ipv, ipt] = 0.0`.

Lines 146-152 — the untransiting branch and the accumulation:

```python
                    if z >= 1.0 + kpb:
                        continue
                    aplanet, (dadk, dadz) = ccia_and_grad(1.0, kpb, z)
                    g = z / (1.0 + kpb)
                    iplanet, dIp_dg = ldm_lookup_and_grad(g, gc, n1, coef[ipb])
                    flux[ipv, ipt] += -(iplanet * aplanet) / ldi_pb
```

Line 193 — the limb-darkening term. `1.0 - flux_old` is exactly `-flux_new`, so the subtraction disappears:

```python
                # Add the dldi contribution for this passband's LD coefficients.
                # flux is now the deficit, so the old `(1.0 - flux)` reconstruction --
                # a catastrophic cancellation in single precision -- is just `-flux`.
                for j in range(nldc):
                    dflux[ipv, ipt, 7 + ipb * nldc + j] += -flux[ipv, ipt] * dldi[ipv, ipb, j] / ldi_pb
```

Leave every other `dflux[...]` line untouched.

- [ ] **Step 5: Convert the JAX kernel**

In `pytransit/backends/jax/rrmodel.py`, line 271:

```python
            sample_flux = -(iplanet * aplanet) / ldi_val
```

and line 276:

```python
        return jnp.where(in_window, flux, 0.0)
```

- [ ] **Step 6: Convert the OpenCL kernels**

In `pytransit/backends/opencl/rrmodel.cl`:

Line 31 (file-header comment) and line 142 (inline comment) both claim an untransiting sample "degenerates to (ldi-0)/ldi = 1". Update both, e.g.:

```c
   g > 1 returns 0 so that an untransiting sample degenerates to -0/ldi = 0, the
   correct out-of-transit deficit; the numba kernels guard the same case with `z >= 1 + k`.
```

Line 128 (forward kernel) and line 206 (gradient kernel) — both `REAL f = (REAL)1.0;` become:

```c
    REAL f = (REAL)0.0;
```

Line 145 (forward) and line 236 (gradient) — both accumulations:

```c
            f += -(ip * ap) / ldiv;
```

Line 268 — the limb-darkening term:

```c
        /* f is the deficit, so the old ((REAL)1.0 - f) reconstruction is just -f.
           That subtraction was a catastrophic cancellation in the fp32 build. */
        for (int j = 0; j < NLDC; j++)
            df[7 + j] += -f * dldi_pb[j] / ldiv;
```

- [ ] **Step 7: Run the new test to verify it passes**

Run: `python -m pytest tests/test_numba_rrmodel.py -k out_of_transit -x -q`
Expected: PASS

- [ ] **Step 8: Update the reference comparison in `tests/test_numba_rrmodel.py`**

`reference_flux(z, k)` integrates the limb-darkened disk exactly and returns *normalised* flux. Keep it that way and convert at the call site, so the convention is visible in the test rather than hidden in the reference. Around lines 57 and 69, replace `model.evaluate(...)` / `flux` with `1.0 + ...`:

```python
                worst = max(worst, 1e6 * npabs((1.0 + flux) - reference_flux(z, k)).max())
```

```python
            errs.append(npabs((1.0 + model.evaluate(0.2, 0.0, P, A, inc, 0.0, 0.0, ldc=LDC))
                              - reference_flux(z, 0.2)).max())
```

- [ ] **Step 9: Update the in-transit masks**

The model is now negative in transit, so `flux < 0.9999` never fires. Replace every occurrence with `flux < -1e-6`, matching what the OpenCL uniform-disk tests already use.

- `tests/test_numba_rrmodel_grad.py` lines 65, 86, 126, 182, 256: `in_transit = flux0 < 0.9999` becomes `in_transit = flux0 < -1e-6`
- `tests/test_opencl_rrmodel_class.py:61`: `assert (flux_nb < 0.9999).any(), ...` becomes `assert (flux_nb < -1e-6).any(), ...`
- `tests/test_opencl_rrmodel_grad.py:8` (module docstring): "RoadRunner returns normalised flux, so the in-transit mask is ``flux < 0.9999``" becomes "RoadRunner returns the flux deficit, so the in-transit mask is ``flux < -1e-6``". Update the mask itself wherever it appears in that file.
- `tests/test_jax_rrmodel.py:58`: `self.assertLess(1 - flux_numba.min(), 0.99)` asserted a real transit was present. It becomes `self.assertLess(flux_numba.min(), -1e-3)`.
- `tests/test_opencl_rrmodel.py:206`: the message says "reference flux should be physical once the ldc axis is right". Check the assertion it guards and convert any `> 0` / `<= 1` bound to the deficit range.

- [ ] **Step 10: Run every RoadRunner test file**

```bash
python -m pytest tests/test_numba_rrmodel.py -q
python -m pytest tests/test_numba_rrmodel_grad.py -q
JAX_PLATFORMS=cpu python -m pytest tests/test_jax_rrmodel.py -q
python -m pytest tests/test_opencl_rrmodel.py -q
python -m pytest tests/test_opencl_rrmodel_grad.py -q
python -m pytest tests/test_opencl_rrmodel_class.py -q
```

Expected: all PASS. The gradient tests must pass **without tolerance changes** — gradients are unchanged by this work, so a gradient failure means the conversion is wrong.

- [ ] **Step 11: Commit**

```bash
git add pytransit/backends/numba/rrmodel.py pytransit/backends/numba/rrmodel_grad.py \
        pytransit/backends/jax/rrmodel.py pytransit/backends/opencl/rrmodel.cl \
        tests/test_numba_rrmodel.py tests/test_numba_rrmodel_grad.py tests/test_jax_rrmodel.py \
        tests/test_opencl_rrmodel.py tests/test_opencl_rrmodel_grad.py tests/test_opencl_rrmodel_class.py
git commit -m "feat!: RoadRunnerModel returns the flux deficit instead of normalised flux

Out-of-transit flux is now exactly 0.0 and mid-transit is about -k**2.
Gradients are unchanged: dF and F differ by a constant. The limb-darkening
gradient term drops its (1.0 - flux) reconstruction, which was a catastrophic
cancellation costing ULP(1)/depth relative error in the fp32 build."
```

---

### Task 2: TransmissionSpectroscopyModel returns the flux deficit

**Files:**
- Modify: `pytransit/backends/numba/tsmodel.py:81`, `:93`, `:102`
- Modify: `pytransit/backends/numba/tsmodel_grad.py:161`, `:194`, `:241`, `:276`
- Modify: `pytransit/backends/jax/tsmodel.py:118`, `:125`
- Test: `tests/test_tsmodel_grad.py`

**Interfaces:**
- Consumes: the convention established in Task 1.
- Produces: `TransmissionSpectroscopyModel.evaluate(k, ldc, t0, p, ...)` returning `dF` of shape `(npv, npb, npt)`, out-of-transit exactly `0.0`. Gradient column order unchanged: `[k, t0, p, a, i, e, w, ldc..., kmean]`.

- [ ] **Step 1: Write the failing test**

`tests/test_tsmodel_grad.py` exercises the kernel directly, not the model class, and
`TestTSModelAndGrad` already provides a `self._call()` helper returning `(flux, dflux)` from fixed
fixtures. Add this method to that class:

```python
    def test_out_of_transit_flux_is_exactly_zero(self):
        """The deficit convention: outside the transit the model contributes nothing at all."""
        flux, _ = self._call()
        # Exact equality, not a tolerance: 0.0 is representable and being exactly zero
        # out of transit is the point of the convention.
        self.assertEqual(flux.max(), 0.0)
        self.assertTrue((flux == 0.0).any(),
                        'test setup should include out-of-transit points')
        self.assertLess(flux.min(), -1e-3, 'a real transit should be present')
```

The assertion is written against the flux itself rather than a time cut, so it does not depend on
recomputing `T14` for the fixture's eccentric orbit.

Note: the model-level `TransmissionSpectroscopyModel` is not constructed anywhere in the repo, and it
derives `npb` from `ldc` inside `evaluate` (`models/tsmodel.py:227`) rather than from `set_data`. Do
not invent a model-level setup here; the kernel-level assertion is what this task needs.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tsmodel_grad.py -k out_of_transit -x -q`
Expected: FAIL — out-of-transit flux is `1.0`.

- [ ] **Step 3: Convert the numba forward kernel**

In `pytransit/backends/numba/tsmodel.py`:

Line 81 — `flux[ipv, :, ipt] = 1.0` becomes `flux[ipv, :, ipt] = 0.0`.

Line 93 (inner, fully-contained case):

```python
                            flux[ipv, ipb, ipt] += -(iplanet * ap0 * afac[ipb] * inv_istar[ipb]) * inv_nsamples
```

Line 102 (edge case):

```python
                            flux[ipv, ipb, ipt] += -(iplanet * ap_b * inv_istar[ipb]) * inv_nsamples
```

- [ ] **Step 4: Convert the numba gradient kernel**

In `pytransit/backends/numba/tsmodel_grad.py`:

Line 161 — `flux[ipv, :, ipt] = 1.0` becomes `flux[ipv, :, ipt] = 0.0`.

Line 194:

```python
                            flux[ipv, ipb, ipt] += -(iplanet * aeff * inv_is) * inv_nsamples
```

Line 241:

```python
                            flux[ipv, ipb, ipt] += -(iplanet * aplanet * inv_is) * inv_nsamples
```

Line 276 — the `distar` term, same cancellation removal as RoadRunner:

```python
                        # flux is the deficit, so the old `(1.0 - flux)` reconstruction is `-flux`.
                        dflux[ipv, ipb, ipt, 7 + j] += -flux[ipv, ipb, ipt] * distar[ipv, ipb, j] * inv_is
```

Leave every other `dflux[...]` line untouched.

- [ ] **Step 5: Convert the JAX kernel**

In `pytransit/backends/jax/tsmodel.py`, line 118:

```python
                return -(iplanet * area_adj) / istar_pb
```

and line 125:

```python
        return jnp.where(in_window, flux, 0.0)
```

- [ ] **Step 6: Run the tests**

```bash
python -m pytest tests/test_tsmodel_grad.py -q
```

Expected: PASS, including the new test and every pre-existing gradient assertion without tolerance changes.

- [ ] **Step 7: Commit**

```bash
git add pytransit/backends/numba/tsmodel.py pytransit/backends/numba/tsmodel_grad.py \
        pytransit/backends/jax/tsmodel.py tests/test_tsmodel_grad.py
git commit -m "feat!: TransmissionSpectroscopyModel returns the flux deficit"
```

---

### Task 3: Pin the convention across every model and backend

One test that fails loudly if any backend drifts, and that documents which models deliberately differ.

**Files:**
- Create: `tests/test_flux_convention.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the test**

```python
"""The library-wide flux convention: models return the deficit dF = F - 1.

Out-of-transit flux is exactly 0.0 -- an equality, not a tolerance, because 0.0 is
representable and being exactly zero is the point of the convention.

SecondaryEclipseModel is deliberately excluded from the deficit assertion: it returns
the planet's visible flux on the planet's own surface-brightness scale (out-of-eclipse
pi*k**2, zero at total eclipse), which is already a contribution rather than a
normalised flux. It gets its own test here so the exception is recorded rather than
silently tolerated.

TransmissionSpectroscopyModel is covered at kernel level in test_tsmodel_grad.py
instead: nothing in the repo constructs it through the model class, and it takes npb
from `ldc` inside evaluate() rather than from set_data().
"""
import unittest

import numpy as np
from numpy import linspace, array

from pytransit.models.rrmodel import RoadRunnerModel
from pytransit.models.udmodel import UniformDiskModel

P, A, INC, K = 3.0, 6.0, 1.45, 0.10
LDC = array([0.30, 0.10])
TIMES = linspace(-0.5, 0.5, 500)
OOT = np.abs(TIMES) > 0.2


def _backends(model_cls):
    """Every backend the model supports on this machine."""
    available = ['numba', 'jax']
    try:
        import pyopencl as cl
        if cl.get_platforms():
            available.append('opencl')
    except Exception:
        pass
    return available


class TestFluxDeficitConvention(unittest.TestCase):
    def _check(self, flux, name):
        flux = np.asarray(flux, float)
        self.assertTrue(OOT.any(), 'test setup should have out-of-transit points')
        self.assertTrue((flux[OOT] == 0.0).all(),
                        f'{name}: out-of-transit flux must be exactly 0.0, '
                        f'got max |flux| = {np.abs(flux[OOT]).max()}')
        self.assertLess(flux.min(), -1e-3, f'{name}: in-transit deficit should be negative')
        self.assertLessEqual(flux.max(), 0.0, f'{name}: deficit must never be positive')

    def test_roadrunner(self):
        for backend in _backends(RoadRunnerModel):
            with self.subTest(backend=backend):
                m = RoadRunnerModel(backend=backend)
                m.set_data(TIMES)
                self._check(m.evaluate(K, 0.0, P, A, INC, 0.0, 0.0, ldc=LDC),
                            f'RoadRunnerModel/{backend}')

    def test_uniform_disk_already_conformed(self):
        """UniformDiskModel used this convention before the change; it must still hold."""
        for backend in _backends(UniformDiskModel):
            with self.subTest(backend=backend):
                m = UniformDiskModel(backend=backend)
                m.set_data(TIMES)
                self._check(m.evaluate(K, 0.0, P, A, INC, 0.0, 0.0),
                            f'UniformDiskModel/{backend}')

    def test_secondary_eclipse_keeps_its_own_scale(self):
        """Documents the deliberate exception rather than silently tolerating it."""
        from pytransit.models.semodel import SecondaryEclipseModel
        m = SecondaryEclipseModel()
        m.set_data(TIMES)
        flux = np.asarray(m.evaluate(K, 0.0, P, A, INC, 0.0, 0.0), float)
        self.assertAlmostEqual(flux[OOT].max(), np.pi * K ** 2, places=10)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run it**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_flux_convention.py -q`
Expected: PASS. If `UniformDiskModel.evaluate` does not take `ldc`, check its signature and adjust; if `SecondaryEclipseModel.evaluate` needs `rstar`, pass the default.

- [ ] **Step 3: Commit**

```bash
git add tests/test_flux_convention.py
git commit -m "test: pin the flux deficit convention across models and backends"
```

---

### Task 4: Log-likelihood baseline and its gradient column

**Files:**
- Modify: `pytransit/models/transitmodel.py:189-260` (`loglikelihood`)
- Modify: `pytransit/backends/opencl/reduction.cl:30-34` (slot defines), `:83` (residual), and the `if` chain around `:76-96`
- Modify: `pytransit/backends/opencl/_base.py:247` (`reduce_loglikelihood` signature), and the `nslot = 4 + col.size` / `pad = zeros(4, int32)` lines around `:296-306`
- Modify: `pytransit/loglikelihoods/wnloglikelihood.py:25-50`
- Test: `tests/test_opencl_loglikelihood.py`

**Interfaces:**
- Consumes: Tasks 1 and 2 (`evaluate` returns `dF`).
- Produces:
  - `TransitModel.loglikelihood(obs, err, jitter, k, t0, p, a, i, e=0.0, w=0.0, baseline=1.0, **kwargs) -> (lnl, grad)`
  - `grad` columns: `evaluate`'s Jacobian columns, then `d lnL/d baseline`, then `d lnL/d jitter`. **`grad[..., -1]` remains the jitter derivative.**
  - `CLModel.reduce_loglikelihood(jitter, baseline, slots, ngrad, npv) -> ndarray` of shape `(npv, 5 + n_gradient_slots)`.

- [ ] **Step 1: Extend the NumPy oracle**

`tests/test_opencl_loglikelihood.py:34` defines `reference_loglikelihood(model, obs, err, jitter, **pars)`, the host-side oracle every test in the file compares against. It must learn about `baseline`, or the whole file breaks once the device kernel changes. Replace it with:

```python
def reference_loglikelihood(model, obs, err, jitter, baseline=1.0, **pars):
    """NumPy oracle: reduce the model's own expanded Jacobian on the host.

    The residual is (obs - baseline) - dF, matching reduction.cl. Gradient columns are
    the Jacobian's, then d lnL/d baseline, then d lnL/d jitter -- jitter stays last.
    """
    flux, jac = model.evaluate(**pars)
    flux, jac = np.asarray(flux), np.asarray(jac)
    e2 = np.asarray(err) ** 2 + jitter ** 2
    r = (np.asarray(obs) - baseline) - flux
    lnl = (-0.5 * r.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(e2))
           - 0.5 * np.sum(r ** 2 / e2))
    grad = (r / e2) @ jac
    db = np.sum(r / e2)
    dj = jitter * (np.sum(r ** 2 / e2 ** 2) - np.sum(1.0 / e2))
    return lnl, np.concatenate([grad, [db], [dj]])
```

Every existing call site passes normalised observations, so the `baseline=1.0` default keeps them correct once the models return deficits. Also check `self._lnl(...)`, the forward-only helper used by the finite-difference tests, and give it the same `baseline` handling.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_opencl_loglikelihood.py`, in the `TestReferenceMathIsRight` class, mirroring the existing `test_jitter_gradient` at line 114:

```python
    def test_baseline_gradient(self):
        times, err, rng = _dataset()
        grad_model, fwd_model = _rr(times), _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, baseline, eps = 5e-4, 1.0, 1e-6

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, baseline, **self.PARS)
        fd = (self._lnl(fwd_model, obs, err, jitter, baseline + eps)
              - self._lnl(fwd_model, obs, err, jitter, baseline - eps)) / (2 * eps)
        # baseline is second from last; jitter keeps the last column.
        np.testing.assert_allclose(grad[-2], fd, rtol=1e-5)

    def test_jitter_is_still_the_last_column(self):
        times, err, rng = _dataset()
        grad_model, fwd_model = _rr(times), _rr(times, return_grad=False)
        obs = 1.0 + np.asarray(fwd_model.evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, eps = 5e-4, 1e-9

        _, grad = reference_loglikelihood(grad_model, obs, err, jitter, 1.0, **self.PARS)
        fd = (self._lnl(fwd_model, obs, err, jitter + eps, 1.0)
              - self._lnl(fwd_model, obs, err, jitter - eps, 1.0)) / (2 * eps)
        np.testing.assert_allclose(grad[-1], fd, rtol=1e-5)
```

And in `TestDeviceMatchesReference`, assert the device agrees with the extended oracle:

```python
    def test_device_matches_reference_with_baseline(self):
        times, err, rng = _dataset()
        model = _rr(times)
        obs = 1.0 + np.asarray(_rr(times, return_grad=False).evaluate(**self.PARS)) + rng.normal(0, err)
        jitter, baseline = 5e-4, 0.9997

        want_lnl, want_grad = reference_loglikelihood(model, obs, err, jitter, baseline, **self.PARS)
        got_lnl, got_grad = model.loglikelihood(obs, err, jitter, baseline=baseline, **self.PARS)
        np.testing.assert_allclose(got_lnl, want_lnl, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(got_grad), want_grad, rtol=1e-8, atol=1e-12)
```

Note `obs` now needs the explicit `1.0 +` because `evaluate` returns a deficit and these observations are normalised photometry. Every existing `obs = np.asarray(fwd_model.evaluate(...)) + rng.normal(...)` line in the file needs the same treatment — grep for `rng.normal` and fix each.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_opencl_loglikelihood.py -k baseline -x -q`
Expected: FAIL with `TypeError: loglikelihood() got an unexpected keyword argument 'baseline'`.

- [ ] **Step 4: Add the fifth scalar slot to the reduction kernel**

In `pytransit/backends/opencl/reduction.cl`, extend the defines at lines 30-34:

```c
#define SLOT_CHI2       0   /* sum r^2/e2   */
#define SLOT_LOG_E2     1   /* sum log(e2)  */
#define SLOT_CHI2_E2    2   /* sum r^2/e2^2 */
#define SLOT_INV_E2     3   /* sum 1/e2     */
#define SLOT_R_E2       4   /* sum r/e2  -> d lnL/d baseline */
#define NSCALAR         5
```

Add `const REAL baseline` to the `wn_loglikelihood` signature immediately after `const REAL jitter`, then change the residual and add the new branch:

```c
            REAL r = (obs[ipt] - baseline) - flux[(size_t)ipv * npt + ipt];
            if (s == SLOT_CHI2) {
                term = r * r / e2;
            } else if (s == SLOT_CHI2_E2) {
                term = r * r / (e2 * e2);
            } else if (s == SLOT_R_E2) {
                term = r / e2;                    /* d lnL/d baseline */
            } else {
```

The residual ordering is load-bearing: `(obs - baseline)` with both near 1.0 is exact by the Sterbenz lemma, and everything after stays on the deficit scale. Add a comment saying so. The new slot inherits the kernel's Kahan compensation automatically.

- [ ] **Step 5: Thread `baseline` through the host reduction**

In `pytransit/backends/opencl/_base.py`, change `reduce_loglikelihood(self, jitter: float, slots, ngrad: int, npv: int)` to `reduce_loglikelihood(self, jitter: float, baseline: float, slots, ngrad: int, npv: int)`, document `baseline` in the docstring, and update the three places that hard-code the scalar count:

```python
        nslot = 5 + col.size
```

```python
            # The five scalar sums occupy the leading slots; their col/kind/grp
            # are unused by the kernel but must be present so the arrays line up.
            pad = zeros(5, int32)
```

and add the argument to the kernel launch, immediately after the jitter argument:

```python
            int32(self.npt), int32(ngrad), int32(nslot),
            self.dtype(jitter), self.dtype(baseline),
```

Also update the docstring's "Shape ``(npv, 4 + n_gradient_slots)``" to `5 + n_gradient_slots`.

- [ ] **Step 6: Update `TransitModel.loglikelihood`**

In `pytransit/models/transitmodel.py`, add `baseline: float = 1.0` to the signature (before `**kwargs`), document it, and change the unpacking and gradient assembly:

```python
        sums = self._model.reduce_loglikelihood(jitter, baseline, slots, ncol, npv)

        # Slot order matches reduction.cl: chi2, log_e2, chi2_e2, inv_e2, r_e2,
        # then one reduced column per gradient slot.
        chi2, log_e2, chi2_e2, inv_e2, r_e2 = (sums[:, j] for j in range(5))
        lnl = -0.5 * self.npt * log(2.0 * pi) - 0.5 * log_e2 - 0.5 * chi2
        djitter = jitter * (chi2_e2 - inv_e2)
        dbaseline = r_e2                      # d lnL/d baseline = sum(r/e2)
        # baseline goes before jitter so that grad[..., -1] is still d lnL/d jitter.
        grad = concatenate([sums[:, 5:], dbaseline[:, None], djitter[:, None]], axis=1)
```

Add to the docstring: the `baseline` parameter, that the residual is `(obs - baseline) - dF`, and that the gradient now carries `d lnL/d baseline` followed by `d lnL/d jitter`.

- [ ] **Step 7: Update `WNLogLikelihood`**

In `pytransit/loglikelihoods/wnloglikelihood.py`, the JAX closure gains `baseline` before `jitter` so the gradient order matches:

```python
            def loglike(k, t0, p, a, i, e, w, baseline, jitter):
                fmod = model(k, t0, p, a, i, e, w)
                e2 = self.err**2 + jitter**2
                r = (self.obs - baseline) - fmod
                return (-0.5 * self.obs.size * jnp.log(2 * jnp.pi)
                        - 0.5 * jnp.sum(jnp.log(e2)) - 0.5 * jnp.sum(r ** 2 / e2))

            if self.tm.return_grad:
                self._loglikelihood = jax.jit(jax.value_and_grad(loglike, argnums=tuple(range(9))))
            else:
                self._loglikelihood = jax.jit(loglike)
```

and the OpenCL closure forwards it:

```python
            def loglike(k, t0, p, a, i, e=0.0, w=0.0, jitter=0.0, baseline=1.0, **kwargs):
                return self.tm.loglikelihood(self.obs, self.err, jitter,
                                             k, t0, p, a, i, e, w,
                                             baseline=baseline, **kwargs)
```

Update the class docstring: the noise model description gains the baseline, and the gradient description becomes "one column per model parameter as passed, then ``d lnL/d baseline``, then ``d lnL/d jitter``".

- [ ] **Step 8: Run the tests**

```bash
python -m pytest tests/test_opencl_loglikelihood.py -q
python -m pytest tests/test_transitmodel.py -q
```

Expected: PASS. If `test_opencl_loglikelihood.py` has existing assertions on the gradient width, they need `+1`.

- [ ] **Step 9: Commit**

```bash
git add pytransit/models/transitmodel.py pytransit/backends/opencl/reduction.cl \
        pytransit/backends/opencl/_base.py pytransit/loglikelihoods/wnloglikelihood.py \
        tests/test_opencl_loglikelihood.py
git commit -m "feat!: add a fitted baseline to the white-noise log-likelihood

The residual is (obs - baseline) - dF, computed in that order so the near-1
subtraction happens once and exactly. d lnL/d baseline = sum(r/e2) rides in a
fifth scalar reduction slot; the column sits before jitter so grad[..., -1] is
still the jitter derivative."
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:296` (Key Conventions), the OpenCL "Porting gotchas" bullet, the "RoadRunner specifics" list, the device-side log-likelihood paragraph

**Interfaces:**
- Consumes: Tasks 1, 2 and 4.
- Produces: nothing.

- [ ] **Step 1: Rewrite the Key Conventions line**

Line 296, `- Flux is normalised (out-of-transit = 1.0)`, becomes:

```markdown
- Models return the flux **deficit** `dF = F - 1`: out-of-transit is exactly `0.0` and mid-transit is
  about `-k**2`. Observed flux is `baseline + dF`. This is a 3.0 breaking change, made because storing
  a value near 1.0 costs ~1 ULP(1) of absolute error regardless of transit depth, so in single
  precision the error relative to the signal grows as 1/depth (measured: 8.5e-4 at k=0.01). The
  deficit sits log2(1/depth) binades lower and recovers those bits. `SecondaryEclipseModel` is the
  documented exception: it returns the planet's visible flux on the planet's own surface-brightness
  scale (out-of-eclipse `pi*k**2`).
```

- [ ] **Step 2: Fix the OpenCL porting gotcha**

The bullet beginning "`flux` is a *deficit*: it starts at 0 and is only decremented..." currently frames this as a uniform-disk quirk that differs from RoadRunner. It is now the library-wide convention. Rewrite it to say so, and keep the warning about the deleted v2 `ma_uniform.cl` returning `1 - lambdae`.

- [ ] **Step 3: Fix the RoadRunner specifics list**

Delete the first entry, "Flux is **normalised** (1.0 out of transit); the uniform disk returns a deficit (0.0). Test masks are `flux < 0.9999`, not `flux < -1e-6`." — RoadRunner and the uniform disk now agree. Replace the sentence that introduces the list ("Three RoadRunner conventions differ...") with "Two RoadRunner conventions differ...", and state the mask once: in-transit masks are `flux < -1e-6` for every model.

- [ ] **Step 4: Update the device-side log-likelihood paragraph**

That paragraph describes slots 0-3 as the scalar sums and says the gradient has "`evaluate`'s Jacobian columns plus a trailing `d lnL / d jitter`". Update it: there are now five scalar sums (the fifth is `sum r/e2`), the noise model is `r = (obs - baseline) - dF`, and the gradient carries `d lnL/d baseline` then `d lnL/d jitter`. Note that the residual ordering is deliberate and must not be "simplified".

- [ ] **Step 5: Update the testing-gradients guidance**

The "Testing gradients" paragraph says to compare "only where the model is smooth (e.g. `flux < 0.99`)". Change that example to `flux < -1e-6`.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: describe the flux deficit convention and the likelihood baseline"
```

---

### Task 6: Verify the numerical payoff

The change exists for a measured reason; this task confirms the reason actually materialised rather than assuming it.

**Files:**
- Modify: `benchmarks/check_precision.py` (the reference is now a deficit)
- Test: the whole suite

**Interfaces:**
- Consumes: every previous task.
- Produces: nothing.

- [ ] **Step 1: Fix the benchmark's depth calculation**

`benchmarks/check_precision.py` computes `depth = 1.0 - ref_f.min()` and masks with `ref_f < 0.9999`. With the deficit both are wrong. In `compare()`:

```python
def compare(ref_f, ref_d, got_f, got_d):
    """Flux error, and per-column Jacobian relative error over the in-transit points."""
    mask = ref_f < -1e-6                        # models return the deficit dF = F - 1
    depth = -ref_f.min()
```

and in `check_loglikelihood()` the synthetic observations must move to deficit space — `obs = flux + ERR * rng.normal(...)` still works because `flux` is now `dF`, but the call gains `baseline=0.0` since those observations carry no offset:

```python
            lnl, grad = m.loglikelihood(obs, ERR, JITTER, *pars, ldc=LDC, baseline=0.0)
```

The reference gradient gains a baseline entry before the jitter entry:

```python
        ref_dbaseline = np.sum(r)
        ref_full = np.concatenate([ref_grad, [ref_dbaseline], [ref_djit]])
```

where `r = (obs - 0.0 - flux) / e2` as already computed.

- [ ] **Step 2: Run the precision check**

Run: `python benchmarks/check_precision.py`
Expected: OpenCL `single` flux error drops from ~7.8e-06 of depth to the 1e-07 range, and the `ld0`/`ld1` Jacobian columns drop from ~9e-06 to roughly 1e-07. The double-precision rows must be unchanged at ~1e-14.

If the fp32 flux error does **not** improve, stop: the deficit is being re-normalised somewhere before it reaches the API boundary, and the change has not achieved its purpose.

- [ ] **Step 3: Run the full test suite file by file**

```bash
for f in tests/test_*.py; do echo "== $f"; JAX_PLATFORMS=cpu python -m pytest "$f" -q || echo "FAILED $f"; done
```

Expected: no `FAILED` lines. Investigate any failure rather than adjusting tolerances — gradients are unchanged by this work, so a gradient failure is a real bug.

- [ ] **Step 4: Confirm cross-backend agreement is untouched**

Run: `python -m pytest tests/test_opencl_rrmodel.py tests/test_opencl_rrmodel_grad.py -q`
Expected: PASS at the existing ~1e-14 tolerances, with no tolerance loosening anywhere in the diff.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/check_precision.py
git commit -m "test: measure the fp32 accuracy gain from the flux deficit convention"
```

---

## Self-review notes

- The spec's `UniformDiskModel` and `SecondaryEclipseModel` "no change" decisions are covered by Task 3, which asserts both rather than leaving them untested.
- The spec's risk "a missed call site would break quietly" is covered by Task 3 (convention pinned per backend) and Task 6 Step 3 (whole suite).
- The spec's risk about the `(obs - baseline) - dF` ordering is covered by a required comment in Task 4 Step 3 and a Global Constraint.
- `models/_utils.py::_expand_gradient` needs no change: it touches `flux` only through `isnan()`.
- `tests/reference_flux.npz` is unused by any test and is not touched.
- Out of scope, tracked separately: `doc/source/notebooks/models/roadrunner/roadrunner_model_theory.ipynb` also documents the normalised convention.
