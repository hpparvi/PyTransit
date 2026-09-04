# Flux deficit convention for PyTransit 3

**Date:** 2026-09-04
**Status:** approved design, not yet implemented
**Scope:** `RoadRunnerModel`, `TransmissionSpectroscopyModel`, the white-noise log-likelihoods, the
test suite, and `CLAUDE.md`.

## Summary

Transit models return the **flux deficit** `dF = F - 1` instead of the normalised absolute flux `F`.
Out-of-transit flux becomes exactly `0.0`; mid-transit is approximately `-k^2`.

This is a breaking API change, taken deliberately during the 3.0 major release.

## Motivation

The change is a numerical one, and it was measured rather than assumed.

In single precision the absolute error of the RoadRunner flux is **flat at ~1.4 ULP of 1.0
(8.5e-08) across a 400x range of transit depths**:

| k | depth | max abs error | in ULP(1) | relative to depth |
|---|---|---|---|---|
| 0.01 | 1.0e-04 | 8.63e-08 | 1.45 | **8.5e-04** |
| 0.05 | 2.5e-03 | 8.37e-08 | 1.40 | 3.3e-05 |
| 0.20 | 4.0e-02 | 8.96e-08 | 1.50 | 2.2e-06 |

That flatness is the diagnosis: the error is the grid spacing of float32 near 1.0 (5.96e-08), not
anything the model computes. Because `dF` sits `log2(1/depth)` binades below 1.0, storing it instead
recovers those mantissa bits. Measured representation floors give a gain of **16x at k=0.2 rising to
8191x at k=0.01** -- largest exactly where the present scheme is worst. Supersampling makes the
current scheme worse still (1.4 -> 3.2 ULP at `nsamples=10`), because `ns` values of magnitude ~1 are
summed before dividing.

The gradient kernels have a second, related defect. `opencl/rrmodel.cl:268` and
`numba/rrmodel_grad.py:193` compute the limb-darkening term as `(1.0 - f) * dldi / ldi`, reconstructing
the deficit by subtracting the accumulated near-1 flux from 1 -- catastrophic cancellation. The
predicted relative error `ULP(1)/depth` matches measurement closely:

| k | ld0 column | ld1 column | ULP(1)/depth |
|---|---|---|---|
| 0.01 | 4.7e-04 | 2.3e-04 | 5.9e-04 |
| 0.10 | 8.9e-06 | 6.0e-06 | 5.9e-06 |

Carrying the deficit removes that subtraction entirely.

Double precision is unaffected either way (error 2.2e-16), so this is a change for the fp32 path and
for any future reduced-precision work. It also unifies a convention that is currently inconsistent
across the library.

## Decisions

Three decisions were settled before design; they are recorded here because each has a defensible
alternative.

1. **Sign: `dF = F - 1`, negative in transit.** Chosen over `1 - F` because it matches the existing
   `UniformDiskModel`, keeps every gradient column's sign unchanged (`d(F-1)/dtheta = dF/dtheta`), and
   composes additively as `F_observed = baseline + dF`.
2. **`SecondaryEclipseModel` is left alone.** It returns the planet's visible flux on the planet's own
   surface-brightness scale (out-of-eclipse `pi*k^2`, zero at total eclipse). It is already zero-based,
   already a contribution rather than a normalised flux, and has no near-1 precision problem. Its
   documented "scale by the surface-brightness ratio, then add" workflow keeps working unchanged.
3. **Log-likelihoods gain a `baseline` scalar, default 1.0**, and form the residual as
   `(obs - baseline) - dF`. Users keep passing raw normalised photometry. `baseline` is a **fitted
   parameter**: it carries its own gradient column, so an out-of-transit normalisation can be sampled
   alongside the model parameters.

## Current state

Not every model needs changing. Verified by reading the kernels:

| Model | Out-of-transit today | Action |
|---|---|---|
| `UniformDiskModel` | `0.0` (numba, JAX, OpenCL all accumulate `-area/pi` from zero) | **none -- already conforms** |
| `RoadRunnerModel` | `1.0` | convert |
| `TransmissionSpectroscopyModel` | `1.0` | convert |
| `SecondaryEclipseModel` | `pi*k^2` (planet scale) | **none -- by decision** |

## Design

### The kernel change

Every forward kernel currently accumulates a per-sample contribution of the form
`(I_star - I_planet * A_planet) / I_star`, which is `1 - blocked/I_star`. The change is to accumulate
`-(I_planet * A_planet) / I_star` and to initialise the out-of-window value to `0.0` rather than `1.0`.

Because `dF` and `F` differ by a constant, **no gradient column changes sign and no gradient
arithmetic changes**. This is the property that keeps the change tractable, and it means the existing
gradient tests remain valid as correctness evidence.

Two gradient terms do change, and both get simpler: the limb-darkening coefficient term
`(1.0 - flux) * dldi / ldi` becomes `-flux * dldi / ldi`, since `1 - F_old` is exactly `-dF`.

### Files and edits

Line numbers are indicative, from the state at the time of writing.

**RoadRunner, numba forward** (`backends/numba/rrmodel.py`)
- `:228` out-of-window `flux[ipv, ipt] = 1.0` -> `0.0`.
- `:234-235` the `z >= 1 + k` early-out contributes `+= 1.0` -> contributes nothing. Keep the branch:
  it exists to skip work, and under the new convention its contribution is zero.
- `:239` `flux += (ldi - iplanet*aplanet) / ldi` -> `flux += -(iplanet*aplanet) / ldi`.
- `:240` the `/= nsamples` averaging is unchanged and remains correct (`0/n == 0`).

**RoadRunner, numba gradient** (`backends/numba/rrmodel_grad.py`)
- `:135`, `:147`, `:152` as above.
- `:193` `(1.0 - flux[ipv, ipt]) * dldi[...] / ldi_pb` -> `-flux[ipv, ipt] * dldi[...] / ldi_pb`.
- No other gradient line changes.

**Transmission spectroscopy, numba** (`backends/numba/tsmodel.py`, `tsmodel_grad.py`)
- `tsmodel.py:81` and `tsmodel_grad.py:161` out-of-window `1.0` -> `0.0`.
- `tsmodel.py:93,102` and `tsmodel_grad.py:194,241` `(1.0 - X) * inv_nsamples` -> `-X * inv_nsamples`.
- `tsmodel_grad.py:276` `(1.0 - flux[...]) * distar[...]` -> `-flux[...] * distar[...]`.

**JAX** (`backends/jax/rrmodel.py`, `tsmodel.py`)
- `rrmodel.py:271` `sample_flux = (ldi_val - iplanet*aplanet) / ldi_val` -> `-(iplanet*aplanet) / ldi_val`.
- `rrmodel.py:276` `jnp.where(in_window, flux, 1.0)` -> `..., 0.0)`.
- `tsmodel.py:118` `(istar_pb - iplanet*area_adj) / istar_pb` -> `-(iplanet*area_adj) / istar_pb`.
- `tsmodel.py:125` `jnp.where(in_window, flux, 1.0)` -> `..., 0.0)`.

**OpenCL** (`backends/opencl/rrmodel.cl`)
- `REAL f = (REAL)1.0;` appears **twice** -- `:128` in the forward kernel and `:206` in the gradient
  kernel. Both become `(REAL)0.0`. The inner `f = (REAL)0.0` reset then becomes redundant but is kept
  for clarity.
- `:145` `f += (ldiv - ip*ap) / ldiv;` -> `f += -(ip*ap) / ldiv;`, and the same accumulation in the
  gradient kernel.
- The comment "an untransiting sample degenerates to (ldi-0)/ldi = 1" appears **twice**, at `:31` in the
  file header and `:142` inline. Both must be updated: the sample now degenerates to `-0/ldi = 0`,
  which is the correct out-of-transit value.
- `:268` `((REAL)1.0 - f) * dldi_pb[j] / ldiv` -> `-f * dldi_pb[j] / ldiv`.
- No change to the validity guard, which writes NaN for an invalid parameter vector.

### Log-likelihood

`TransitModel.loglikelihood(obs, err, jitter, k, t0, p, a, i, e, w, *, baseline=1.0, **kwargs)` and
`WNLogLikelihood` gain the `baseline` scalar. The residual is

```
r = (obs - baseline) - dF
```

evaluated in that order. `obs - baseline` with both near 1.0 is exact by the Sterbenz lemma, and
everything after it stays on the deficit scale, which is what preserves the fp32 benefit end to end.
Computing `obs - (baseline + dF)` instead would re-form a near-1 intermediate and throw the benefit
away; the ordering is load-bearing, not stylistic.

For `backends/opencl/reduction.cl`, `baseline` is passed as an additional `REAL` kernel argument and
`:83` becomes `REAL r = (obs[ipt] - baseline) - flux[...]`. It is deliberately *not* pre-subtracted on
the host: `obs` is uploaded through an identity-keyed cache, and pre-subtracting would force a
re-upload whenever the baseline changed.

#### The baseline gradient

With `r = (obs - baseline) - dF` and `dr/dbaseline = -1`,

```
d lnL / d baseline = sum(r / e2)
```

(positive; the sign follows from the two negatives cancelling, and it is derived the same way as the
existing `djitter = jitter * (chi2_e2 - inv_e2)`).

**It is a scalar sum, not a Jacobian contraction.** That matters for the implementation: it does not
need the slot machinery in `reduction.cl` that routes points to per-passband or per-epoch columns, and
it does not touch `TransitModel._gradient_slots` or the compact Jacobian width `ncol`. It slots into the
existing *scalar* block instead, which currently holds four sums:

| slot | sum |
|---|---|
| `SLOT_CHI2` | `sum r^2/e2` |
| `SLOT_LOG_E2` | `sum log e2` |
| `SLOT_CHI2_E2` | `sum r^2/e2^2` |
| `SLOT_INV_E2` | `sum 1/e2` |
| **`SLOT_R_E2`** (new) | **`sum r/e2`** |

So `NSCALAR` goes 4 -> 5, one branch is added to the kernel's `if` chain, and the host reads
`dbaseline = sums[:, 4]` exactly as it already computes `djitter` from `chi2_e2` and `inv_e2`. The new
slot inherits the kernel's Kahan compensation automatically. Gradient slots shift from `sums[:, 4:]` to
`sums[:, 5:]`.

**Column order:** `[evaluate's Jacobian columns] + [baseline] + [jitter]`. Baseline is inserted
*before* jitter, not appended after it, so that `grad[..., -1]` remains `d lnL / d jitter` and existing
code indexing the last column keeps working. Mean-model parameters stay grouped ahead of the noise
parameter.

`baseline` is a scalar, matching the existing treatment of `jitter` (`const REAL jitter` in the
kernel). Per-parameter-vector baselines are not supported, for the same reason per-vector jitter is
not; lifting both together is a separate change.

For the JAX `WNLogLikelihood`, the closure becomes
`loglike(k, t0, p, a, i, e, w, baseline, jitter)` with `argnums=range(9)`, which yields the column
order above directly. (Note the pre-existing limitation that the JAX path differentiates only the
parameters `get_callable()` exposes, so `ldc` is absent there; this change neither worsens nor fixes
that.)

For the numba path -- where `WNLogLikelihood` raises and users contract `evaluate()`'s Jacobian by hand
-- the baseline derivative is likewise `sum(r/e2)`, and the documentation must say so.

## Testing

The change is mostly invisible to the gradient tests, which is the main safety property: gradients are
unchanged, so those tests keep their full value as regression evidence.

**Mask updates.** In-transit masks `flux < 0.9999` become `flux < -1e-6`, matching what the OpenCL
uniform-disk tests already use. Affects `test_numba_rrmodel_grad.py` (5 sites),
`test_opencl_rrmodel_grad.py`, `test_opencl_rrmodel_class.py`, `test_jax_rrmodel.py`.

**Reference comparisons.** `test_numba_rrmodel.py` compares against a locally defined exact-integration
`reference_flux(z, k)`. Keep that reference in normalised space and compare `1.0 + model.evaluate(...)`,
so the convention change is explicit at the call site rather than buried in the reference. The same
applies to the physical-range assertion in `test_limb_darkening.py:302`.

**New invariant.** Add a test asserting out-of-transit flux is **exactly** `0.0` -- an equality
assertion, not a tolerance -- for every model and backend combination. Nothing currently pins this, and
it is precisely the property the change buys.

**Log-likelihood.** `test_opencl_loglikelihood.py` gains baseline coverage:
- the default `baseline=1.0` reproduces the pre-change log-likelihood for normalised observations;
- a non-unit baseline shifts the residual as expected;
- `d lnL / d baseline` matches central finite differences of `lnL` with respect to `baseline`, and
  `d lnL / d jitter` is still the last column;
- the gradient is unchanged in every other column when only `baseline` varies.

**Precision payoff.** Re-run `benchmarks/check_precision.py`. The predicted fp32 flux accuracy gain is
~64x at k=0.1 and ~8000x at k=0.01, and the ld-column gradient error should drop from ~9e-06 to the
1e-07 floor. This is the acceptance test for the whole change: if the gain does not materialise, the
deficit is not actually surviving to the API boundary.

**Cross-backend agreement** must remain at ~1e-14 between numba, JAX and OpenCL in double precision.

## Documentation

`CLAUDE.md`:
- Key Conventions `:296` "Flux is normalised (out-of-transit = 1.0)" -> the deficit convention, with the
  sign stated explicitly.
- The OpenCL "Porting gotchas" bullet describing `flux` as a deficit *unlike* RoadRunner is now wrong in
  its contrast; rewrite it as the single library-wide convention.
- "RoadRunner specifics": the three-way divergence list loses its first entry (normalised vs deficit),
  and the test-mask guidance becomes `flux < -1e-6` everywhere.
- Record the rationale in one or two sentences so the decision is documented rather than archaeological.
- Note that `SecondaryEclipseModel` deliberately keeps its own planet-scale convention.

**Out of scope:** `doc/source/notebooks/models/roadrunner/roadrunner_model_theory.ipynb` also describes
the normalised convention. Notebooks are not covered by this change and are tracked separately.

## Risks

- **Silent breakage for users.** Existing code that plots or fits `model.evaluate(...)` keeps running and
  produces a result that is wrong by 1.0. There is no way to make this loud from inside `evaluate()`. The
  mitigation is release-note prominence, not code.
- **Missed call site.** Any internal consumer that assumes normalisation and is not listed here would
  break quietly. `models/_utils.py::_expand_gradient` was checked and touches `flux` only via `isnan()`,
  so it is convention-independent. `tests/reference_flux.npz` is unused by any test.
- **The `baseline` ordering** is easy to "simplify" into `obs - (baseline + dF)` during review, which
  silently discards the numerical benefit while passing every correctness test. It needs a comment at
  the site saying why.

## Acceptance criteria

1. Out-of-transit flux is exactly `0.0` for `RoadRunnerModel` and `TransmissionSpectroscopyModel` on
   every backend; `UniformDiskModel` unchanged and still `0.0`; `SecondaryEclipseModel` unchanged.
2. Gradients are numerically identical to before the change, in double precision.
3. Cross-backend agreement holds at ~1e-14.
4. `WNLogLikelihood` with `baseline=1.0` on normalised observations reproduces the pre-change value.
5. `d lnL / d baseline` agrees with central finite differences, and `grad[..., -1]` is still
   `d lnL / d jitter`.
6. `benchmarks/check_precision.py` shows the predicted fp32 improvement.
7. Full test suite passes.
