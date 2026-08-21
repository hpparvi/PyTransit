# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTransit is an exoplanet transit light curve modelling library. It provides optimized transit models with Numba 
(JIT-compiled) and JAX backends for evaluating how starlight dims when a planet passes in front of its host star. All 
models support gradient computation for use in gradient-based inference. Orbital mechanics (projected star-planet
distance) are provided by the `meepmeep` dependency.

## Common Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_numba_udmodel.py

# Run a specific test
python -m pytest tests/test_numba_rrmodel_grad.py -k "test_name"
```

The OpenCL tests skip themselves when pyopencl or an OpenCL device is unavailable. The two JAX test
files (`test_jax_udmodel.py`, `test_grad_expansion_ud_se.py`) hard-abort the interpreter on GPUs
newer than JAX's bundled PTX supports (e.g. "PTX version 8.4 does not support target 'sm_120a'" on
Blackwell); that is an environment issue, not a test failure, and it kills the whole `pytest tests/`
run, so run the suite file by file when triaging.

The full suite takes ~2 min; most of that is Numba JIT compilation of the model kernels on first call, not the assertions.

## Architecture

### Package Structure

```
pytransit/
  models/           # Public API: model classes (rr, ud, ts, se, ldmodel)
  backends/
    numba/           # Numba implementations (forward + gradient)
    jax/             # JAX implementations (forward; gradients via JAX autodiff)
    opencl/          # OpenCL implementations (forward + gradient); udmodel, rrmodel
  loglikelihoods/   # LogLikelihood, WNLogLikelihood helpers
```

This is the PyTransit v3 (`__version__ == "3.0.0"`) rewrite: a Numba/JAX/OpenCL backend split,
per-light-curve heterogeneous data (passbands/epochs/supersampling), and analytic gradients
on every Numba model.

### TransitModel Base Class (`models/transitmodel.py`)

All models inherit from `TransitModel`:

- **Constructor:** `TransitModel(backend="numba", return_grad=False, parallel=False, n_threads=None)`
  - `backend` — `"numba"`, `"jax"`, or `"opencl"` (`UniformDiskModel`, `RoadRunnerModel`)
  - OpenCL-only kwargs, passed through `**kwargs`: `precision` (`"double"`/`"single"`),
    `cl_ctx`, `cl_queue`
  - `return_grad` — if `True`, `evaluate()` returns `(flux, jacobian)` instead of just `flux`
- **`set_data(times, lcids, pbids, epids, nsamples, exptimes, include_orbit_variations=False)`** — configure observation times and metadata (light curve IDs, passband IDs, epoch IDs, supersampling). `pbids` and `epids` must be contiguous integers `0..n-1`. Sets the shape attributes the kernels and gradient layout depend on:
  - `npb = unique(pbids).size` — passbands; `k` and `ldc` are per-passband.
  - `ntc = unique(epids).size` — transit centres; `t0` is per-epoch.
  - `nor` — orbit variations: `ntc` if `include_orbit_variations=True` (per-epoch `p,a,i,e,w`), else `1` (one orbit shared across all epochs). This `ntc`-vs-`nor` split is why `t0` can expand to many columns while the orbital params stay shared.
- **`evaluate(k, t0, p, a, i, e, w, ...)`** — compute model flux (and optionally gradients). A parameter passed as a scalar is shared; passing a per-passband (`k`, `ldc`) or per-epoch (`t0`, orbital) array expands it. `e=0, w=0` default to a circular orbit.

**Key parameters:**
- `k` — planet-to-star radius ratio
- `t0` — mid-transit time
- `p` — orbital period
- `a` — scaled semi-major axis (a/R*)
- `i` — orbital inclination
- `e`, `w` — eccentricity and argument of periastron

### Models

- **`RoadRunnerModel`** (`models/rrmodel.py`) — Recommended. Fast, supports any radially symmetric limb darkening via the `LDModel` helper. Numba + JAX + OpenCL backends.
- **`UniformDiskModel`** (`models/udmodel.py`) — Simplest model: uniform stellar disk (no limb darkening). Numba + JAX + OpenCL backends.
- **`TransmissionSpectroscopyModel`** (`models/tsmodel.py`) — Optimized for wavelength-dependent transit depth (`evaluate(k, ldc, t0, p, ...)` — note `ldc` is the 2nd arg here). Numba + JAX backends.
- **`SecondaryEclipseModel`** (`models/semodel.py`) — Eclipse (occultation) model. Numba backend only; `evaluate(..., rstar=1.0)` takes a stellar radius for the light-travel-time correction.

The Numba backends (and, for `UniformDiskModel` and `RoadRunnerModel`, OpenCL) support the analytic `return_grad=True`
Jacobian. The JAX backends provide forward evaluation; obtain JAX gradients with
`jax.grad`/`jax.jacobian` on the forward model (RoadRunner raises `NotImplementedError` for
`backend="jax", return_grad=True`).

### Backend Pattern

**Numba** (`backends/numba/`): Each model has a forward-only file (e.g. `udmodel.py`) and a `_grad.py` variant (e.g. 
`udmodel_grad.py`) with explicit analytic gradient computation. Functions use `@njit`.

**JAX** (`backends/jax/`): Each model has a single file using `jnp` operations; no separate gradient files —
JAX autodiff handles differentiation. `rrmodel` and `tsmodel` rely on plain JAX tracing (the branchless
`jnp.where` kernels are already differentiable); only `udmodel` wraps an explicit `@jax.custom_jvp` to feed
the Taylor-coefficient derivative chain.

**OpenCL** (`backends/opencl/`, `udmodel` and `rrmodel`): `.cl` device sources next to a Python host
wrapper. `common.cl` holds the shared device code (circle-circle intersection, Horner separation,
time folding), `<model>.cl` the forward and gradient kernels, `<model>.py` the host class,
`_base.py` the `CLModel` base (context, buffers, time shift, read-back), `context.py` the shared
context/queue/program cache, and `_host.py` the jitted MeepMeep and limb-darkening prep.
See "OpenCL Backend" below.

### Limb Darkening (`backends/{numba,jax}/limb_darkening/`)

Available profiles: quadratic, power-2, linear, nonlinear, general, uniform, quadratic_tri (+ logarithmic, exponential, 
square_root in Numba only).

**Numba** provides a 4-function tuple per profile:
- `ld_*` — intensity profile I(mu)
- `ldd_*` — derivatives (dI/dmu, dI/d_coeffs)
- `ldi_*` — disk-integrated intensity
- `ldig_*` — gradient of disk-integrated intensity

**JAX** provides only `ld_*` and `ldi_*` (derivatives are handled by autodiff).

The `evaluate_*` helpers in `backends/numba/limb_darkening/evaluate.py` accept `ldc` as 1D
`(nldc,)`, 2D `(npb, nldc)` — the **leading** axis is the passband — or 3D `(npv, npb, nldc)`.
The 2D case used to reshape on `shape[1]`, which reinterpreted rather than transposed the buffer and
returned silent garbage whenever `npb != nldc`; every test used `npb == nldc == 2`, which hid it.

### Circle-Circle Intersection (`backends/{numba,jax}/ccintersection.py`, `backends/opencl/common.cl`)

Core geometric kernel computing the overlap area between planet and star disks. Numba uses if/else branching; JAX uses 
branchless `jnp.where` to remain traceable and differentiable; OpenCL ports the Numba if/else directly (neighbouring
time samples almost always take the same branch, so divergence is confined to the ingress/egress wavefronts).

### OpenCL Backend (`backends/opencl/`)

`UniformDiskModel` and `RoadRunnerModel`. Both are drop-ins for `backend="numba"`: same `evaluate`
call, same NumPy return, agreement to ~1e-14. `CLModel` (`_base.py`) owns everything the two share —
context, precision, buffer lifecycle, the time shift, read-back — so a third model only needs its
`.cl` kernels, a host prep function, and a thin `__call__`.

**What runs where.** The Taylor coefficients are solved on the *host* with MeepMeep
(`_host.py`, jitted) and uploaded; only the per-datapoint separation, `ccia`, and supersampling
loop run on the device. So Kepler's equation and the `bounding_box` bisection never cross the
device boundary, and the transit window stays bit-identical to Numba. Keep the host prep jitted —
a Python loop over `solve2d` would dominate a large-`npv` evaluation and negate the GPU.

The device never sees `a, i, e, w`: they only feed `solve2d`. It gets `k`, `t0`, `p`, the
coefficients, and the contact-point window.

**Work decomposition.** One work item per `(parameter vector, time)`, looping over its own
supersamples. The global range is `(npt, npv)` — the *time* index is dimension 0, so neighbouring
work items write neighbouring elements of the `(npv, npt)` output.

**Precision.** `precision="double"` (default) matches Numba; `precision="single"` is an opt-in.
On consumer NVIDIA cards fp64 throughput is ~1/64 of fp32, so the fp32 kernel is roughly 8x faster,
at the cost of ~1e-6 flux error from cancellation in the kite-area term near the contact points.
It also returns float32 flux and gradients. `REAL` is set by a `-DREAL=double|float` build option;
every literal in the `.cl` sources must be cast `(REAL)...` so the fp32 build does not silently
promote. Do not add `-cl-fast-relaxed-math`: it breaks agreement with the Numba reference.

**Time origin (the fp32 trap).** `set_data` subtracts a reference epoch `tref` (the centre of the
data span) from `times` in float64 before casting, and `__call__` subtracts the same `tref` from
`t0`. The model only uses `t - t0`, so this is mathematically a no-op — but it is what makes fp32
usable at all. A float32 ULP at BJD 2458325 is **0.25 d**, so folding the absolute time in single
precision quantises a whole transit onto two representable values, silently and with no error
raised (measured: 1e-2 flux error, i.e. the full transit depth, before the shift; 6e-9 after).
Any new OpenCL kernel that consumes absolute times must do the same. `tests/…_class.py`
`TestRealisticTimeStamps` parametrises over BJD/BTJD/zero origins and is what pins this.

**`copy=False` is the point of the backend.** `evaluate(..., copy=False)` skips the readback and
returns `None`, leaving results in `model._model._b_flux` / `_b_dflux`. For `npv=200, npt=1e5` with
gradients the Jacobian is 1.12 GB per evaluation, so the transfer, not the physics, sets the
runtime: ~1.2x over threaded Numba with `copy=True` versus ~105x with `copy=False`. Note the
device-resident gradient is the *compact* `(npv, npt, 7)` form — `_expand_gradient` is host-side, so
no expansion happens on that path. Passing `copy=False` to the Numba or JAX backends raises.

**Porting gotchas** (both already handled, do not "fix" them back):
- `flux` is a *deficit*: it starts at 0 and is only decremented, so out of transit it is 0.0 and at
  mid-transit `-k**2`. It is not normalised to 1. The deleted v2 `ma_uniform.cl` returned
  `1 - lambdae`; do not copy that.
- In `ccia_and_grad`'s full-containment branch (`b <= r1 - r2`, most of a non-grazing transit) the
  Python source leaves `a_kite` unbound and Numba happens to zero it, giving the correct
  `dadb = 0`. The `.cl` port sets it to zero explicitly — an uninitialised read is UB in C.

**RoadRunner specifics.** Limb darkening never reaches the device as a function:
`RoadRunnerModel.evaluate` already tabulates it (`ldp`, `ldg`, `ldi`, `dldi`), and
`_host.rr_precompute*` contracts it against the weight table into `ldm`, the mean intensity under
the planet disk versus the grazing parameter. The kernels only interpolate that, so the device code
is **limb-darkening-agnostic** — every profile, including a user `LDModel`, works with no device
code. Keeping the contraction on the host also keeps the 8 MB weight table off the device and
reproduces the out-of-`klims` `calculate_weights_2d` fallback exactly; measured, it is only 5-6% of
the `copy=False` runtime, so moving it to a device pre-pass kernel is not worth it.

Three RoadRunner conventions differ from the uniform disk and are easy to carry over wrongly:
- Flux is **normalised** (1.0 out of transit); the uniform disk returns a deficit (0.0). Test masks
  are `flux < 0.9999`, not `flux < -1e-6`.
- There is a real per-parameter-vector **validity guard** (`a <= 1`, `e < 0`, NaN in `ldp`), which
  the uniform disk lacks. An invalid vector yields an all-NaN flux and gradient row. It inspects
  **passband 0 only** — a quirk of the reference.
- The gradient row is `7 + npb*nldc` wide, and `nldc` is only known once `ldg` arrives, so the
  program is built lazily with `-DNLDC=<n>` and cached per `nldc`. That keeps the private
  accumulator a static `REAL df[7+NLDC]`; only the point's own passband block is ever non-zero, so
  it accumulates `7+nldc` values and scatters on write-out.

**Device memory.** The Jacobian is `npv * npt * (7 + npb*nldc)` reals — 1.44 GB at `npv=200,
npt=1e5`. `CLModel._check_allocation` compares against the device's `max_mem_alloc_size` (a quarter
of device memory on this card) and raises a `MemoryError` naming the size, because `cl.Buffer`
allocation is lazy and an oversized request otherwise fails as a bare
`MEM_OBJECT_ALLOCATION_FAILURE` at kernel-enqueue time.

**One deliberate divergence from Numba.** `interpolate_mean_limb_darkening` reads `lda[ng]`, one
past the end, when `g` is within ~1e-7 of 1.0 (because `dg = (1-1e-7)/(ng-1)`, so
`floor(1.0/dg) == ng-1`). Numba returns whatever follows the array; on a device that is an
out-of-bounds global read, so the kernel clamps the index to `ng-2`, as the JAX backend does.
Matching undefined behaviour is not possible.

**Guard NaN before any float-to-int conversion in a kernel.** `k` is not covered by the validity
guard, so `g = z/(1+k)` can be NaN. NaN compares false against both `g < 0` and `g > 1`, and
`(int)floor(NaN)` is **INT_MIN** on NVIDIA — which slips under an upper-bound clamp and indexes the
table at a -16 GB offset. That is a memory fault, and it kills the *shared* context for the rest of
the process, so a sampler that hands the model one NaN cannot continue. `interp_ldm*` therefore
tests `isnan` first and clamps the index from below; either alone suffices, both are kept.
Numba returns NaN gracefully here, which is the behaviour to match.

**Testing.** `tests/test_opencl_{ud,rr}model{,_grad,_class}.py`, guarded with `pytest.importorskip` plus
a platform check so a machine with no ICD skips rather than errors. The decisive tests compare
against Numba over heterogeneous data; note the per-epoch orbital parameters must genuinely differ
between epochs, or the `nor > 1` indexing is unobservable and the comparison passes even when every
point wrongly reads epoch 0. Mutation-test new kernels rather than trusting green tests; that is
what caught the weak per-epoch test and a lifecycle test that never allocated the buffers it meant
to invalidate.

### Orbital Mechanics (via `meepmeep`)

Projected star-planet distance is computed using 5th-order Taylor-series polynomial solvers from the
`meepmeep` package. The Numba backend uses MeepMeep's public low-level 2D API (`meepmeep.numba2d`):
- `solve2d` — Taylor coefficient solver for the projected (x, y) position; returns a `(2, 5)` matrix
- `sep_c` — projected separation at an expansion-point-centered time
- `bounding_box` — transit contact-point window (T1, T4)
- `solve2d_d` / `sep_cd` — analytic-derivative variants. Their gradient is the 7-element
  `(tc, p, a, i, e, w, lan)` vector (`dcf` shape `(7, 2, 5)`); PyTransit uses the first six and
  ignores `lan`.

The JAX backend uses MeepMeep's JAX primitives (`meepmeep.backends.jax.ts2d`: `solve_xy_p5`,
`solve_xy_p5_d`, `pd_t15_d`), whose gradients follow a 6-element `(t0, p, a, i, e, w)` convention.

### Gradients & Jacobian Layout

Numba `_grad` kernels return a **compact** per-point gradient: the first 7 columns are
`[k, t0, p, a, i, e, w]` for that point's own passband/epoch, followed by per-passband LD-coefficient
blocks. `models/_utils.py::_expand_gradient` then scatters this to **match the parameters as passed**,
output order `[k, t0, p, a, i, e, w, ldc]`: a scalar param keeps one shared column, while a per-passband
`k`/`ldc` or per-epoch `t0`/orbital param expands to one column per passband/epoch, with each point's
derivative landing only in its own column. So with scalar everything but `t0=[t0a, t0b]`, the columns are
`[k, t0a, t0b, p, a, i, e, w, ldc...]`.

**Time-folding period-derivative gotcha:** kernels fold each time as `t_eval = t - t0 - epoch*p`
(`_folded_time`), so the folded time itself depends on `p` through `-epoch*p`. `sep_cd` only gives
`∂z/∂p` at *fixed* folded time, so the period derivative must add the chain-rule term
`epoch * (transit-centre/t0 term)` (zero at epoch 0, growing with epoch). All three Numba gradient
kernels (`rrmodel_grad`, `tsmodel_grad`, `udmodel_grad`) must include this — it is easy to omit and only
shows up when data spans multiple epochs.

**Testing gradients:** verify analytic gradients against central finite differences of the forward-only
model. Compare only where the model is smooth (e.g. `flux < 0.99`); ingress/egress contact points carry
finite-difference truncation noise (bounding-box cutoff + supersample steps), not gradient error, so
exclude them rather than loosening tolerances globally.

## Key Conventions

- Time is in days (typically BJD)
- Flux is normalised (out-of-transit = 1.0)
- Arrays use NumPy with `float64` precision
- Numba functions use `@njit` with `prange` for parallelisation
- JAX functions use `jnp`; differentiation is via plain tracing (or `@jax.custom_jvp` in `udmodel`)
- `_grad.py` file naming convention for Numba gradient variants; OpenCL keeps both kernels in one `.cl` file
- NumPy-style docstrings
- Python >= 3.10, modern type hints (`float | ndarray`, `tuple[...]`)
