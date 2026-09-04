# RoadRunner backend benchmarks

Wall-clock comparison of `RoadRunnerModel` on the numba, OpenCL and JAX backends, for
value-only evaluation and for two different notions of "with gradients".

```bash
python benchmarks/bench_rrmodel.py                      # full sweep, writes results/results.json
python benchmarks/report.py benchmarks/results/results.json
```

Nothing here is imported by the package (`pyproject.toml` packages only `pytransit*`), and
`benchmarks/results/` is gitignored.

## Scripts

| Script | What it answers |
|---|---|
| `bench_rrmodel.py` | The main sweep: backend x mode x `npt` x `npv`. Writes JSON. |
| `report.py` | Renders that JSON as absolute-time and speedup tables. |
| `bench_prange_scaling.py` | Why threaded numba gets ~3x rather than ~16x on 16 cores. |
| `bench_cl_kernel_retrieval.py` | How much of a small-`npv` OpenCL call is pyopencl dispatch. |
| `check_precision.py` | What fp32 costs in accuracy, at two time origins. Run this before trusting any fp32 timing. |
| `_common.py` | Shared geometry, parameter shapes and the timing policy. |

## What the three modes mean

`RoadRunnerModel(backend='jax', return_grad=True)` raises `NotImplementedError`, so the JAX
gradients come from autodiff over a kernel the benchmark builds itself. Which autodiff mode
you pick changes the verdict by an order of magnitude, so both are measured:

- **`value`** -- flux only.
- **`jacobian`** -- the full `(npt, ncol)` Jacobian. Analytic on numba and OpenCL;
  `jax.jacfwd` over all parameters on JAX, which is the like-for-like output but pays for one
  tangent per parameter.
- **`grad_lnl`** -- the gradient of a scalar white-noise log-likelihood, which is what a
  sampler actually calls. Reverse-mode `jax.value_and_grad` on JAX, the device-side reduction
  (`reduction.cl`) on OpenCL, and the Jacobian contracted with `(obs - f) / e2` on numba,
  since `WNLogLikelihood` rejects the numba backend and points at `evaluate()` for exactly
  this.

`opencl-nocopy` is `evaluate(..., copy=False)`, which leaves results in device memory. It has
no `grad_lnl` row by construction: keeping results on the device *is* what `loglikelihood()`
does, so the `opencl` row already covers it.

## Three things that are easy to get wrong

**The in-transit fraction is held constant.** RoadRunner only does real work inside the
`bounding_box` window; points outside it return early on every backend. The time span
therefore grows with `npt` around a fixed `T14 ~ 0.1333 d` so that ~1/3 of points are always
in transit. A fixed span would make large `npt` spuriously cheap and turn the scaling curve
into an artefact of the sampling.

**Every backend runs in its own subprocess.** `JAX_PLATFORMS` is read at import, so CPU and
GPU cannot be compared in one process, and separate processes also stop XLA's allocator from
contending with the OpenCL context over the same device memory. A group whose backend is
unavailable (no OpenCL ICD, no CUDA plugin) exits non-zero and is skipped, not fatal.

**Timings must be synchronised.** JAX dispatches asynchronously and OpenCL's `copy=False`
path returns before the queue drains, so `timeit` callables block with
`jax.block_until_ready` and `queue.finish()` respectively. Without that you are timing
enqueue, not compute.

## Precision

Everything defaults to **float64** on every backend, so the headline tables compare like with
like. `--precision single` adds an fp32 sweep of the two device backends (numba has no fp32
path) and tags those rows `-fp32`:

```bash
python benchmarks/bench_rrmodel.py --precision double single
python benchmarks/check_precision.py
```

This matters because consumer NVIDIA cards run fp64 at a small fraction of their fp32 rate, so
the default GPU numbers are a *lower bound* on what the hardware can do. But the two backends
are not equally ready for fp32, and `check_precision.py` measures the difference against the
numba float64 reference:

| | zero-centred times | times near BJD 2458325 |
|---|---|---|
| OpenCL `precision='single'` | max &#124;dflux&#124; 7.9e-08 (7.8e-06 of depth), Jacobian <= 9e-06 rel. | **identical** -- the guard holds |
| JAX with x64 disabled | **648 / 5000 points NaN** | **flux error = 1.0x the transit depth** |

The device-side log-likelihood is checked separately, because `evaluate()` being accurate in
fp32 does not make `loglikelihood()` accurate -- the reduction sums `npt` terms. It holds up:
relative error ~1e-07 on lnL and ~6e-05 on the gradient, and, importantly, **flat in `npt`**
from 1e3 to 1e5, so the accumulator is not degrading as the series grows.

**OpenCL fp32 is usable.** Its numbers are the same at both time origins because `set_data`
subtracts a reference epoch in float64 before casting, which is what keeps an absolute BJD
timestamp from being quantised (a float32 ULP at BJD 2458325 is ~0.25 d). That guard is the
whole reason the backend survives realistic timestamps.

**JAX fp32 is not usable**, for two independent reasons:

1. Even with times centred on zero, the flux goes NaN over a band of points beginning exactly
   at the `_compute_half_window` boundary, `0.025 + T14/2`. The window-edge arithmetic loses
   its sign in single precision. This is kernel arithmetic, not a device quirk: CPU and GPU
   produce the identical 648 NaNs at the identical boundary.
2. The JAX backend has no equivalent of the OpenCL reference-epoch shift, so with real BJD
   timestamps the transit is quantised away entirely -- a 100% error, not a degraded one.

The `jax-*-fp32` rows therefore exist to show what the arithmetic *would* cost, not as a
usable configuration. `report.py` prints a reminder whenever any `-fp32` row is present.

### What fp32 actually buys

On the reference run, at `npt=1e5, npv=100`, against the same backend in fp64:

| | fp64 | fp32 | gain |
|---|---|---|---|
| `opencl` value | 16.8 ms | 8.39 ms | 2.0x |
| `opencl-nocopy` Jacobian | 15.6 ms | 8.04 ms | 1.9x |
| `opencl` grad_lnl | 24.8 ms | 9.45 ms | 2.6x |

So fp32 takes the OpenCL log-likelihood from 45x threaded numba to **118x**. Note this is
well short of the ~8x that a pure fp64-vs-fp32 ALU ratio would predict (and that CLAUDE.md
quotes for the OpenCL backend generally): RoadRunner spends much of its time on table
interpolation and memory traffic, neither of which scales with the fp64 penalty. Below about
`npt=1e4` or `npv=10` there is no fp32 gain at all -- those cells are bound by Python-side
dispatch, which is precision-independent.

## Timing policy

One untimed warm-up call absorbs numba JIT compilation, JAX tracing and the OpenCL program
build. A calibration call then sets the repetition count so the timed block runs for at least
0.4 s. Both the minimum and the median are stored; `report.py --stat` selects which to show.
The minimum is best-case throughput with the least scheduler noise, the median is closer to
what a fit feels. On the reference run they agreed to within a few percent everywhere.

## Interpreting the numbers

Backends agree to ~5e-15 on both flux and Jacobian, so the timings compare identical work.
Two caveats worth carrying:

- The default configuration is one passband, one epoch, no supersampling and quadratic limb
  darkening, giving a 9-column Jacobian. Heterogeneous data widens the Jacobian and shifts
  the balance toward the backends that reduce on-device.
- Cells near the 0.1 ms floor are dominated by Python-side dispatch rather than by either
  kernel, which is the regime `bench_cl_kernel_retrieval.py` exists to probe.
