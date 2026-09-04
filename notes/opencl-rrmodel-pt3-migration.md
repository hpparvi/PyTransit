# Migrating the OpenCL RoadRunner changes to PyTransit3

Source: PyTransit commits `a664826` and the precision commit that follows it, on branch `dev`.
Target: `pt3` (PyTransit3), `pytransit/backends/opencl/`.

Three independent changes. Changes 1 and 2 port directly and are pure wins.
Change 3 **must not** be ported verbatim — pt3 already resolves the same ambiguity
by a different rule. Read the divergence note before touching it.

---

## Change 1 — Bind kernels once (port directly, highest value)

**Problem.** `program.some_kernel(queue, ...)` goes through
`pyopencl.Program.__getattr__`, which constructs a **new `Kernel` object on every
access** and regenerates its invoker. Invoker generation consults pyopencl's
`pytools.persistent_dict` cache, so each access costs an **SQLite query + unpickle +
marshal load**. With three kernels per evaluation this was **53% of the wall time** of
a small model evaluation (measured by `cProfile`: 0.135 s of 0.254 s over 300 calls).

**Fix.** Look the kernels up once, after the program is built, and keep them.

```python
# after build_program(...)
self._k_ldm = program.calculate_ldm
self._k_coefficients = program.calculate_coefficients
self._k_flux = program.rr_flux
```

**pt3 target sites** (both use the per-call pattern today):
- `pytransit/backends/opencl/rrmodel.py:132` — `program.rrmodel_grad(self.queue, shape, None, ...)`
- `pytransit/backends/opencl/rrmodel.py:140` — `program.rrmodel(self.queue, shape, None, ...)`
- `pytransit/backends/opencl/udmodel.py:99` — `self.prg.udmodel_grad(...)`
- `pytransit/backends/opencl/udmodel.py:104` — `self.prg.udmodel(...)`

**pt3-specific gotcha.** `rrmodel.py:83` caches *programs* per `nldc`
(`self._programs[nldc] = build_program(...)`). Kernels belong to a program, so the
kernel cache must be keyed the same way — cache per `nldc`, e.g. alongside the program
in `self._programs`, or as `self._kernels[nldc] = (program.rrmodel, program.rrmodel_grad)`.
A single `self._k_flux` attribute would return the wrong program's kernel after an
`nldc` change.

**Effect:** 570 us -> 250 us per call.

---

## Change 2 — Set kernel arguments only when they change (port directly)

**Problem.** pyopencl re-marshals every argument on each launch. `rr_flux` takes 17
arguments; three launches cost 114 us of Python.

**Fix.** `Kernel.set_args(...)` once, then launch with the global size only:

```python
if not self._kernel_args_set:
    self._set_kernel_args()
cl.enqueue_nd_range_kernel(self.queue, self._k_flux, (npv, self.nptb), None)
```

Three launches: 114 us -> **26 us**.

**The invariant this depends on.** `set_args` records device buffer handles. Any
reallocated buffer leaves the kernel pointing at freed memory, which yields
**silently wrong flux, not an error**. So:

> Every code path that creates or releases a device buffer must clear
> `_kernel_args_set`.

In PyTransit that is four sites: `init_integration`, `set_data`, `_allocate`, and the
`_b_p` reallocation branch in `_evaluate_pv`. In pt3 the buffer sites are
`_base.py:110` (`.release()`), `_base.py:120`, `_base.py:144`, and `_base.py:312`
(`cl.Buffer(...)`) — the reduction buffer at 312 belongs to the separate `_reduction`
program, so it only affects that program's kernels.

**Redundancy note (do not rely on it).** In PyTransit, `set_data` and
`init_integration` both end with `self.npv = None`, which forces `_allocate` on the
next call, and `_allocate` clears the flag itself. Their own invalidation is therefore
redundant *today*. Mutation testing confirmed only the `_b_p` invalidation is
load-bearing for flux-level tests. Keep the explicit invalidations anyway — the
coupling is invisible and one refactor away from breaking.

**Test it white-box.** Flux comparisons cannot distinguish a redundant invalidation
from a missing one. Assert the flag directly, or a removed invalidation will pass
unnoticed:

```python
tc.set_data(time);        assert tc._kernel_args_set is False
tc.evaluate(*args);       assert tc._kernel_args_set is True
tc.init_integration(10, 120); assert tc._kernel_args_set is False
tc._allocate(4);          assert tc._kernel_args_set is False
```
See `TestKernelArgumentCaching` in `tests/test_rrmodel_cl.py`. All four
invalidation-removal mutations fail this test; only one fails without it.

**Effect:** 250 us -> 120 us per call.

---

## Change 3 — Radius ratio shape normalisation (DIVERGES from pt3)

**PyTransit's bug.** A 1D `k` of length `npv` was read as `(1, npv)` by both models.
The Numba model silently returned **one** light curve from `k[0]`; the OpenCL model
raised `IndexError` at `nk = k.shape[1]`.

**PyTransit's fix.** `radius_ratio_array(k, npv)` in
`pytransit/models/roadrunner/common.py`, shared by both models. `npv` comes from the
**orbital parameters** (`p`/`t0`); `k` is then interpreted against it:

| input | npv == 1 | npv > 1 |
|---|---|---|
| scalar | `(1, 1)` | `(npv, 1)` broadcast |
| 1D, size n | `(1, n)` per passband | `(n, 1)` per parameter vector; `n` must equal `npv` or 1 |
| 2D | `(1, nk)` | `(npv, nk)`; leading 1 broadcasts |
| otherwise | `ValueError` | `ValueError` |

**Why pt3 cannot take this verbatim.** pt3 already has `_npv_from_k(k, npb)` in
`pytransit/models/_utils.py:61`, which **derives `npv` from `k`** and disambiguates a
1D array using **`npb`**, not the orbital parameters:

```python
elif k.ndim == 1:
    return 1 if k.size == npb else k.size    # pt3
```

The two rules therefore differ exactly when `k` is 1D:
- pt3: `k.size == npb` -> single parameter vector, per-passband.
- PyTransit: decided by `npv` from the orbital parameters.

They disagree whenever `npv == npb` (pt3 assumes one parameter vector, PyTransit a
population), and pt3's rule is already self-consistent with its `@overload`-ed Numba
path. **Do not replace `_npv_from_k`.** Port only the two things pt3 is missing:

1. **Reject mismatched shapes loudly.** Ensure a 1D `k` whose size matches neither
   `npb` nor the population size raises, rather than silently truncating.
2. **Return a writeable array.** See the Numba gotcha below.

**Numba gotcha (applies to both repos).** Return a `.copy()` of any broadcast, never
the read-only view. Numba types a non-writeable array as
`readonly array(float64, 2d, C)` — a **distinct type** from the writable one — so a
broadcast view triggers a **full recompilation** of the kernel on first use. The copy
is `npv * nk` floats; the recompile is seconds.

---

## Change 4 — Selectable precision (pt3 already has this; nothing to port)

PyTransit's kernel was single-precision only and has been converted to pt3's own pattern, so
this change flows *from* pt3, not to it. Recorded here so the two stay recognisable to each
other:

- `rrmodel.cl` now opens with the `#ifdef USE_FP64 / #pragma OPENCL EXTENSION cl_khr_fp64` guard
  and `PI_R`/`TWO_PI`/`HALF_PI` macros, all 112 `float` are `REAL`, and all 70 literals are cast
  `(REAL)x`. `M_PI_F`/`M_PI_2_F` are gone.
- The host takes `precision='single'|'double'`, resolves it to `self.dtype` via a device
  `double_fp_config` check, and builds with `-DREAL=double -DUSE_FP64` or `-DREAL=float`.
- **PyTransit defaults to `'single'`, pt3 defaults to `'double'`.** Deliberate: PyTransit's
  existing behaviour and benchmarks are single, and fp64 is 46x slower here (below). If the two
  are ever unified, this is the one user-visible difference to decide on.
- The precision is constructor-only in both. With the kernel-argument caching of Change 2 a
  runtime switch would have to rebuild the program, rebind the kernels, *and* clear
  `_kernel_args_set` — not worth supporting.

**Guard the source-level invariant.** A reintroduced `float` or `0.5f` breaks the double build
by mixing types, and a bare `0.5` silently promotes the single build to double arithmetic. None
of that surfaces in a flux test. `TestKernelSourceIsPrecisionAgnostic` in
`tests/test_rrmodel_cl.py` greps the comment-stripped kernel for `float`, `f`-suffixed literals,
`M_[A-Z]+_F` and `native_`/`half_` intrinsics; all four mutations fail it.

### What double precision actually buys

The difference from the Numba model is the sum of rounding and the quadrature port, and which
dominates depends on the geometry — so a single tolerance number is misleading:

| a/R* | single | double | gain |
|---|---|---|---|
| 4 | 2.2e-06 | 2.0e-06 | 1.1x |
| 6 | 6.8e-07 | 3.8e-07 | 1.8x |
| 10 | 6.2e-07 | 4.8e-08 | 12.8x |
| 20 | 1.2e-06 | 6.7e-09 | **176x** |

For a short, steep transit the ~2e-6 quadrature difference swamps the rounding and double buys
nothing. Only once the transit is long enough for the quadrature difference to fall away does
the precision become the limit. **Set test tolerances from the quadrature difference, not from
the precision**, and use a long transit (a = 20) if you want a test that actually exercises fp64.

### fp64 cost (RTX 5070, kernels only, `copy=False` + `finish()`)

| npt x npv | CL single | CL double | fp64 cost |
|---|---|---|---|
| 1e3 x 1 | 124 us | 229 us | 1.9x |
| 1e4 x 1 | 146 us | 933 us | 6.4x |
| 1e4 x 1000 | 932 us | 27.0 ms | 29x |
| 1e5 x 1000 | 5.6 ms | 257 ms | 46x |

The cost rises from 1.9x to 46x as the fixed per-call overhead stops dominating, converging on
the ~64x fp64:fp32 ratio of consumer NVIDIA. Double is still **1.8x faster than 16-thread Numba**
at 1e5 x 1000 (257 vs 473 ms), so it is usable for validation runs; it is not a production
default.

---

## Deliberately rejected: non-blocking parameter uploads

`cl.enqueue_copy(..., is_blocking=False)` on the host->device uploads would save ~20 us
of the remaining 120 us. **Do not do it.** The host arrays are overwritten by the next
`evaluate()` call with nothing synchronising them against the in-flight transfer — a
real data race whenever `copy=False` is used in a loop, which is the intended use.
pt3's `_base.py:47-50` already documents the related in-order-queue requirement for the
same reason; this would break that guarantee on the host side instead.

---

## Measurements (RTX 5070, quadratic LD, ng=100, nq=8)

`npt` x `npv`, microseconds, `copy=True` / `copy=False`+`queue.finish()`:

| npt x npv | before | after | speedup |
|---|---|---|---|
| 1e3 x 1 | 603 / 581 | **133 / 121** | 4.5x / 4.8x |
| 1e4 x 1 | 619 / 587 | **149 / 136** | 4.1x / 4.3x |
| 1e3 x 100 | 664 / 614 | **200 / 153** | 3.3x / 4.0x |
| 1e4 x 100 | 1120 / 683 | **626 / 221** | 1.8x / 3.1x |
| 1e5 x 1000 | 46841 / 6112 | 49554 / 5653 | ~1x (transfer-bound) |

A fixed ~460 us is removed, so the gain is large for small models and nil for
transfer-bound ones. Consequence worth knowing: OpenCL now **beats** 16-thread Numba
for a single light curve at `npt >= 1e4` (149 vs 179 us; 361 vs 462 us at 1e5), where
it previously lost at every `npv == 1`.

**Remaining 120 us floor:** ~27 us blocking host->device uploads (5), ~26 us kernel
launches (3), ~9 us Numba dispatch in `evaluate_ld`/`evaluate_ldi`, ~7 us readback,
~40 us Python/NumPy in `_evaluate_pv`. Further gains need preallocated `astype`
temporaries and coalesced uploads — perhaps 30-40 us, for materially uglier code.

---

## Verification recipe

1. **Correctness against the Numba backend** across `npv` in `{1, 5, 12}`, `npt`
   changes, and `init_integration` changes, interleaved on **one** model instance.
   Interleaving is what catches stale arguments; a fresh model per case does not.
2. **Bit-identical repeats:** 20 identical `evaluate` calls must return `array_equal`
   results.
3. **White-box invalidation assertions** (Change 2 above).
4. **Mutation-test the invalidations:** remove each one in turn and confirm a test
   fails. If none fails, the test is not a guard.
5. Profile with `cProfile` sorted by `tottime` and confirm neither
   `pyopencl/__init__.py:__getattr__` nor `sqlite3` appears.
