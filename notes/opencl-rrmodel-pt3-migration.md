# Migrating the OpenCL RoadRunner changes to PyTransit3

Source: PyTransit commits `a664826`, `920669f`, `d6454fb` and `2ab63c5` on branch `dev`.
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

Now that both backends share the orbit (Change 6), the orbit cancels and the only thing double
precision has left to remove is rounding. Agreement with the Numba model is ~1e-8 or better in
double and at the float32 floor in single, **independent of geometry**:

| a/R* | single | double |
|---|---|---|
| 4 | 9.2e-07 | 7.5e-09 |
| 10 | 9.1e-07 | 7.1e-09 |
| 20 | 9.5e-07 | 6.3e-09 |

Before the orbits were shared, the double column read 2.0e-06 / 4.8e-08 / 6.7e-09 — a strong
geometry dependence that was the signature of the mismatch. **If that dependence ever comes
back, the orbit has stopped matching**; that is what the `TestPrecision` tolerances are set to
catch.

### fp64 cost (RTX 5070, kernels only, `copy=False` + `finish()`)

| npt x npv | CL single | CL double | fp64 cost |
|---|---|---|---|
| 1e3 x 1 | 188 us | 208 us | 1.1x |
| 1e4 x 1 | 159 us | 242 us | 1.5x |
| 1e4 x 1000 | 1.35 ms | 4.22 ms | 3.1x |
| 1e5 x 1000 | 4.50 ms | 32.1 ms | 7.1x |

**Change 6 cut this from 46x to 7.1x.** With the Keplerian solver the kernel was bound by fp64
transcendentals (`sin`, `cos`, `atan2`, `fmod` per sample), which consumer NVIDIA runs at a
sixty-fourth of the single precision rate. A degree-4 polynomial is multiply-add and the kernel
is table-bound instead, so the fp64 penalty largely disappears. Double is now **13.7x faster
than 16-thread Numba** at 1e5 x 1000 (32 vs 440 ms), against 1.8x before, which makes it a
practical choice rather than a validation-only one.

---

## Change 5 — Kepler's equation by Newton (SUPERSEDED by Change 6, kept for the lesson)

The kernel no longer solves Kepler's equation at all — Change 6 moved the orbit to MeepMeep — so
there is nothing here to port. The finding is kept because it applies to any hand-rolled Kepler
solver in a GPU kernel.

Found while chasing the above. `z_iter` solved Kepler's equation with the fixed point iteration
`E = M + e sin(E)`, which converges **linearly at a rate of e**: reaching double precision needs
~80 steps at e = 0.7, so the loop ran at most 15 and settled for a `1e-4` convergence threshold.
That capped the projected distance at ~1e-4 R_star for *any* eccentric orbit, which in double
precision made the orbit the accuracy bottleneck and negated fp64 entirely.

Replaced with Newton's method from the same starting guess, which doubles the correct digits per
step and converges in four or five, with the threshold tied to the build's precision
(`KEPLER_TOL`, 1e-13 double / 1e-6 single).

| e, w | before | after |
|---|---|---|
| 0.1, 1.0 | 7.7e-06 | 7.5e-09 |
| 0.3, 1.0 | 3.1e-05 | 1.8e-08 |
| 0.5, 1.0 | 8.9e-05 | 6.2e-08 |

(Numba vs OpenCL-double at a/R* = 20, where the expansion error is ~1e-8.) Circular orbits are
**bit-identical** — both solvers converge on the first step at e = 0 — and there is no measurable
performance difference, because the orbit solve is not what the flux kernel is bound by.

The old loop also declared its counter as `int i`, **shadowing the inclination parameter `i`** of
the enclosing function. Harmless there because the loop body never read it, but worth not
reproducing.

---

## Change 6 — Take the orbit from MeepMeep's expansion (the fix for the mismatch)

The kernel now evaluates the *same* Taylor expansion as the Numba model instead of solving the
orbit itself, which removes the difference of Change 4 entirely. MeepMeep ships its evaluators
as OpenCL **device functions**, so this is mostly plumbing:

```python
from meepmeep.backends.opencl import read_kernel_source, build_options
source = read_kernel_source('point2d.cl') + open('rrmodel.cl').read()
cl.Program(ctx, source).build(options=build_options(precision))
```

- `sep_c2(t, c)` is the device twin of `meepmeep.numba2d.sep_c`; `c` is the flattened `(2, 5)`
  `solve2d` matrix. MeepMeep's `build_options` uses the **same `-DREAL=` / `USE_FP64` convention**
  as Change 4, so the two compose without changes.
- Its `common.cl` defines `PI_R`, `TWO_PI_R`, `HALF_PI_R` and the fp64 pragma — **delete your own
  copies or they clash**. Ours became `TWO_PI_R`, and `PI_R` / the pragma were dropped.
- Solvers stay host-side by MeepMeep's contract: `solve2d` and `bounding_box` run on the host and
  the coefficients are uploaded; the device only evaluates the polynomial.

**Two traps, both of which bite silently.**

1. **The expansion is meaningless away from the transit.** It is a degree-4 polynomial, and half
   an orbit away it happily dips below `1 + k` and produces a full-depth *spurious transit*. The
   Numba model guards this with `bounding_box`, and the kernel must too — reject on the box
   before evaluating, never on the value of the separation. The old Keplerian `z_iter` returned
   -1 on the far side and needed no such guard, so this is a new requirement.
2. **The bounding box is widened by the exposure time**, so the exposure time default matters.
   Ours defaulted `exptimes` to `ones` while `TransitModel.set_data` defaults to `zeros`; that
   was harmless while nothing read it (with `nsamples=1` the supersampling offset is exactly zero
   either way) but it stretched the box by a full day, across the far side of a two-day orbit,
   and put a 1.1e-02 spurious transit between every pair of transits. Check this default before
   wiring in the box, and test with data spanning several epochs — single-transit tests cannot
   see it. The same default was independently wrong for supersampling: `set_data(t, nsamples=10)`
   with no exposure time smeared the model over a whole day and returned a transit 5x too
   shallow (depth 0.0021 against 0.0114).
3. **A scalar exposure time is a zero-dimensional array.** `asarray(0.02)` has `ndim == 0`,
   which is fine as a buffer but cannot be indexed inside the compiled expansion loop, and
   fails at *compile* time with a `NumbaTypeError` naming the wrong line. `atleast_1d` it, as
   `TransitModel.set_data` does.

Match the Numba fold exactly: the epoch comes from the **unshifted** sample time and the
supersampling offsets are added to the centred time afterwards, so a sample never crosses into a
neighbouring epoch.

**Performance.** The polynomial is cheaper than the Kepler solve, but the host now precomputes
the coefficients. Keep that loop **compiled** — a Python loop over `solve2d` costs ~6 us per
parameter vector in dispatch alone and dominates everything else at npv = 1000. In an `@njit`
helper it is ~315 ns per vector. Net, kernel-only, against the Keplerian version:

| npt x npv | Kepler | expansion |
|---|---|---|
| 1e3 x 1000 | 456 us | 1107 us |
| 1e4 x 1000 | 950 us | 1431 us |
| 1e5 x 100 | 874 us | 542 us |
| 1e5 x 1000 | 5.65 ms | 4.28 ms |

Faster wherever the kernel dominates, slower where the per-call host precompute does — the same
precompute the Numba model already pays.

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

`before` is the state at `ac72acc`, `after` is `2ab63c5` — Changes 1, 2 and 6 combined:

| npt x npv | before | after | speedup |
|---|---|---|---|
| 1e3 x 1 | 603 / 581 | **163 / 151** | 3.7x / 3.8x |
| 1e4 x 1 | 619 / 587 | **168 / 151** | 3.7x / 3.9x |
| 1e3 x 100 | 664 / 614 | **273 / 222** | 2.4x / 2.8x |
| 1e4 x 100 | 1120 / 683 | **651 / 248** | 1.7x / 2.8x |
| 1e5 x 100 | 5526 / 1345 | **4631 / 542** | 1.2x / 2.5x |
| 1e5 x 1000 | 46841 / 6112 | 44829 / 4283 | 1.0x / 1.4x |

Changes 1 and 2 remove a fixed ~460 us per call; Change 6 gives some of it back as host-side
precompute (~315 ns per parameter vector) and takes more off the kernel. Consequence worth
knowing: OpenCL now **beats** 16-thread Numba for a single light curve at `npt >= 1e4`
(168 vs 178 us; 285 vs 483 us at 1e5), where it previously lost at every `npv == 1`.

**Remaining ~150 us floor:** ~27 us blocking host->device uploads (now 7), ~26 us kernel
launches (3), ~9 us Numba dispatch in `evaluate_ld`/`evaluate_ldi`, ~7 us readback, the
expansion precompute, and ~40 us Python/NumPy in `_evaluate_pv`. Further gains need
preallocated `astype` temporaries and coalesced uploads — perhaps 30-40 us, for materially
uglier code.

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
5. **Data spanning several epochs**, not one transit. The bounding box, the epoch fold and the
   exposure time default are all invisible to a single-transit test, and all three were wrong
   at some point in this work. Assert that nothing dips between the transits and that the two
   backends report the *same number* of in-transit samples.
6. **Eccentric orbits at several `w`**, which is where a Kepler solver's convergence shows.
7. Profile with `cProfile` sorted by `tottime` and confirm neither
   `pyopencl/__init__.py:__getattr__` nor `sqlite3` appears, and that no per-parameter-vector
   Python call into `solve2d` does either.
