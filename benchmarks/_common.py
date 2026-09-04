"""Shared setup for the RoadRunner benchmarks: geometry, timing policy, warning filters.

The transit geometry is fixed across every benchmark so numbers from different scripts
are comparable. The one non-obvious choice is the time span (see `make_times`).
"""
import warnings
from statistics import median
from time import perf_counter

import numpy as np

# Fixed transit geometry. With these parameters T14 ~ 0.1333 d.
P, A, INC, K0 = 3.0, 6.0, 1.45, 0.10
LDC = np.array([0.30, 0.10])
T14 = 0.1333
SPAN = 3 * T14          # so ~1/3 of the points fall inside the transit at every npt
ERR = 3e-4
JITTER = 1e-4


def quiet_pyopencl():
    """Silence pyopencl's per-call chatter without hiding unrelated warnings.

    The RoadRunner OpenCL launch site retrieves its kernel by attribute access, so
    pyopencl raises RepeatedKernelRetrieval on every evaluation; the program build also
    emits a CompilerWarning. Both would swamp the benchmark output.
    """
    try:
        import pyopencl as cl
    except ImportError:
        return
    for name in ('CompilerWarning', 'RepeatedKernelRetrieval'):
        category = getattr(cl, name, None)
        if isinstance(category, type) and issubclass(category, Warning):
            warnings.filterwarnings('ignore', category=category)


def make_times(npt: int) -> np.ndarray:
    """Time stamps for `npt` points spanning 3*T14 centred on the transit.

    The span grows with `npt` rather than staying fixed, so the in-transit fraction stays
    at ~1/3. RoadRunner only does real work inside the `bounding_box` window -- points
    outside it return early on every backend -- so a fixed span would make large `npt`
    spuriously cheap and turn the scaling curve into an artefact of the sampling.
    """
    return np.linspace(-0.5 * SPAN, 0.5 * SPAN, npt)


def make_params(npv: int):
    """Parameters for `npv` vectors: scalars at npv=1, 1D arrays of length npv otherwise.

    With a single passband a 1D `k` of length npv != 1 is unambiguously per-parameter-vector
    (see `models._utils._npv_from_k`), so the orbital parameters follow the same shape.
    """
    if npv == 1:
        return K0, 0.0, P, A, INC, 0.0, 0.0
    rng = np.random.default_rng(42)
    return (K0 * (1 + 0.05 * rng.normal(size=npv)),
            np.zeros(npv), np.full(npv, P), np.full(npv, A),
            INC + 0.005 * rng.normal(size=npv), np.zeros(npv), np.zeros(npv))


def timeit(fn, min_time: float = 0.40, min_reps: int = 3, max_reps: int = 50,
           max_total: float = 10.0) -> tuple[float, float, int]:
    """Time `fn`, returning ``(min, median, nreps)`` in seconds.

    One untimed warm-up call absorbs numba JIT compilation, JAX tracing and the OpenCL
    program build; a calibration call then sets the repetition count so the timed block
    runs for at least `min_time`. Both statistics are returned because they answer
    different questions: the minimum is best-case throughput with the least scheduler
    noise, the median is closer to what a fit actually feels.

    `fn` must be synchronous -- block JAX with `jax.block_until_ready` and the OpenCL
    `copy=False` path with `queue.finish()`, or this measures dispatch, not compute.
    """
    fn()
    t0 = perf_counter()
    fn()
    dt = perf_counter() - t0
    reps = int(min(max(min_time / max(dt, 1e-9), min_reps), max_reps))
    ts = []
    for _ in range(reps):
        t0 = perf_counter()
        fn()
        ts.append(perf_counter() - t0)
        if sum(ts) > max_total:
            break
    return min(ts), median(ts), len(ts)
