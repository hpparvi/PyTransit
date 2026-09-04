"""Benchmark RoadRunnerModel across the numba, OpenCL and JAX backends.

Sweeps light-curve length (`npt`) against parameter-vector count (`npv`) for three things
you might ask the model for:

  value      -- flux only
  jacobian   -- flux and the full (npt, ncol) Jacobian; `jax.jacfwd` on the JAX backends,
                which have no `return_grad`
  grad_lnl   -- flux and the gradient of a scalar white-noise log-likelihood, which is what
                a sampler actually calls: reverse-mode on JAX, the device-side reduction on
                OpenCL, and the Jacobian plus a chain-rule contraction on numba

Run with no arguments to benchmark every backend and write a merged `results.json`::

    python benchmarks/bench_rrmodel.py
    python benchmarks/report.py benchmarks/results/results.json

Each backend runs in its own subprocess. That is not incidental: JAX CPU and GPU are
selected by `JAX_PLATFORMS`, which is read at import, and it keeps XLA's allocator from
contending with the OpenCL context over the same device memory.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import ERR, JITTER, LDC, make_params, make_times, quiet_pyopencl, timeit  # noqa: E402

DEFAULT_NPT = (1_000, 10_000, 100_000)
DEFAULT_NPV = (1, 10, 100)
MODES = ('value', 'jacobian', 'grad_lnl')


# --------------------------------------------------------------------------- numba
def bench_numba(results, npts, npvs, parallel):
    from pytransit.models.rrmodel import RoadRunnerModel

    tag = 'numba-par' if parallel else 'numba'
    for npt in npts:
        times = make_times(npt)
        mv = RoadRunnerModel(backend='numba', parallel=parallel)
        mg = RoadRunnerModel(backend='numba', return_grad=True, parallel=parallel)
        for m in (mv, mg):
            m.set_data(times)
        obs = np.asarray(mv.evaluate(*make_params(1), ldc=LDC))
        obs = obs + ERR * np.random.default_rng(1).normal(size=npt)
        e2 = ERR ** 2 + JITTER ** 2

        for npv in npvs:
            pars = make_params(npv)
            record(results, tag, 'value', npt, npv, lambda: mv.evaluate(*pars, ldc=LDC))
            record(results, tag, 'jacobian', npt, npv, lambda: mg.evaluate(*pars, ldc=LDC))

            def grad_lnl():
                # WNLogLikelihood rejects the numba backend and points at evaluate(); this
                # is that contraction, and it is the same reduction reduction.cl does on
                # the device for the OpenCL backend.
                flux, dflux = mg.evaluate(*pars, ldc=LDC)
                r = (obs - flux) / e2
                g = r @ dflux if flux.ndim == 1 else np.einsum('vt,vtc->vc', r, dflux)
                lnl = (-0.5 * npt * np.log(2 * np.pi) - 0.5 * npt * np.log(e2)
                       - 0.5 * np.sum(r * (obs - flux), axis=-1))
                return lnl, g

            record(results, tag, 'grad_lnl', npt, npv, grad_lnl)


# -------------------------------------------------------------------------- opencl
def bench_opencl(results, npts, npvs, precision='double'):
    quiet_pyopencl()
    from pytransit.models.rrmodel import RoadRunnerModel

    sfx = '' if precision == 'double' else '-fp32'
    for npt in npts:
        times = make_times(npt)
        mv = RoadRunnerModel(backend='opencl', precision=precision)
        mg = RoadRunnerModel(backend='opencl', return_grad=True, precision=precision)
        for m in (mv, mg):
            m.set_data(times)
        obs = np.asarray(mv.evaluate(*make_params(1), ldc=LDC))
        obs = obs + ERR * np.random.default_rng(1).normal(size=npt)

        for npv in npvs:
            pars = make_params(npv)
            record(results, 'opencl' + sfx, 'value', npt, npv, lambda: mv.evaluate(*pars, ldc=LDC))
            record(results, 'opencl' + sfx, 'jacobian', npt, npv, lambda: mg.evaluate(*pars, ldc=LDC))

            # copy=False leaves the results on the device, so the queue must be drained
            # explicitly or this times the enqueue rather than the kernel.
            def nocopy(model):
                def run():
                    model.evaluate(*pars, ldc=LDC, copy=False)
                    model._model.queue.finish()
                return run

            record(results, 'opencl-nocopy' + sfx, 'value', npt, npv, nocopy(mv))
            record(results, 'opencl-nocopy' + sfx, 'jacobian', npt, npv, nocopy(mg))
            record(results, 'opencl' + sfx, 'grad_lnl', npt, npv,
                   lambda: mg.loglikelihood(obs, ERR, JITTER, *pars, ldc=LDC))


# ----------------------------------------------------------------------------- jax
def make_jax_kernel(model, npb, nep):
    """A pure-JAX ``f(k, t0, p, a, i, e, w, ldc) -> flux`` for one parameter vector.

    RoadRunnerModel implements no `get_callable()`, so the differentiable kernel is built
    here. The limb-darkening tabulation lives inside the traced function, which keeps `ldc`
    differentiable and matches the other backends: `evaluate()` re-tabulates the profile
    through numba on every call for those too.
    """
    import jax
    import jax.numpy as jnp
    from pytransit.backends.jax.limb_darkening import ld_quadratic, ldi_quadratic
    from pytransit.backends.jax.rrmodel import rrmodel as jax_rrmodel

    times = jnp.asarray(model.times)
    lcids, pbids, epids = (jnp.asarray(x) for x in (model.lcids, model.pbids, model.epids))
    ns, et = jnp.asarray(model.nsamples), jnp.asarray(model.exptimes)
    rules, mu = jnp.asarray(model._rules), jnp.asarray(model.mu)
    pt0, pdt, ng = model._pt0, model._pdt, model.ng
    max_ns = int(model.nsamples.max())

    def single(k, t0, p, a, i, e, w, ldc):
        ldp = jax.vmap(lambda c: ld_quadratic(mu, c))(ldc)
        ldi = jax.vmap(ldi_quadratic)(ldc)
        return jax_rrmodel(times, k, t0, p, a, i, e, w, lcids, pbids, epids, ns, et,
                           ldp, None, ldi, None, rules, pt0, pdt, ng, npb, nep, max_ns)

    return single


def bench_jax(results, npts, npvs, precision='double'):
    import jax
    import jax.numpy as jnp
    if precision == 'double':
        jax.config.update('jax_enable_x64', True)
    from pytransit.models._utils import _normalize_parameter_shapes
    from pytransit.models.rrmodel import RoadRunnerModel

    tag = f"jax-{'gpu' if jax.devices()[0].platform == 'gpu' else 'cpu'}"
    tag += '' if precision == 'double' else '-fp32'
    ldc2 = jnp.asarray(LDC.reshape(1, -1))

    for npt in npts:
        times = make_times(npt)
        m = RoadRunnerModel(backend='jax')
        m.set_data(times)
        single = make_jax_kernel(m, m.npb, m.nor)
        obs = np.asarray(m.evaluate(*make_params(1), ldc=LDC))
        obs = jnp.asarray(obs + ERR * np.random.default_rng(1).normal(size=npt))
        e2 = ERR ** 2 + JITTER ** 2

        def lnl(k, t0, p, a, i, e, w, ldc):
            f = single(k, t0, p, a, i, e, w, ldc)
            return (-0.5 * f.size * jnp.log(2 * jnp.pi) - 0.5 * f.size * jnp.log(e2)
                    - 0.5 * jnp.sum((obs - f) ** 2) / e2)

        # Parameters are per-vector (axis 0); ldc is shared across the population.
        ax = (0,) * 7 + (None,)
        argnums = tuple(range(8))
        fns = {'value': jax.jit(jax.vmap(single, in_axes=ax)),
               'jacobian': jax.jit(jax.vmap(jax.jacfwd(single, argnums=argnums), in_axes=ax)),
               'grad_lnl': jax.jit(jax.vmap(jax.value_and_grad(lnl, argnums=argnums), in_axes=ax))}

        for npv in npvs:
            nrm = [jnp.asarray(x) for x in
                   _normalize_parameter_shapes(*make_params(npv), m.npb, m.ntc, m.nor)]
            args = (*nrm, ldc2)
            for mode in MODES:
                record(results, tag, mode, npt, npv,
                       lambda fn=fns[mode]: jax.block_until_ready(fn(*args)))


# -------------------------------------------------------------------------- driver
def record(results, backend, mode, npt, npv, fn):
    key = f'{backend}|{mode}|{npt}|{npv}'
    try:
        tmin, tmed, reps = timeit(fn)
        results[key] = dict(min=tmin, median=tmed, reps=reps)
        print(f'  {key:44s} {tmin * 1e3:10.3f} ms  (median {tmed * 1e3:.3f}, n={reps})', flush=True)
    except Exception as exc:                      # a cell may OOM; the rest still runs
        msg = f'{type(exc).__name__}: {str(exc).splitlines()[0][:120]}'
        results[key] = dict(error=msg)
        print(f'  {key:44s} FAILED  {msg}', flush=True)


GROUPS = {
    'numba':     lambda r, t, v, p: bench_numba(r, t, v, parallel=False),
    'numba-par': lambda r, t, v, p: bench_numba(r, t, v, parallel=True),
    'opencl':    bench_opencl,
    'jax':       bench_jax,
}

# numba is float64 only, so an fp32 sweep covers the two device backends.
FP32_CAPABLE = ('opencl', 'jax-cpu', 'jax-gpu')

# Group name -> (benchmark group, extra environment). JAX is run twice, once per platform.
PLAN = {'numba':     ('numba', {}),
        'numba-par': ('numba-par', {}),
        'opencl':    ('opencl', {}),
        'jax-cpu':   ('jax', {'JAX_PLATFORMS': 'cpu'}),
        'jax-gpu':   ('jax', {'JAX_PLATFORMS': 'cuda'})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--group', choices=sorted(GROUPS), help='benchmark one group in this process')
    ap.add_argument('--out', help='JSON output path for a single group')
    ap.add_argument('--out-dir', default=str(Path(__file__).parent / 'results'),
                    help='directory for the per-group and merged JSON (default: benchmarks/results)')
    ap.add_argument('--npt', type=int, nargs='+', default=list(DEFAULT_NPT))
    ap.add_argument('--npv', type=int, nargs='+', default=list(DEFAULT_NPV))
    ap.add_argument('--only', nargs='+', choices=sorted(PLAN),
                    help='run only these groups (default: all)')
    ap.add_argument('--precision', nargs='+', choices=('double', 'single'), default=['double'],
                    help="float width. 'single' covers only the device backends (numba is "
                         "float64 only) and is tagged '-fp32' in the results. Check "
                         'check_precision.py before trusting an fp32 timing.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.group:
        results = {}
        GROUPS[args.group](results, args.npt, args.npv, args.precision[0])
        Path(args.out).write_text(json.dumps(results, indent=1))
        return 0

    merged = {}
    for precision in args.precision:
        for name in (args.only or PLAN):
            if precision == 'single' and name not in FP32_CAPABLE:
                continue                      # numba has no fp32 path
            group, extra = PLAN[name]
            label = name if precision == 'double' else f'{name} fp32'
            out = out_dir / f'res_{name}{"" if precision == "double" else "_fp32"}.json'
            print(f'=== {label} ===', flush=True)
            env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', **extra)
            cmd = [sys.executable, __file__, '--group', group, '--out', str(out),
                   '--precision', precision,
                   '--npt', *map(str, args.npt), '--npv', *map(str, args.npv)]
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                # A missing OpenCL ICD or CUDA plugin takes out one group, not the run.
                print(f'  group {label!r} exited with {proc.returncode} -- skipped', flush=True)
            if out.exists():
                merged.update(json.loads(out.read_text()))

    path = out_dir / 'results.json'
    path.write_text(json.dumps(merged, indent=1))
    print(f'\nwrote {path}: {len(merged)} cells')
    print(f'render with: python {Path(__file__).parent / "report.py"} {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
