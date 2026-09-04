"""How much accuracy does the fp32 RoadRunner give up, and where?

Compares single-precision flux and Jacobian against the numba float64 reference for the
backends that offer an fp32 path: OpenCL (`precision='single'`) and JAX (x64 disabled).

Two time origins are tested, because that is where fp32 transit models go wrong:

    zero   times centred on zero          -- the easy case
    bjd    times near BJD 2458325         -- what real data looks like

A float32 ULP at BJD 2458325 is ~0.25 d, so folding an absolute timestamp in single
precision quantises a whole transit onto a couple of representable values. The OpenCL
backend defends against this by subtracting a reference epoch in float64 inside `set_data`
(see CLAUDE.md, "Time origin (the fp32 trap)"); this script checks whether each backend
actually survives it rather than assuming.

Each (backend, precision) combination runs in its own process because `jax_enable_x64` is
read when arrays are built, then the results are compared against the reference.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import ERR, INC, JITTER, K0, LDC, P, A, make_times, quiet_pyopencl  # noqa: E402

NPT = 5_000
COLS = ('k', 't0', 'p', 'a', 'i', 'e', 'w', 'ld0', 'ld1')
ORIGINS = {'zero': 0.0, 'bjd': 2458325.0}
COMBOS = (('opencl', 'double'), ('opencl', 'single'), ('jax', 'double'), ('jax', 'single'))


def evaluate(backend, precision, origin_name):
    """Flux and Jacobian for one backend/precision/time-origin, as float64 arrays."""
    tref = ORIGINS[origin_name]
    times = make_times(NPT) + tref
    pars = (K0, tref, P, A, INC, 0.0, 0.0)

    if backend == 'numba':
        from pytransit.models.rrmodel import RoadRunnerModel
        m = RoadRunnerModel(backend='numba', return_grad=True)
        m.set_data(times)
        f, d = m.evaluate(*pars, ldc=LDC)
        return np.asarray(f, float), np.asarray(d, float)

    if backend == 'opencl':
        quiet_pyopencl()
        from pytransit.models.rrmodel import RoadRunnerModel
        m = RoadRunnerModel(backend='opencl', return_grad=True, precision=precision)
        m.set_data(times)
        f, d = m.evaluate(*pars, ldc=LDC)
        return np.asarray(f, float), np.asarray(d, float)

    if backend == 'jax':
        import jax
        import jax.numpy as jnp
        if precision == 'double':
            jax.config.update('jax_enable_x64', True)
        from bench_rrmodel import make_jax_kernel
        from pytransit.models._utils import _normalize_parameter_shapes
        from pytransit.models.rrmodel import RoadRunnerModel

        m = RoadRunnerModel(backend='jax')
        m.set_data(times)
        single = make_jax_kernel(m, m.npb, m.nor)
        nrm = _normalize_parameter_shapes(*pars, m.npb, m.ntc, m.nor)
        args = (*[jnp.asarray(x)[0] for x in nrm], jnp.asarray(LDC.reshape(1, -1)))
        f = single(*args)
        jac = jax.jacfwd(single, argnums=tuple(range(8)))(*args)
        d = np.concatenate([np.asarray(x, float).reshape(NPT, -1) for x in jac], axis=1)
        return np.asarray(f, float), d

    raise ValueError(backend)


def compare(ref_f, ref_d, got_f, got_d):
    """Flux error, and per-column Jacobian relative error over the in-transit points."""
    # models return the deficit dF = F - 1; -1e-4 is the analogue of the old `ref_f < 0.9999`
    # cut. A wider mask (e.g. -1e-6) admits the shallow ingress/egress fringe where the JAX
    # fp32 Jacobian has NaN entries, which turns whole columns of this report into `nan`.
    mask = ref_f < -1e-4
    depth = -ref_f.min()                        # we compare two analytic results, not an FD
    out = {'flux_abs': float(np.abs(got_f - ref_f).max()),
           'flux_frac_depth': float(np.abs(got_f - ref_f).max() / depth),
           'cols': {}}
    for j, name in enumerate(COLS):
        a, b = got_d[mask, j], ref_d[mask, j]
        scale = np.abs(b).max()
        if scale < 1e-12:                       # w is identically ~0 at e=0
            out['cols'][name] = None
            continue
        out['cols'][name] = float(np.abs(a - b).max() / scale)
    return out


def check_loglikelihood():
    """Accuracy of the fp32 device-side log-likelihood reduction.

    `evaluate()` being accurate in fp32 does not make `loglikelihood()` accurate: the device
    reduction sums npt terms, and a single-precision accumulator loses ground as npt grows.
    The reference is the numba float64 Jacobian contracted on the host, which is what the
    benchmark's `grad_lnl` mode does for the numba backend.
    """
    quiet_pyopencl()
    from pytransit.models.rrmodel import RoadRunnerModel

    print('\n=== device-side log-likelihood (OpenCL), vs numba float64 ===\n')
    print(f'{"npt":>8s}{"precision":>11s}{"rel err lnL":>14s}{"max rel err grad":>19s}')
    print('-' * 52)
    for npt in (1_000, 10_000, 100_000):
        times = make_times(npt)
        pars = (K0, 0.0, P, A, INC, 0.0, 0.0)
        ref_m = RoadRunnerModel(backend='numba', return_grad=True)
        ref_m.set_data(times)
        flux, dflux = ref_m.evaluate(*pars, ldc=LDC)
        obs = flux + ERR * np.random.default_rng(1).normal(size=npt)
        e2 = ERR ** 2 + JITTER ** 2
        r = (obs - flux) / e2
        ref_lnl = (-0.5 * npt * np.log(2 * np.pi) - 0.5 * npt * np.log(e2)
                   - 0.5 * np.sum(r * (obs - flux)))
        ref_grad = r @ dflux
        ref_dbaseline = np.sum(r)
        ref_djit = JITTER * (np.sum(r ** 2) - npt / e2)
        ref_full = np.concatenate([ref_grad, [ref_dbaseline], [ref_djit]])

        for precision in ('double', 'single'):
            m = RoadRunnerModel(backend='opencl', return_grad=True, precision=precision)
            m.set_data(times)
            lnl, grad = m.loglikelihood(obs, ERR, JITTER, *pars, ldc=LDC, baseline=0.0)
            grad = np.asarray(grad, float)
            scale = np.maximum(np.abs(ref_full), np.abs(ref_full).max() * 1e-6)
            print(f'{npt:8d}{precision:>11s}{abs(lnl - ref_lnl) / abs(ref_lnl):14.2e}'
                  f'{np.abs(grad - ref_full).max() / scale.max():19.2e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--backend')
    ap.add_argument('--precision')
    ap.add_argument('--origin')
    ap.add_argument('--out')
    args = ap.parse_args()

    if args.backend:                            # worker: one combination, dumped to .npz
        f, d = evaluate(args.backend, args.precision, args.origin)
        np.savez(args.out, flux=f, jac=d)
        return 0

    tmp = Path(__file__).parent / 'results' / '_precision'
    tmp.mkdir(parents=True, exist_ok=True)
    report = {}

    for origin in ORIGINS:
        ref_f, ref_d = evaluate('numba', 'double', origin)
        report[origin] = {}
        for backend, precision in COMBOS:
            out = tmp / f'{backend}_{precision}_{origin}.npz'
            cmd = [sys.executable, __file__, '--backend', backend, '--precision', precision,
                   '--origin', origin, '--out', str(out)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                report[origin][f'{backend}/{precision}'] = {
                    'error': proc.stderr.strip().splitlines()[-1][:160] if proc.stderr else 'failed'}
                continue
            z = np.load(out)
            report[origin][f'{backend}/{precision}'] = compare(ref_f, ref_d, z['flux'], z['jac'])

    # ---- render
    depth_note = 'flux error as a fraction of the transit depth'
    for origin, rows in report.items():
        label = 'times centred on zero' if origin == 'zero' else 'times near BJD 2458325'
        print(f'\n=== {origin}  ({label}) ===\n')
        print(f'{"backend/precision":22s}{"max |dflux|":>13s}{"/depth":>11s}   '
              + ''.join(f'{c:>10s}' for c in COLS))
        print('-' * (46 + 10 * len(COLS)))
        for name, r in rows.items():
            if 'error' in r:
                print(f'{name:22s}  FAILED: {r["error"]}')
                continue
            cols = ''.join(f'{"-":>10s}' if r['cols'][c] is None else f'{r["cols"][c]:>10.1e}'
                           for c in COLS)
            print(f'{name:22s}{r["flux_abs"]:13.2e}{r["flux_frac_depth"]:11.1e}   {cols}')
    print(f'\nJacobian columns are max relative error over in-transit points; {depth_note}.')
    print('Reference: numba backend, float64.')

    try:
        check_loglikelihood()
    except Exception as exc:                    # no ICD, or an fp32 reduction that refuses
        print(f'\nlog-likelihood check skipped: {type(exc).__name__}: {exc}')

    path = Path(__file__).parent / 'results' / 'precision.json'
    path.write_text(json.dumps(report, indent=1))
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
