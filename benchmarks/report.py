"""Render `bench_rrmodel.py` output as tables: absolute times, and speedup over a baseline.

    python benchmarks/report.py benchmarks/results/results.json
    python benchmarks/report.py results.json --stat median --baseline numba
"""
import argparse
import json
from pathlib import Path

MODES = (('value', 'value only'),
         ('jacobian', 'value + full Jacobian'),
         ('grad_lnl', 'value + gradient of the scalar log-likelihood'))
ORDER = ('numba', 'numba-par',
         'opencl', 'opencl-nocopy', 'jax-cpu', 'jax-gpu',
         'opencl-fp32', 'opencl-nocopy-fp32', 'jax-cpu-fp32', 'jax-gpu-fp32')


def axes(results):
    """Recover the (npt, npv) grid actually present in the file."""
    npts, npvs = set(), set()
    for key in results:
        _, _, npt, npv = key.split('|')
        npts.add(int(npt))
        npvs.add(int(npv))
    return sorted(npts), sorted(npvs)


def value(entry, stat):
    if entry is None or 'error' in entry:
        return None
    return entry[stat] * 1e3


def fmt(v, error=None):
    if v is None:
        if error and ('MEMORY' in error.upper() or 'ResourceExhausted' in error):
            return 'OOM'
        return 'err' if error else '-'
    return f'{v:.3g}' if v < 1000 else f'{v:.0f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('path', nargs='?', default='benchmarks/results/results.json')
    ap.add_argument('--stat', choices=('min', 'median'), default='min',
                    help="'min' is best-case throughput; 'median' is closer to what a fit feels")
    ap.add_argument('--baseline', default='numba-par', help='row the speedups are relative to')
    args = ap.parse_args()

    results = json.loads(Path(args.path).read_text())
    npts, npvs = axes(results)
    present = [b for b in ORDER if any(k.startswith(b + '|') for k in results)]
    get = lambda b, m, t, v: results.get(f'{b}|{m}|{t}|{v}')

    if any(b.endswith('-fp32') for b in present):
        print('note: -fp32 rows are single precision; see check_precision.py for their accuracy.')

    width = 10 * len(npvs)
    for mode, title in MODES:
        if not any(f'|{mode}|' in k for k in results):
            continue
        print(f'\n### {title} - ms per call ({args.stat} of N)\n')
        head = f'{"backend":16s}' + ''.join(f'{f"npt={n:.0e}":>{width + 2}s}' for n in npts)
        print(head)
        print(f'{"":16s}' + ''.join('  ' + ''.join(f'{f"npv={v}":>10s}' for v in npvs) for _ in npts))
        print('-' * len(head))
        for b in present:
            row = f'{b:16s}'
            for npt in npts:
                row += '  ' + ''.join(
                    f'{fmt(value(get(b, mode, npt, v), args.stat), (get(b, mode, npt, v) or {}).get("error")):>10s}'
                    for v in npvs)
            print(row)

        if args.baseline not in present:
            continue
        print(f'\n    speedup over {args.baseline} (x)\n')
        for b in present:
            if b == args.baseline:
                continue
            row = f'{b:16s}'
            for npt in npts:
                for npv in npvs:
                    a = value(get(b, mode, npt, npv), args.stat)
                    r = value(get(args.baseline, mode, npt, npv), args.stat)
                    row += f'{"-":>10s}' if (a is None or r is None) else f'{r / a:>9.1f}x'
                row += '  '
            print(row)


if __name__ == '__main__':
    main()
