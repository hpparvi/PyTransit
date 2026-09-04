"""Why does the threaded numba RoadRunner get only ~3x on 16 cores?

`rrmodel.py` and `rrmodel_grad.py` parallelise the inner `for ipt in prange(npt)` loop, and
points outside the `bounding_box` window return early. A real light curve has its transit in
one contiguous block, so with prange's static block schedule only the threads owning that
block do work while the rest exit immediately.

This compares three time-array layouts holding the *in-transit point count fixed*, so total
arithmetic is identical and only its distribution across threads changes:

    contiguous  transit in the middle third of a 3*T14 span   (what real data looks like)
    shuffled    the same points, randomly permuted            (balanced, identical work)
    full        span == T14, every point in transit           (balanced and compact)

If `shuffled` scales better than `contiguous` it is load imbalance. If it does not, the
kernel is limited by memory traffic instead -- note that shuffling *costs* cache locality,
so a shuffled win in parallel despite a serial loss is the unambiguous signature.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import INC, K0, LDC, P, T14, A, timeit  # noqa: E402

NPT = 100_000
PARS = (K0, 0.0, P, A, INC, 0.0, 0.0)


def layouts():
    wide = np.linspace(-1.5 * T14, 1.5 * T14, NPT)
    n_in = int(np.sum(np.abs(wide) < 0.5 * T14))
    return {'contiguous': wide,
            'shuffled': np.random.default_rng(0).permutation(wide),
            'full': np.linspace(-0.5 * T14, 0.5 * T14, n_in)}


def main():
    from pytransit.models.rrmodel import RoadRunnerModel

    print(f'npt={NPT}, in-transit fraction 1/3 for contiguous and shuffled\n')
    print(f'{"mode":11s}{"layout":12s}{"serial ms":>11s}{"threaded ms":>13s}{"speedup":>9s}')
    print('-' * 56)
    for grad in (False, True):
        ms = RoadRunnerModel(backend='numba', return_grad=grad, parallel=False)
        mp = RoadRunnerModel(backend='numba', return_grad=grad, parallel=True)
        for name, times in layouts().items():
            ms.set_data(times)
            mp.set_data(times)
            a = timeit(lambda: ms.evaluate(*PARS, ldc=LDC))[0]
            b = timeit(lambda: mp.evaluate(*PARS, ldc=LDC))[0]
            tag = 'jacobian' if grad else 'value'
            print(f'{tag:11s}{name:12s}{a * 1e3:11.3f}{b * 1e3:13.3f}{a / b:8.2f}x')
        print()


if __name__ == '__main__':
    main()
