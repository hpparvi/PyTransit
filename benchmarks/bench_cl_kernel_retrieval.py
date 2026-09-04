"""How much of a small-npv OpenCL call is pyopencl kernel retrieval?

`backends/opencl/rrmodel.py` launches with `program.rrmodel(...)` / `program.rrmodel_grad(...)`.
Attribute access on a `cl.Program` constructs a *new* `cl.Kernel` on every call -- pyopencl
says so itself, raising `RepeatedKernelRetrieval` each time and warning that it costs
"possibly considerable expense". This isolates that cost from the rest of an evaluate().

It matters on the npv=1 axis, where the OpenCL backend is already overhead-bound: a whole
call there is a fraction of a millisecond, most of it Python-side dispatch rather than GPU
work. Caching `cl.Kernel(prg, name)` once per program is the fix.
"""
import sys
from pathlib import Path

import numpy as np
import pyopencl as cl

sys.path.insert(0, str(Path(__file__).parent))
from _common import INC, K0, LDC, P, A, quiet_pyopencl, timeit  # noqa: E402

NPT = 10_000
PARS = (K0, 0.0, P, A, INC, 0.0, 0.0)


def main():
    quiet_pyopencl()
    from pytransit.models.rrmodel import RoadRunnerModel

    times = np.linspace(-0.2, 0.2, NPT)
    for grad in (False, True):
        model = RoadRunnerModel(backend='opencl', return_grad=grad)
        model.set_data(times)
        model.evaluate(*PARS, ldc=LDC)              # force the program build and allocation
        backend = model._model
        program = backend._program(LDC.size if grad else 0)
        name = 'rrmodel_grad' if grad else 'rrmodel'

        retrieval = timeit(lambda: getattr(program, name))[0]
        kernel = cl.Kernel(program, name)
        cached = timeit(lambda: kernel)[0]

        def nocopy():
            model.evaluate(*PARS, ldc=LDC, copy=False)
            backend.queue.finish()

        call = timeit(nocopy)[0]

        rows = ((f'getattr(program, {name!r})', retrieval),
                ('cached cl.Kernel reference', cached),
                ('whole evaluate(copy=False)', call))
        print(f'{"gradient" if grad else "forward"} kernel, npt={NPT}, npv=1')
        for label, t in rows:
            print(f'   {label:<34s}{t * 1e3:9.4f} ms')
        print(f'   {"retrieval share of the call":<34s}{100 * retrieval / call:9.1f} %\n')


if __name__ == '__main__':
    main()
