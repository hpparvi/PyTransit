#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""RoadRunner against the quadratic Mandel & Agol model: evaluation time against the exposure count.

Timings depend on the machine, and measuring them takes minutes, so the documentation does not
rerun the benchmark at build time. Running this file as a script measures the timings and writes
them to ``rr_benchmark.csv`` next to it, and the documentation plots that file:

    python docs/source/_figures/rr_benchmark.py

Both models run on a single thread, for a light curve where about half of the exposures are in
transit.
"""

from pathlib import Path

from matplotlib.pyplot import subplots, setp
from numpy import arccos, array, linspace, loadtxt, pi

__all__ = ['SETTINGS', 'run_benchmark', 'plot_benchmark']

CSV = Path(__file__).with_name('rr_benchmark.csv')

# The RoadRunner accuracy levels, each the cheapest (nq, ng) whose worst-case error against the
# exact quadratic model stays below the target for radius ratios up to 0.2 and impact parameters
# up to 0.9, with the measured worst-case error in ppm.
SETTINGS = {'RoadRunner (1 ppm)': dict(nq=5, ng=300, error=0.93),
            'RoadRunner (default)': dict(nq=8, ng=100, error=5.9),
            'RoadRunner (10 ppm)': dict(nq=4, ng=75, error=7.2),
            'RoadRunner (100 ppm)': dict(nq=3, ng=20, error=48.0),
            'RoadRunner (500 ppm)': dict(nq=2, ng=10, error=316.0)}

NPTS = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)


def _models():
    from pytransit import QuadraticModel, RoadRunnerModel
    models = {'Quadratic (exact)': QuadraticModel(interpolate=False),
              'Quadratic (interpolated)': QuadraticModel(interpolate=True)}
    for name, s in SETTINGS.items():
        models[name] = RoadRunnerModel('quadratic', nq=s['nq'], ng=s['ng'])
    return models


def run_benchmark(npts=NPTS, path=CSV):
    """Time every model for every number of exposures and write the results to `path`."""
    import numba
    from timeit import repeat
    numba.set_num_threads(1)

    p, a, k, b, ldc = 4.0, 13.0, 0.1, 0.3, [0.4, 0.25]
    models = _models()
    names = list(models)
    rows = []
    for npt in npts:
        time = linspace(-0.1, 0.1, npt)
        row = [npt]
        for name in names:
            tm = models[name]
            tm.set_data(time)
            f = lambda: tm.evaluate(k, ldc, 0.0, p, a, arccos(b / a))
            f()
            number = max(1, int(2e5 // npt))
            row.append(min(repeat(f, number=number, repeat=7)) / number)
        rows.append(row)
        print(npt, ' '.join(f'{t * 1e3:9.4f}' for t in row[1:]))
    with open(path, 'w') as out:
        out.write('npt,' + ','.join(names) + '\n')
        for row in rows:
            out.write(f'{row[0]},' + ','.join(f'{t:.6e}' for t in row[1:]) + '\n')


def plot_benchmark(path=CSV):
    """Plot the evaluation time per exposure against the number of exposures.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with open(path) as f:
        names = f.readline().strip().split(',')[1:]
    data = loadtxt(path, delimiter=',', skiprows=1)
    npt = data[:, 0]

    styles = {'Quadratic (exact)': dict(c='k', lw=2),
              'Quadratic (interpolated)': dict(c='k', ls='--')}
    fig, ax = subplots(figsize=(9, 4.2))
    for i, name in enumerate(names):
        ax.plot(npt, 1e9 * data[:, i + 1] / npt, label=name, **styles.get(name, {}))
    setp(ax, xscale='log', yscale='log', xlabel='Number of exposures',
         ylabel='Time per exposure [ns]', title='Single-threaded evaluation time')
    ax.legend(fontsize='small', frameon=False, ncols=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    run_benchmark()
