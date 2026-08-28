#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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

"""Tests for the BaseLPF light curve plotting."""

import matplotlib
import pytest

matplotlib.use('Agg')  # A headless backend, set before pytransit imports pyplot.

from matplotlib.pyplot import close
from numpy import linspace
from numpy.random import default_rng

from pytransit import BaseLPF, RoadRunnerModel
from pytransit.lpf.mptlpf import MPTLPF

NPT = 80


def make_lpf(nlc: int = 1):
    rng = default_rng(0)
    times = [linspace(0.9 + i, 1.1 + i, NPT) for i in range(nlc)]
    fluxes = [1.0 + rng.normal(0.0, 1e-3, NPT) for _ in range(nlc)]
    lpf = BaseLPF('test', ['g'], times=times, fluxes=fluxes, tm=RoadRunnerModel('quadratic'))
    lpf.set_prior('tc', 'NP', 1.0, 0.01)
    lpf.set_prior('p', 'NP', 2.5, 1e-4)
    return lpf


class TestPlanetParameterLookup:
    """BaseLPF names its single planet 'tc' and 'p', the multiplanet LPFs number theirs from one."""

    def test_a_single_planet_lpf_uses_the_unsuffixed_names(self):
        lpf = make_lpf()
        tid, pid = lpf._planet_pids(1)
        assert (lpf.ps.names[tid], lpf.ps.names[pid]) == ('tc', 'p')

    def test_a_multiplanet_lpf_uses_the_numbered_names(self):
        lpf = MPTLPF('mp', 2, ['g'], times=[linspace(0.9, 1.1, NPT)],
                     fluxes=[1.0 + default_rng(0).normal(0.0, 1e-3, NPT)],
                     tm=RoadRunnerModel('quadratic'))
        for planet in (1, 2):
            tid, pid = lpf._planet_pids(planet)
            assert (lpf.ps.names[tid], lpf.ps.names[pid]) == (f'tc_{planet}', f'p_{planet}')

    def test_a_missing_planet_raises(self):
        with pytest.raises(KeyError, match='planet 2'):
            make_lpf()._planet_pids(2)


class TestPlotLightCurves:
    def test_a_single_light_curve_lpf_plots(self):
        """The default width comes from the time span, which used to call the removed ndarray.ptp."""
        lpf = make_lpf(nlc=1)
        lpf.optimize_global(niter=5, npop=20, plot_convergence=False, use_tqdm=False)
        fig = lpf.plot_light_curves(method='de')
        assert len(fig.axes) == 1
        close(fig)

    def test_several_light_curves_plot(self):
        lpf = make_lpf(nlc=3)
        lpf.optimize_global(niter=5, npop=20, plot_convergence=False, use_tqdm=False)
        fig = lpf.plot_light_curves(method='de', ncol=2)
        assert len(fig.axes) == 3
        close(fig)

    def test_an_unknown_method_raises(self):
        with pytest.raises(ValueError, match='The "method" needs to be one of'):
            make_lpf().plot_light_curves(method='nonsense')
