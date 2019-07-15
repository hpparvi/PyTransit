#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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

import warnings

import matplotlib.pyplot as pl
from exodata.astroquantities import Quantity as Qty
from numpy import linspace, zeros, atleast_2d, diag, tile, newaxis, arccos, sqrt, array, arange
from numpy.random import multivariate_normal, seed

from pytransit import QuadraticModel
from pytransit.contamination import SMContamination, Instrument
from pytransit.contamination.filter import *
from pytransit.orbits.orbits_py import duration_eccentric
from pytransit.param import NormalPrior as NP

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from george import GP
    from george.kernels import ExpKernel, Matern32Kernel
    with_george = True
except ImportError:
    with_george = False

from .core import LCData, LCDataSet

class SimulationSetup:
    orbits = {'long_transit':  [40.00, 75.06, 1.5], # period, semi-major axis, baseline observing time
              'short_transit': [ 5.00, 18.80, 1.0]} # Roughly an M-dwarf with rho = 5.0

    stars = {'F': (6900, [[0.60, 0.13], [0.43, 0.14], [0.35, 0.14], [0.27, 0.15]]),  # Teff, ldcs
             'G': (5800, []),
             'M': (3600, [[0.50, 0.27], [0.48, 0.26], [0.32, 0.27], [0.24, 0.26]])}  # M1

    def __init__(self, stype, k, b, c, orbit='long_transit', cteff=None, nights=1,
                 know_host=True, misidentify_host=False, know_orbit=False):
        assert 0.0 < k < 1
        assert 0.0 <= b <= 1 + k
        assert stype in self.stars.keys()
        assert orbit in self.orbits.keys()
        assert 0 < nights < 10

        self.orbit = orbit
        self.stype = stype
        self.rho = 5.0
        self.hteff, self.ldcs = self.stars[stype]
        self.cteff = cteff or self.hteff
        self.k_apparent, self.b, self.c = k, b, c
        self.p, self.a, self.t_baseline = self.orbits[orbit]
        self.i = arccos(self.b / self.a)
        self.nights = nights
        self.know_host = know_host
        self.know_orbit = know_orbit
        self.misidentify_host = misidentify_host

        if know_orbit:
            self.priors = {'b': NP(b, 0.01), 'rho':NP(5.0, 0.05)}
        else:
            self.priors = {}

    @property
    def orbital_parameters(self):
        return self.k_apparent, self.p, self.a, self.b, self.i

    @property
    def name(self):
        tmp = "{}{}_{:s}_k{:4.2f}_b{:4.2f}_c{:4.2f}_t{:4.0f}_n{:d}_{}"
        if self.know_host:
            if self.misidentify_host:
                scase = 'x'
            else:
                scase = 'c'
        else:
            scase = 'u'
        ocase = 'constrained_orbit' if self.know_orbit else 'free_orbit'
        return tmp.format(self.stype, scase, self.orbit.split('_')[0], self.k_apparent, self.b, self.c, self.cteff, self.nights, ocase).replace('.', '')


class MockLC:
    pb_names = "g' r' i' z'".split()
    pb_centers = 1e-9 * array([470, 640, 780, 900])
    npb = len(pb_names)

    def __init__(self, setup: SimulationSetup, **kwargs):
        self.setup = self.s = s = setup
        self.t_exposure_d = Qty(kwargs.get('exptime', 60), 's').rescale('d')
        self.t_baseline_d = Qty(s.t_baseline, 'h').rescale('d')
        self.ldcs = s.ldcs
        self.tm = QuadraticModel(klims=(0.01, 0.99), nk=512)

        self.filters = "g' r' i' z'".split()
        self.npb = len(self.filters)
        self.k_apparent, self.p, self.a, self.b, self.i = s.orbital_parameters

        self.duration_d = Qty(duration_eccentric(self.p, self.k_apparent, self.a, self.i, 0, 0, 1), 'd')

        # Contamination
        # -------------
        qe_be = TabulatedFilter('1024B_eXcelon',
                                [300, 325, 350, 400, 450, 500, 700, 800, 850, 900, 950, 1050, 1150],
                                [0.0, 0.1, 0.25, 0.60, 0.85, 0.92, 0.96, 0.85, 0.70, 0.50, 0.30, 0.05, 0.0])
        qe_b = TabulatedFilter('2014B',
                               [300, 350, 500, 550, 700, 800, 1000, 1050],
                               [0.10, 0.20, 0.90, 0.96, 0.90, 0.75, 0.11, 0.05])
        qes = qe_be, qe_b, qe_be, qe_be

        self.instrument = instrument = Instrument('MuSCAT2', (sdss_g, sdss_r, sdss_i, sdss_z), qes)
        self.contaminator = SMContamination(instrument, "i'")

        self.hteff = setup.hteff
        self.cteff = setup.cteff
        self.i_contamination = setup.c
        self.k_true = setup.k_apparent / sqrt(1 - self.i_contamination)
        self.contamination = self.contaminator.contamination(self.i_contamination, self.hteff, self.cteff)

    @property
    def t_total_d(self):
        return self.duration_d + 2 * self.t_baseline_d

    @property
    def duration_h(self):
        return self.duration_d.rescale('h')

    @property
    def n_exp(self):
        return int(self.t_total_d // self.t_exposure_d)

    def __call__(self, rseed=0, ldcs=None, wnsigma=None, rnsigma=None, rntscale=0.5):
        return self.create(rseed, ldcs, wnsigma, rnsigma, rntscale)

    def create(self, rseed=0, ldcs=None, wnsigma=None, rnsigma=None, rntscale=0.5, nights=1):
        ldcs = ldcs if ldcs is not None else self.ldcs
        seed(rseed)

        self.time = linspace(-0.5 * float(self.t_total_d), 0.5 * float(self.t_total_d), self.n_exp)
        self.time = (tile(self.time, [nights, 1]) + (self.p * arange(nights))[:, newaxis]).ravel()
        self.npt = self.time.size
        self.tm.set_data(self.time)

        self.transit = zeros([self.npt, 4])
        for i, (ldc, c) in enumerate(zip(ldcs, self.contamination)):
            self.transit[:, i] = self.tm.evaluate_ps(self.k_true, ldc, 0, self.p, self.a, self.i)
            self.transit[:, i] = c + (1-c)*self.transit[:, i]

        # White noise
        # -----------
        if wnsigma is not None:
            self.wnoise = multivariate_normal(zeros(atleast_2d(self.transit).shape[1]), diag(wnsigma)**2, self.npt)
        else:
            self.wnoise = zeros_like(self.transit)

        # Red noise
        # ---------
        if rnsigma and with_george:
            self.gp = GP(rnsigma**2 * ExpKernel(rntscale))
            self.gp.compute(self.time)
            self.rnoise = self.gp.sample(self.time, self.npb).T
            self.rnoise -= self.rnoise.mean(0)
        else:
            self.rnoise = zeros_like(self.transit)

        # Final light curve
        # -----------------
        self.time_h = Qty(self.time, 'd').rescale('h')
        self.flux = self.transit + self.wnoise + self.rnoise
        return self.lcdataset

    @property
    def lcdataset(self):
        return LCDataSet([LCData(self.time, flux, pb) for pb, flux in zip(self.pb_names, self.flux.T)], self.instrument)


    def plot(self, figsize=(13, 4), yoffset=0.01):
        fig, axs = pl.subplots(1, 3, figsize=figsize, sharex='all', sharey='all')
        yshift = yoffset * arange(4)
        axs[0].plot(self.time_h, self.flux + yshift)
        axs[1].plot(self.time_h, self.transit + yshift)
        axs[2].plot(self.time_h, 1 + self.rnoise + yshift)
        pl.setp(axs, xlabel='Time [h]', xlim=self.time_h[[0, -1]])
        pl.setp(axs[0], ylabel='Normalised flux')
        [pl.setp(ax, title=title) for ax, title in
         zip(axs, 'Transit model + noise, Transit model, Red noise'.split(', '))]
        fig.tight_layout()
        return fig, axs

    def plot_color_difference(self, figsize=(13, 4)):
        fig, axs = pl.subplots(2, 3, figsize=figsize, sharex='all', sharey='all')
        [ax.plot(self.time_h, 100 * (fl - self.transit[:, -1])) for ax, fl in zip(axs[0], self.transit[:, :-1].T)]
        [ax.plot(self.time_h, 100 * (fl - self.flux[:, -1])) for ax, fl in zip(axs[1], self.flux[:, :-1].T)]
        [pl.setp(ax, title='F$_{}$ - F$_z$'.format(pb)) for ax, pb in zip(axs[0], self.pb_names[:-1])]
        pl.setp(axs[:, 0], ylabel='$\Delta F$ [%]')
        pl.setp(axs[1, :], xlabel='Time [h]')
        pl.setp(axs, xlim=self.time_h[[0, -1]])
        fig.tight_layout()
        return fig
