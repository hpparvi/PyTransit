#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2021  Hannu Parviainen
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
from pathlib import Path
from typing import Iterable

from astropy.table import Column
from matplotlib.pyplot import subplots, setp
from numpy import ndarray, array, median, floor, percentile, squeeze
from numpy.random import permutation

from .lpf import BaseLPF
from .. import TransitModel


class TransitLPF(BaseLPF):
    def __init__(self, name: str, passbands: Iterable, times: Iterable, fluxes: Iterable, errors: list = None,
                 pbids: list = None, covariates: list = None, wnids: list = None, tm: TransitModel = None,
                 nsamples: tuple = 1, exptimes: tuple = 0., init_data=True, result_dir: Path = None, tref: float = 0.0,
                 lnlikelihood: str = 'wn'):

        def transform_input(a):
            if isinstance(a, (list, tuple)):
                return a
            elif isinstance(a, Column):
                return [a.data.astype('d')]
            elif isinstance(a, ndarray):
                return [a]

        times = transform_input(times)
        fluxes = transform_input(fluxes)

        super().__init__(name, passbands, times, fluxes, errors, pbids, covariates, wnids, tm,
                         nsamples, exptimes, init_data, result_dir, tref, lnlikelihood)

    def plot_light_curve(self, model: str = 'de', figsize: tuple = (13, 4)):
        fig, ax = subplots(figsize=figsize, constrained_layout=True)
        tref = floor(self.timea.min())

        if model == 'de':
            pv = self.de.minimum_location
            err = 10 ** pv[7]
        elif model == 'mc':
            fc = array(self.posterior_samples())
            pv = permutation(fc)[:300]
            err = 10 ** median(pv[:, 7], 0)

        ax.errorbar(self.timea - tref, self.ofluxa, err, fmt='.', c='C0', alpha=0.75)

        if model == 'de':
            ax.plot(self.timea - tref, squeeze(self.flux_model(pv)), c='C0')
        if model == 'mc':
            flux_pr = self.flux_model(fc[permutation(fc.shape[0])[:1000]])
            flux_pc = array(percentile(flux_pr, [50, 0.15, 99.85, 2.5, 97.5, 16, 84], 0))
            [ax.fill_between(self.timea - tref, *flux_pc[i:i + 2, :], alpha=0.2, facecolor='C0') for i in
             range(1, 6, 2)]
            ax.plot(self.timea - tref, flux_pc[0], c='C0')
        setp(ax, xlim=self.timea[[0, -1]] - tref, xlabel=f'Time - {tref:.0f} [BJD]', ylabel='Normalised flux')
        return fig, ax