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

from numpy import ceil, sqrt, where, inf
from matplotlib.pyplot import subplots
from pytransit.contamination import Instrument, SMContamination
from pytransit.contamination.filter import sdss_g, sdss_r, sdss_i, sdss_z

from pytransit.lpf.cntlpf import PhysContLPF

from .mocklc import MockLC

class MockLPF(PhysContLPF):
    def __init__(self, name: str, lc: MockLC):

        super().__init__(name, passbands=lc.pb_names, times=lc.npb * [lc.time],
                         fluxes=list(lc.flux.T), pbids=list(range(lc.npb)))

        self._lc = lc
        self.know_host = lc.setup.know_host
        self.misidentify_host = lc.setup.misidentify_host
        self.hteff = lc.hteff if not self.misidentify_host else lc.cteff
        self.cteff = lc.cteff
        self.t0_bjd = 0.0
        self.period = lc.p
        self.sma = lc.a
        self.inc = lc.i
        self.k_apparent = lc.k_apparent
        self.b = lc.b

        self.set_prior('p', 'NP', lc.p, 1e-7)
        if lc.setup.know_orbit:
            self.set_prior('rho', 'NP', 5.0, 0.05)
            self.set_prior('b', 'NP', lc.b, 0.01)

        if lc.setup.know_host:
            if lc.setup.misidentify_host:
                self.set_prior('teff_h', 'NP', self._lc.cteff, 10)
            else:
                self.set_prior('teff_h', 'NP', self._lc.hteff, 10)

    def _init_instrument(self):
        """Set up the instrument and contamination model."""
        self.instrument = Instrument('MockInstrument', [sdss_g, sdss_r, sdss_i, sdss_z])
        self.cm = SMContamination(self.instrument, "i'")
        self._additional_log_priors.append(lambda pv: where(pv[:, 4] < pv[:, 5], 0, -inf))

    def plot_light_curves(self, ncols: int = 2, figsize: tuple = (13, 5)):
        nrows = int(ceil(self.nlc) / ncols)
        fig, axs = subplots(nrows, ncols, figsize=figsize, sharex='all', sharey='all', constrained_layout=True)
        fmodel = self.flux_model(self.de.population)[self.de.minimum_index]
        for i, ax in enumerate(axs.flat):
            ax.plot(self.times[i], self.fluxes[i], '.', alpha=0.25)
            ax.plot(self.times[i], fmodel[self.lcslices[i]], 'k')

    def posterior_samples(self, burn: int = 0, thin: int = 1, include_ldc: bool = False):
        df = super().posterior_samples(burn, thin, include_ldc)
        df['k_app'] = sqrt(df.k2_app)
        df['k_true'] = sqrt(df.k2_true)
        df['cnt'] = 1. - df.k2_app / df.k2_true
        return df
