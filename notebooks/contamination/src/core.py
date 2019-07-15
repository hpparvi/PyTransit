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

import pandas as pd
import xarray as xa

from numpy import array, sqrt, asarray


class LCData:
    def __init__(self, time, flux, pb):
        self.time = asarray(time)
        self.flux = asarray(flux)
        self.npt = self.time.size
        self.pb = pb


class LCDataSet:
    def __init__(self, data, instrument):
        self.data = data
        self.times = [d.time for d in data]
        self.fluxes = [d.flux for d in data]
        self.pbs = pd.Categorical([d.pb for d in data],
                                  categories=instrument.pb_names,
                                  ordered=True).remove_unused_categories()
        self.npb = self.pbs.unique().size
        self.pbis = self.pbs.codes


def load_simulation(fname, burn=0, thin=1, return_chains=False):
    chains = xa.open_dataarray(fname).load()
    chains.close()
    if return_chains:
        return chains
    else:
        fc = pd.DataFrame(array(chains)[:, burn::thin, :].reshape([-1, chains.shape[2]]), columns=chains.coords['parameter'])
        fc['k_app'] = sqrt(fc.ara)
        fc['k_true'] = sqrt(fc.art)
        fc['cnr'] = 1 - fc.ara/fc.art
        fc['k_ratio'] = 1/sqrt(1-fc.cnr)

        q1cols = [c for c in fc.columns if 'q1' in c]
        q2cols = [c for c in fc.columns if 'q2' in c]

        ucols = [q.replace('q1', 'u') for q in q1cols]
        vcols = [q.replace('q2', 'v') for q in q2cols]

        for q1c, q2c, uc, vc in zip(q1cols, q2cols, ucols, vcols):
            a, b = sqrt(fc[q1c].values), 2 * fc[q2c].values
            fc[uc] = a * b
            fc[vc] = a * (1. - b)
        return fc
