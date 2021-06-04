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
from typing import List, Union, Optional
from astropy.table import Table
import astropy.io.fits as pf
from numpy import concatenate, diff, sqrt, full, median, array


def read_tess_spoc(tic: int,
                   datadir: Union[Path, str],
                   sectors: Optional[Union[List[int], str]] = 'all',
                   use_pdc: bool = False,
                   remove_contamination: bool = True):

    def file_filter(f, partial_tic, sectors):
        _, sector, tic, _, _ = f.name.split('-')
        if sectors != 'all':
            return int(sector[1:]) in sectors and str(partial_tic) in tic
        else:
            return str(partial_tic) in tic

    files = [f for f in sorted(Path(datadir).glob('tess*_lc.fits')) if file_filter(f, tic, sectors)]
    fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
    times, fluxes, sectors = [], [], []
    for f in files:
        tb = Table.read(f)
        bjdrefi = tb.meta['BJDREFI']
        df = tb.to_pandas().dropna(subset=['TIME', 'SAP_FLUX', 'PDCSAP_FLUX'])
        times.append(df['TIME'].values.copy() + bjdrefi)
        fluxes.append(array(df[fcol].values, 'd'))
        fluxes[-1] /= median(fluxes[-1])
        if use_pdc and not remove_contamination:
            contamination = 1 - tb.meta['CROWDSAP']
            fluxes[-1] = contamination + (1 - contamination) * fluxes[-1]
        sectors.append(full(fluxes[-1].size, pf.getval(f, 'sector')))

    return (concatenate(times), concatenate(fluxes), concatenate(sectors),
            [diff(f).std() / sqrt(2) for f in fluxes])
