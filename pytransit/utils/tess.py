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

"""Reading TESS SPOC light curves."""

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

    """Read TESS SPOC light curves of a target from a directory.

    Reads the SPOC light curve files (``tess*_lc.fits``) whose TIC number field contains `tic`,
    removes the points with a non-finite time, SAP flux, or PDCSAP flux, and normalises each light
    curve to its median.

    Parameters
    ----------
    tic
        TIC number of the target.
    datadir
        Directory containing the light curve files.
    sectors
        Sectors to read, or ``'all'`` to read all the sectors found.
    use_pdc
        Use the PDCSAP flux if True and the SAP flux otherwise.
    remove_contamination
        Keep the SPOC crowding correction of the PDCSAP flux if True, and revert it using the
        ``CROWDSAP`` keyword if False. Has no effect on the SAP flux, which is never corrected.

    Returns
    -------
    tuple
        ``(time, flux, sector, wn)``, where `time` is the concatenated time array [BJD_TDB],
        `flux` the concatenated normalised flux array, `sector` the sector of each point, and
        `wn` a list with a point-to-point white noise estimate for each light curve.

    Notes
    -----
    The ``QUALITY`` flags are not applied, so points flagged by the SPOC pipeline are kept unless
    their flux is NaN.
    """

    def file_filter(f, partial_tic, sectors):
        """Select the files of the target and the requested sectors by their name."""
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
