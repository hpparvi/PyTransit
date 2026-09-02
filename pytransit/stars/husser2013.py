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

from importlib.resources import files

import astropy.io.fits as pf
import pandas as pd
from numba import njit
from numpy import linspace, zeros, array, arange, exp
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

__all__ = ['husser2013_file', 'compute_averaged_husser2013_table', 'create_husser2013_interpolator', 'read_husser2013_table']

husser2013_file = str(files(__package__).joinpath("data", "avg_husser2013.fits"))


def get_teff(f):
    return int(f.name[3:8])


def gather_files(datadir):
    files = {}
    for f in sorted(datadir.glob('lte*HiRes.fits')):
        teff = get_teff(f)
        if teff not in files:
            files[teff] = []
        files[teff].append(f)
    return files


def read_spectrum(fname, wl = None):
    fl = d = pf.getdata(fname).astype('d')
    if wl is None:
        h = pf.getheader(fname)
        wl = 0.1*exp(h['CRVAL1'] + arange(h['NAXIS1'])*h['CDELT1'])
    return wl, fl


@njit
def bin_spectrum(wl, fl, lmin: int = 300, lmax: int = 2500):
    bwl = linspace(lmin, lmax, lmax - lmin + 1)
    bfl = zeros(bwl.size)
    istart, iend = 0, 0
    for ibin in range(bfl.size):
        cwl = bwl[ibin]
        n = 0
        while wl[istart] < cwl - 0.5:
            istart += 1
        iend = istart
        while wl[iend] < cwl + 0.5 and iend < wl.size - 1:
            bfl[ibin] += fl[istart]
            n += 1
            iend += 1
        istart = iend
        if n > 0:
            bfl[ibin] /= n
    return bwl, bfl


def average_over_teff(files, teff):
    wl, fl = None, None
    for f in tqdm(files[teff], leave=False):
        if wl is None:
            wl, fl = read_spectrum(f)
        else:
            _, flc = read_spectrum(f, wl)
            fl += flc
    fl /= len(files[teff])
    return wl, fl


def create_table(files):
    teffs = list(files.keys())
    spectra = []
    for teff in tqdm(teffs):
        wl, fl = average_over_teff(files, teff)
        bwl, bfl = bin_spectrum(wl, fl)
        spectra.append(bfl)
    spectra = array(spectra)
    return pd.DataFrame(spectra, index=pd.Index(teffs, name='TEff [K]'),
                        columns=pd.Index(bwl, name='Wavelength [nm]'))


def write_table(df):
    hdu0 = pf.PrimaryHDU(data=df.values)
    hdu0.header['CRVAL1'] = (300, 'Wavelength of the reference pixel')
    hdu0.header['CDELT1'] = (1, 'Wavelength grid step size')
    hdu0.header['CRPIX1'] = (1, 'Reference pixel coordinates')
    hdu0.header['CTYPE1'] = ('WAVE', 'Type of wavelength grid')
    hdu0.header['CUNIT1'] = ('nm', 'Unit of wavelength grid')
    hdu0.header['CTYPE2'] = ('TEFF', 'TEff array defind in "TEFF" HDU')
    cols = pf.ColDefs([pf.Column(name='TEff', format='E', unit='K', array=df.index.values)])
    hdu1 = pf.BinTableHDU.from_columns(cols)
    hdu1.header['EXTNAME'] = 'TEFF'
    pf.HDUList([hdu0, hdu1]).writeto(husser2013_file, overwrite=True)


def read_husser2013_table():
    """Read the wavelength-binned Husser et al. (2013) PHOENIX stellar spectrum table shipped with PyTransit.

    Reads the precomputed spectrum table from ``husser2013_file``. The table holds spectra
    averaged over surface gravity and metallicity and binned to a uniform 1 nm wavelength grid
    covering 300-2500 nm at 73 effective temperatures from 2300 K to 12000 K. The binning keeps
    the table small enough to distribute with the package while remaining accurate enough for
    broadband flux ratios. The temperature coverage extends to hotter stars than the BT-Settl
    grid, but the wavelength coverage is narrower.

    Returns
    -------
    pandas.DataFrame
        Spectra with the effective temperatures in K as the index and the wavelengths in nm as
        the columns.

    See Also
    --------
    create_husser2013_interpolator : an interpolator over the same table.
    """
    spectra = pf.getdata(husser2013_file).astype('d')
    teff = pf.getdata(husser2013_file, 1)['TEff'].astype('d')
    wl = pf.getval(husser2013_file, 'CRVAL1') + arange(spectra.shape[1]) * pf.getval(husser2013_file, 'CDELT1')
    return pd.DataFrame(spectra, index=pd.Index(teff, name='TEff [K]'), columns=pd.Index(wl, name='Wavelength [nm]'))


def create_husser2013_interpolator():
    """Create a 2D interpolator over the Husser et al. (2013) PHOENIX spectrum table.

    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
        Interpolator over (effective temperature [K], wavelength [nm]) returning the flux.

    Examples
    --------
    ::

        from pytransit.stars import create_husser2013_interpolator

        ip = create_husser2013_interpolator()
        flux = ip([[5500.0, 650.0]])
    """
    df = read_husser2013_table()
    rgi = RegularGridInterpolator((df.index.values, df.columns.values.astype('d')), df.values)
    return rgi


def compute_averaged_husser2013_table(datadir):
    """Recompute the binned Husser et al. (2013) PHOENIX spectrum table from the original spectra.

    Reads the original Husser et al. (2013) PHOENIX spectra from `datadir`, averages them over surface gravity
    and metallicity for each effective temperature, bins them to a 1 nm grid, and overwrites the
    table at ``husser2013_file``.

    This is a package maintenance function: PyTransit ships with a precomputed table, so it only
    needs to be run to rebuild that table from a locally downloaded spectrum grid.

    Parameters
    ----------
    datadir : pathlib.Path
        Directory holding the original spectra as ``*.fits.gz`` files.
    """
    files = gather_files(datadir)
    df = create_table(files)
    write_table(df)
