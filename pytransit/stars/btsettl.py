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

from os.path import join

import astropy.io.fits as pf
import pandas as pd
from astropy.table import Table
from numba import njit
from numpy import linspace, zeros, array, arange
from pkg_resources import resource_filename
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

__all__ = ['bt_settl_file', 'compute_averaged_bt_settl_table', 'create_bt_settl_interpolator', 'read_bt_settl_table']

bt_settl_file = resource_filename(__name__, join("data", "avg_bt_settl.fits"))

@njit
def bin_spectrum(wl, fl, lmin: int = 10, lmax: int = 30_000):
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


def read_spectrum(fname):
    df = Table.read(fname).to_pandas()
    wl = df.Wavelength.values.astype('d') * 1000
    fl = df.Flux.values.astype('d')
    return wl, fl


def get_teff(f):
    return int(1e2 * float(f.name[3:8]))


def gather_files(datadir):
    files = {}
    for f in sorted(datadir.glob('*.fits.gz')):
        teff = get_teff(f)
        if teff not in files:
            files[teff] = []
        files[teff].append(f)
    return files


def create_table(files):
    teffs = list(files.keys())
    averaged_fluxes = []
    for teff in tqdm(teffs):
        fluxes_per_teff = []
        for f in files[teff]:
            wl, fl = read_spectrum(f)
            bwl, bfl = bin_spectrum(wl, fl)
            fluxes_per_teff.append(bfl)
        averaged_fluxes.append(array(fluxes_per_teff).mean(0))
    return pd.DataFrame(averaged_fluxes, index=pd.Index(teffs, name='TEff [K]'),
                        columns=pd.Index(bwl, name='Wavelength [nm]'))


def write_table(df):
    hdu0 = pf.PrimaryHDU(data=df.values)
    hdu0.header['CRVAL1'] = (10, 'Wavelength of the reference pixel')
    hdu0.header['CDELT1'] = (1, 'Wavelength grid step size')
    hdu0.header['CRPIX1'] = (1, 'Reference pixel coordinates')
    hdu0.header['CTYPE1'] = ('WAVE', 'Type of wavelength grid')
    hdu0.header['CUNIT1'] = ('nm', 'Unit of wavelength grid')
    hdu0.header['CTYPE2'] = ('TEFF', 'TEff array defind in "TEFF" HDU')
    cols = pf.ColDefs([pf.Column(name='TEff', format='E', unit='K', array=df.index.values)])
    hdu1 = pf.BinTableHDU.from_columns(cols)
    hdu1.header['EXTNAME'] = 'TEFF'
    pf.HDUList([hdu0, hdu1]).writeto(bt_settl_file, overwrite=True)


def read_bt_settl_table():
    spectra = pf.getdata(bt_settl_file)
    teff = pf.getdata(bt_settl_file, 1)['TEff'].astype('d')
    wl = pf.getval(bt_settl_file, 'CRVAL1') + arange(spectra.shape[1]) * pf.getval(bt_settl_file, 'CDELT1')
    return pd.DataFrame(spectra, index=pd.Index(teff, name='TEff [K]'), columns=pd.Index(wl, name='Wavelength [nm]'))


def create_bt_settl_interpolator():
    df = read_bt_settl_table()
    rgi = RegularGridInterpolator((df.index.values, df.columns.values.astype('d')), df.values)
    return rgi


def compute_averaged_bt_settl_table(datadir):
    files = gather_files(datadir)
    df = create_table(files)
    write_table(df)