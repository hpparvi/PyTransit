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

"""Module to model flux contamination in transit light curves.

Module to model flux contamination in transit light curves. The `pytransit.contamination` module contains two flux
contamination models: a physical model based on simulated stellar spectra (`SMContamination`), and a simple one
based on black body approximation (`BBContamination`). The models share an common interface.
"""

from os.path import join
from typing import Union, Iterable

import pandas as pd
import xarray as xa

from numba import njit
from matplotlib.pyplot import subplots, setp
from numpy import transpose, newaxis, uint32, ndarray, asarray, zeros_like, log, exp, ceil, linspace, array, nan
from pandas import DataFrame
from pkg_resources import resource_filename
from scipy.interpolate import interp1d, RegularGridInterpolator

from .instrument import Instrument
from ..utils.phasecurves import planck


@njit
def contaminate_light_curve(flux: ndarray, contamination: ndarray, pbids: ndarray) -> ndarray:
    """Contaminates a transit light curve.

    Contaminates a transit light curve with npb passbands.

    Parameters
    ----------
    flux: 1d array-like
        Transit light curve with npb passbands.
    contamination: 1d array-like
        Array of per-passband contamination values.
    pbids: 1d array-like
        Passband indices that map each light curve element to a single passband.

    Returns
    -------
    Contaminated transit light curve
    """
    npt = flux.size
    contaminated_flux = zeros_like(flux)
    for i in range(npt):
        contaminated_flux[i] = contamination[pbids[i]] + (1.0-contamination[pbids[i]])*flux[i]
    return contaminated_flux


class _BaseContamination:
    """A base class to model flux contamination (blending) in transit light curves.

    A base class to model flux contamination (blending) in transit light curves.

    Notes
    -----
    This class does not implement all the necessary functionality, and is not meant to be used as-is.
    """

    def __init__(self, instrument: Instrument, ref_pb: str) -> None:
        """
        Parameters
        ----------
        instrument
            Instrument configuration
        ref_pb
            name of the reference passband
        """
        self.instrument = instrument
        self._ri = self.instrument.pb_names.index(ref_pb)
        self._rpb = ref_pb

    def relative_fluxes(self, teff:  Union[float, ndarray]):
        raise NotImplementedError

    def relative_flux_mixture(self, teffs: Iterable, fractions: Iterable):
        raise NotImplementedError

    def contamination(self, cr: float, teff1: float, teff2: float):
        raise NotImplementedError

    def exposure_times(self, teff, rtime, rflux=1.0, tflux=1.0):
        """Exposure times that give equal flux as in the reference passband

        Parameters
        ----------
        teff : float
            Effective stellar temperature [K]
        rtime : float
            Exposure time in the reference passband
        rflux : float, optional
            Flux in the reference passband with the reference exposure time
        tflux : float, optional
            Target flux in the reference passband

        """
        return rtime * tflux / rflux / self.relative_fluxes(teff)

    def c_as_pandas(self, ci, teff1, teff2):
        """Contamination as a pandas DataFrame."""
        return pd.DataFrame(self.contamination(ci, teff1, teff2),
                            columns=pd.Index(self.instrument.pb_names, name='passband'),
                            index=pd.Index(uint32(teff2), name='teff'))

    def c_as_xarray(self, ci, teff1, teff2):
        """Contamination as an xarray DataArray."""
        from xarray import DataArray
        return DataArray(self.contamination(ci, teff1, teff2),
                         name='contamination',
                         dims=['teff', 'passband'],
                         coords=[uint32(teff2), self.instrument.pb_names],
                         attrs={'TEff_1': teff1})

    def plot(self, teff1: float, teff2: float, cref, wlref=600, wlmin=305, wlmax=995, nwl=500, figsize=None, axs=None):
        """Plots the contamination model.

        Parameters
        ----------
        teff1
        teff2
        cref
        wlref
        wlmin
        wlmax
        nwl
        figsize

        Returns
        -------

        """
        wl = linspace(wlmin, wlmax, nwl)
        ft = (1 - cref) * self.relative_flux(teff1, wl, wlref)
        fc = cref * self.relative_flux(teff2, wl, wlref)

        if axs is None:
            fig, axs = subplots(2, 1, figsize=figsize, sharex='all', constrained_layout=True)
        else:
            fig = None

        axs[0].plot(wl, ft, label='host')
        axs[0].plot(wl, fc, label='contaminant')
        axs[1].plot(wl, fc / (ft + fc), 'k')

        for ax in axs:
            ax.axhline(cref, ls='--', c='0.75')
            ax.axvline(wlref, ls='--', c='0.75')

        axs[0].legend(fontsize='small')
        setp(axs, xlim=(wlmin, wlmax))
        setp(axs[0], ylabel='Relative flux')
        setp(axs[1], ylim=(-0.05, 1.05), xlabel='Wavelength [nm]', ylabel='Contamination')
        #fig.tight_layout()
        return fig


class BBContamination(_BaseContamination):
    """Third light contamination based on black body approximation.

    This class offers a simple black-body model for flux contamination in which the target star and the contaminant(s)
    are approximated as black bodies with effective temperatures Tt, Tc1, Tc2, ..., Tcn.
    """

    def __init__(self, instrument: Instrument, ref_pb: str, delta_l: float = 10):
        """

        Parameters
        ----------
        instrument
            Instrument configuration.
        ref_pb
            Reference passband name.
        delta_l
            Wavelength grid spacing in nm.
        """
        super().__init__(instrument, ref_pb)
        self._delta_l = delta_l
        self._wl_grids = []
        for f in self.instrument.filters:
            nwl = int(ceil((f.wl_max - f.wl_min) / self._delta_l))
            self._wl_grids.append(linspace(f.wl_min, f.wl_max, nwl))

    @staticmethod
    def absolute_flux(teff: float, wl: Union[float, Iterable]) -> ndarray:
        """The absolute flux given an effective temperature and wavelength.

        Parameters
        ----------
        teff
            The effective temperature in K
        wl
            The wavelength (or an array of) in nm

        Returns
        -------
        Black body spectral radiance.
        """
        return planck(teff, 1e-9 * wl)

    @staticmethod
    def relative_flux(teff: float, wl: Union[float, ndarray], wlref: float) -> ndarray:
        """The black body flux normalized to a given reference wavelength.

        Parameters
        ----------
        teff
            The effective temperature of the radiating body [K]
        wl
            The wavelength [nm]
        wlref
            The reference wavelength [nm]

        Returns
        -------
        The black body flux normalized to a given reference wavelength
        """
        return planck(teff, 1e-9 * wl) / planck(teff, 1e-9 * wlref)

    def absolute_fluxes(self, teff: float) -> ndarray:
        """Calculates the integrated absolute fluxes for all filters for a star with the given effective temperature

        Parameters
        ----------
        teff
            The effective temperature of the radiating body [K]

        Returns
        -------
        The integrated absolute fluxes for the filters in the instrument.
        """
        return array(
            [(self.absolute_flux(teff, g) * f(g)).mean() for f, g in zip(self.instrument.filters, self._wl_grids)])

    def relative_fluxes(self, teff: Union[float, ndarray]) -> ndarray:
        """Calculates the integrated fluxes for all filters normalized to the reference passband.

        Parameters
        ----------
        teff
            The effective temperature of the radiating body [K]

        Returns
        -------
            The integrated fluxes for all filters normalized to the reference passband.
        """
        fluxes = self.absolute_fluxes(teff)
        return fluxes / fluxes[self._ri]

    def contamination(self, cref: float, teff1: float, teff2: float) -> ndarray:
        """Calculates the contamination factors for all the filters given the contamination in the reference passband.

        Parameters
        ----------
        cref
            Reference passband contamination
        teff1
            Host star effective temperature
        teff2
            Contaminant effective temperature

        Returns
        -------
            Contamination factors for all the filters.
        """
        ft = (1.0 - cref) * self.relative_fluxes(teff1)
        fc = cref * self.relative_fluxes(teff2)
        return fc / (ft + fc)


class SMContamination(_BaseContamination):
    """A class that models flux contamination based on stellar spectrum models.
    """

    def __init__(self, instrument: Instrument, ref_pb: str = None) -> None:
        """

          Parameters
          ----------
          instrument
              Instrument configuration
          ref_pb
              name of the reference passband
          """
        super().__init__(instrument, ref_pb)

        self._spectra: DataFrame = DataFrame(pd.read_hdf(resource_filename(__name__, join("data", "spectra.h5")), 'Z0'))
        self._tr_table = trt = xa.open_dataarray(resource_filename(__name__, join("data", "transmission.nc")))
        self._tr_mean = trm = trt.mean(['airmass', 'pwv'])
        self.extinction = interp1d(trm.wavelength, trm, bounds_error=False, fill_value=0.0)
        self.wl = self._spectra.index.values
        self.lte = lte = self._spectra.columns.values
        self._apply_extinction = True

        # 2D interpolator for the spectrum table
        # --------------------------------------
        self._rgi = RegularGridInterpolator((self.wl, self.lte), self._spectra.values)

        # Dataframe indices
        # -----------------
        self.ipb = pd.Index(self.instrument.pb_names, name='passband')
        self.iteff = pd.Index(lte, name='teff')

        # Per-passband fluxes
        # -------------------
        self._compute_relative_flux_tables(0, ref_pb)

    @property
    def apply_extinction(self):
        return self._apply_extinction

    @apply_extinction.setter
    def apply_extinction(self, value):
        self._apply_extinction = value
        self._compute_relative_flux_tables()

    def reddening(self, a):
        k = -a*log(self.wl) + a*log(self.wl[-1])
        return exp(-k)

    def _integrate_fluxes(self, rdc=None):
        rd = self.reddening(rdc) if rdc is not None else 1.
        i, sp, wl = self.instrument, self._spectra, self.wl
        tr = self.extinction(wl) if self._apply_extinction else 1.
        int_fluxes = transpose([(sp.values.T * f(wl) * qe(wl) * tr * rd).sum(1) for f, qe in zip(i.filters, i.qes)])
        self._af = pd.DataFrame(int_fluxes, columns=self.ipb, index=self.iteff)

    def _compute_relative_flux_tables(self, rdc=None, rpb=None):
        self._integrate_fluxes(rdc)
        af = self._af.values
        if rpb is not None:
            self._rpb = rpb
            self._ri = self.instrument.pb_names.index(rpb)
        self._rf = pd.DataFrame(af / af[:, self._ri][:, newaxis], columns=self.ipb, index=self.iteff)
        self._ip = interp1d(self.iteff.values, self._rf.values.T, bounds_error=False, fill_value=nan)

    def absolute_flux(self, teff: float, wl: Union[float, Iterable]) -> ndarray:
        """The absolute flux given an effective temperature and a set of wavelength.

        Parameters
        ----------
        teff: float
            The effective temperature [K]
        wl: array-like
            The wavelengths to calculate the flux in [nm]

        Returns
        -------
        Spectral radiance.
        """
        return self._rgi((wl, teff))

    def relative_flux(self, teff: float, wl: Union[float, ndarray], wlref: float) -> ndarray:
        """The stellar flux normalized to a given reference wavelength.

        Parameters
        ----------
        teff: float
            The effective temperature of the radiating body [K]
        wl: array-like
            The wavelengths to calculate the flux in [nm]
        wlref: float
            The reference wavelength [nm]

        Returns
        -------
        The flux normalized to a given reference wavelength
        """
        return self.absolute_flux(teff, wl) / self.absolute_flux(teff, wlref)

    def relative_fluxes(self, teff: Union[float, ndarray], rdc=None, rpb=None):
        if (rpb is not None and rpb != self._rpb) or (rdc is not None):
            self._compute_relative_flux_tables(rdc, rpb)
        return self._ip(teff)

    def relative_flux_mixture(self, teffs, fractions, rdc=None):
        teffs = asarray(teffs)
        fractions = asarray(fractions)
        x = zeros_like(teffs)

        assert len(teffs) == len(fractions) + 1
        assert 0.0 < fractions.sum() < 1.0

        x[1:] = fractions
        x[0] = 1. - x[1:].sum()
        return (self.relative_fluxes(teffs, rdc) * x).sum(1)

    def contamination(self, cref: Union[float, ndarray], teff1: Union[float, ndarray], teff2: Union[float, ndarray]):
        """Contamination given reference contamination, host TEff, and contaminant TEff(s)

        Per-passband contamination given the contamination in the reference passband and TEffs of the two stars.

        Parameters
        ----------
        cref : float
            contamination in the reference passband
        teff1
            Effective stellar temperature [K]
        teff2
            Effective stellar temperature [K]

        Returns
        -------
        Per-passband contamination
        """
        cref, teff1, teff2 = asarray(cref), asarray(teff1), asarray(teff2)
        a = (1.0 - cref) * self._ip(teff1)
        b = cref * self._ip(teff2)
        return (b / (a + b)).T
