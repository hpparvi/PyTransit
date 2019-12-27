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

from os.path import join

import pandas as pd
import xarray as xa
from numpy import transpose, newaxis, uint32, ndarray, asarray, zeros_like, log, exp, sqrt
from pkg_resources import resource_filename
from scipy.interpolate import interp1d

from .instrument import Instrument
from ..utils.physics import planck

class Contamination:
    """A base class to deal with transit light curves with flux contamination.
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
        self.instrument = instrument
        self._ri = 0
        self._rpb = self.instrument.pb_names[self._ri]

    def relative_fluxes(self, teff, rdc=None, rpb=None):
        raise NotImplementedError

    def relative_flux_mixture(self, teffs, fractions, rdc=None):
        raise NotImplementedError

    def contamination(self, ci, teff1, teff2, rdc=None, rpb=None):
        raise NotImplementedError

    def exposure_times(self, teff, rtime, rflux=1.0, tflux=1.0, rdc=None, rpb=None):
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
        rpb : str, optional
            Name of the reference passband

        """
        return rtime * tflux / rflux / self.relative_fluxes(teff, rdc, rpb)


class BBContamination(Contamination):
    """Third light contamination based on black body approximation
    """
    def __init__(self, instrument, ref_pb):
        """

        Parameters
        ----------
        :param instrument : Instrument
            Instrument configuration
        :param ref_pb : str, optional
            name of the reference passband
        """
        super().__init__(instrument, ref_pb)
        I = self.instrument

    def cn_fluxes(self, wl, Ttar, Tcon, wlref, cnref):
        """Relative target and contaminant star fluxes given a reference wavelength and a contamination factor.

        Calculates the relative target and contaminant star fluxes
        for wavelengths `wl` given the target and comparison star
        temperatures, a reference wavelength, and a contamination
        factor in the reference wavelength.

        Parameters
        ----------

          wl     : Wavelength [m]
          Ttar   : Target star effective temperature [K]
          Tcon   : Comparison star effective temperature [K]
          wlref  : Reference wavelength [m]
          cnref  : Contamination in the reference wavelength (0-1)

        Returns
        -------

          ftar  : Target flux
          fcon  : contaminant flux
        """
        ftar = (1 - cnref) * planck(Ttar, wl) / planck(Ttar, wlref)
        fcon = cnref * planck(Tcon, wl) / planck(Tcon, wlref)
        return ftar, fcon


    def contamination(self, wl, Ttar, Tcon, wlref, cnref):
        """Contamination given a reference wavelength and a contamination factor.

        Calculates the contamination factor for wavelengths `wl`
        given the target and comparison star temperatures, a
        reference wavelength, and a contamination factor in the
        reference wavelength.


        Parameters
        ----------

          wl     : Wavelength [m]
          Ttar   : Target star effective temperature [K]
          Tcon   : Comparison star effective temperature [K]
          wlref  : Reference wavelength [m]
          cnref  : Contamination in the reference wavelength (0-1)

        Returns
        -------

          c : Contamination in the given wavelength(s)

        """
        ftar, fcon = self.cn_fluxes(wl, Ttar, Tcon, wlref, cnref)
        return fcon / (ftar + fcon)



class SMContamination(Contamination):
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
        I = self.instrument

        self._spectra = sp = pd.read_hdf(resource_filename(__name__, join("data", "spectra.h5")), 'Z0')
        self._tr_table = trt = xa.open_dataarray(resource_filename(__name__, join("data", "transmission.nc")))
        self._tr_mean = trm = trt.mean(['airmass', 'pwv'])
        self.extinction = interp1d(trm.wavelength, trm, bounds_error=False, fill_value=0.0)
        self.wl = wl = sp.index.values
        self.lte = lte = sp.columns.values
        self._apply_extinction = True

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
        I, sp, wl = self.instrument, self._spectra, self.wl
        tr = self.extinction(wl) if self._apply_extinction else 1.
        int_fluxes = transpose([(sp.values.T * f(wl) * qe(wl) * tr * rd).sum(1) for f, qe in zip(I.filters, I.qes)])
        self._af = pd.DataFrame(int_fluxes, columns=self.ipb, index=self.iteff)

    def _compute_relative_flux_tables(self, rdc=None, rpb=None):
        self._integrate_fluxes(rdc)
        af = self._af.values
        if rpb is not None:
            self._rpb = rpb
            self._ri = self.instrument.pb_names.index(rpb)
        self._rf = pd.DataFrame(af / af[:, self._ri][:, newaxis], columns=self.ipb, index=self.iteff)
        self._ip = interp1d(self.iteff.values, self._rf.values.T, bounds_error=True)

    def relative_fluxes(self, teff, rdc=None, rpb=None):
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

    def exposure_times(self, teff, rtime, rflux=1.0, tflux=1.0, rdc=None, rpb=None):
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
        rpb : str, optional
            Name of the reference passband

        """
        return rtime * tflux / rflux / self.relative_fluxes(teff, rdc, rpb)

    def contamination(self, ci, teff1, teff2, rdc=None, rpb=None):
        """Contamination given reference contamination, host TEff, and contaminant TEff(s)

        Per-passband contamination given the contamination in the reference passband and TEffs of the two stars.

        Parameters
        ----------
        ci : float
            contamination in the reference passband
        teff : float
            Effective stellar temperature [K]
        rpb : str, optional
            Name of the reference passband

        Returns
        -------
        contamination : ndarray

        """
        if rpb is not None:
            self._compute_relative_flux_tables(rdc, rpb)
        a = (1 - ci) * self._ip(teff1)
        b = ci * self._ip(teff2)
        return b.T / (a + b.T)

    def c_as_pandas(self, ci, teff1, teff2, rdc=None, rpb=None):
        """Contamination as a pandas DataFrame."""
        return pd.DataFrame(self.contamination(ci, teff1, teff2, rdc, rpb),
                            columns=pd.Index(self.pb_names, name='passband'),
                            index=pd.Index(uint32(teff2), name='teff'))

    def c_as_xarray(self, ci, teff1, teff2, rdc=None, rpb=None):
        """Contamination as an xarray DataArray."""
        from xarray import DataArray
        return DataArray(self.contamination(ci, teff1, teff2, rdc, rpb),
                         name='contamination',
                         dims=['teff', 'passband'],
                         coords=[uint32(teff2), self.pb_names],
                         attrs={'TEff_1': teff1})
