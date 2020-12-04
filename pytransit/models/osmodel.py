#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
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
from typing import Union

from astropy.constants import R_sun, M_sun
from matplotlib.pyplot import subplots, setp
from numpy import linspace, meshgrid, sin, cos, array, ndarray, asarray, squeeze

from .transitmodel import TransitModel
from .numba.osmodel import create_star_xy, create_planet_xy, map_osm, xy_taylor_vt, luminosity_v, oblate_model_s
from ..orbits import i_from_ba


class OblateStarModel(TransitModel):
    """Transit model for a gravity-darkened fast-rotating oblate star.

    Transit model for a gravity-darkened fast-rotating oblate star following Barnes (ApJ, 2009, 705).
    """
    def __init__(self, rstar: float = 1.0, wavelength: float = 510, sres: int = 80, pres: int = 5, tres: int = 60):
        """

        Parameters
        ----------
        rstar
            Stellar radius [R_Sun]
        wavelength
            Effective wavelength [nm]
        sres
            Stellar discretization resolution
        pres
            Planet discretization resolution
        """
        super().__init__()

        self.rstar = rstar*R_sun.value     # Stellar equator radius  [m]
        self.wavelength = wavelength*1e-9  # Effective wavelength    [m]
        self.sres = sres                   # Integration resolution for the star
        self.pres = pres                   # Integration resolution for the planet
        self.tres = tres

        self._ts, self._xs, self._ys = create_star_xy(sres)
        self._xp, self._yp = create_planet_xy(pres)

    def visualize(self, k, b, alpha, rho, rperiod, tpole, phi, beta, ldc, ires: int = 256):
        """Visualize the model for a set of parameters.

        Parameters
        ----------
        k
        b
        alpha
        rho
        rperiod
        tpole
        phi
        beta
        ldc
        ires

        Returns
        -------

        """
        a = 4.5
        mstar, ostar, gpole, f, feff = map_osm(self.rstar, rho, rperiod, tpole, phi)
        i = i_from_ba(b, a)
        times = linspace(-1.1, 1.1)
        ox, oy = xy_taylor_vt(times, alpha, -b, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        x = linspace(-1.1, 1.1, ires)
        y = linspace(-1.1, 1.1, ires)
        x, y = meshgrid(x, y)
        sphi, cphi = sin(phi), cos(phi)

        l = luminosity_v(x.ravel()*self.rstar, y.ravel()*self.rstar, mstar, self.rstar, ostar, tpole, gpole,
                         f, sphi, cphi, beta, ldc, self.wavelength)

        fig, axs = subplots(1, 2, figsize=(13, 4))
        axs[0].imshow(l.reshape(x.shape), extent=(-1.1, 1.1, -1.1, 1.1), origin='lower')
        axs[0].plot(ox, oy, 'w', lw=5, alpha=0.25)
        axs[0].plot(ox, oy, 'k', lw=2)

        setp(axs[0], ylabel='y [R$_\star$]', xlabel='x [R$_\star$]')

        times = linspace(-0.35, 0.35, 500)
        flux = oblate_model_s(times, array([k]), 0.0, 4.0, a, alpha, i, 0.0, 0.0, ldc, mstar, self.rstar, ostar, tpole, gpole,
                              f, feff, sphi, cphi, beta, self.wavelength, self.tres, self._ts, self._xs, self._ys, self._xp, self._yp,
                              self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb)

        axs[1].plot(times, flux, 'k')
        setp(axs[1], ylabel='Normalized flux', xlabel='Time - T$_0$')
        fig.tight_layout()

    def evaluate_ps(self, k: Union[float, ndarray], rho: float, rperiod: float, tpole: float, phi: float,
                    beta: float, ldc: ndarray, t0: float, p: float, a: float, i: float, l: float = 0.0,
                    e: float = 0.0, w: float = 0.0, copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar parameters.

        Parameters
        ----------
        k : array-like
            Radius ratio(s) either as a single float or an 1D array
        rho : float
            Stellar density [g/cm^3]
        rperiod : float
            Stellar rotation period [d]
        tpole : float
            Temperature at the pole [K]
        phi : float
            Star's obliquity to the plane of the sky [rad]
        beta: float
            Gravity darkening parameter
        ldc : array-like
            Limb darkening coefficients as a 1D array
        t0 : float
            Zero epoch
        p : float
            Orbital period [d]
        a : float
            Scaled orbital semi-major axis [R_star]
        i : float
            Orbital inclination [rad]
        l : float
            Orbital azimuth angle [rad]
        e : float, optional
            Orbital eccentricity
        w : float, optional
            Argument of periastron

        Notes
        -----
        This version of the `evaluate` method is optimized for calculating a single transit model (such as when using a
        local optimizer). If you want to evaluate the model for a large number of parameters simultaneously, use either
        `evaluate` or `evaluate_pv`.

        Returns
        -------
        ndarray
            Modelled flux as a 1D ndarray.
        """

        ldc = asarray(ldc)
        k = asarray(k)

        if self.time is None:
            raise ValueError("Need to set the data before calling the transit model.")
        if ldc.size != 2*self.npb:
            raise ValueError("The quadratic model needs two limb darkening coefficients per passband")

        mstar, ostar, gpole, f, feff = map_osm(self.rstar, rho, rperiod, tpole, phi)
        sphi, cphi = sin(phi), cos(phi)

        flux = oblate_model_s(self.time, k, t0, p, a, l, i, e, w, ldc, mstar, self.rstar, ostar, tpole, gpole,
                              f, feff, sphi, cphi, beta, self.wavelength, self.tres, self._ts, self._xs, self._ys, self._xp, self._yp,
                              self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb)

        return squeeze(flux)