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
from typing import Union, Iterable

from astropy.constants import R_sun, M_sun
from matplotlib.patches import Circle
from matplotlib.pyplot import subplots, setp
from numpy import linspace, sin, cos, array, ndarray, asarray, squeeze, cross, newaxis, pi, where, nan, full, degrees, \
    zeros, polyfit, atleast_2d, atleast_1d, nansum, zeros_like
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.spatial.transform.rotation import Rotation

from .transitmodel import TransitModel
from .numba.gdmodel import create_star_xy, create_planet_xy, map_osm, xy_taylor_vt, oblate_model_s, \
    luminosity_v2, planck, create_star_luminosity
from ..contamination.filter import Filter, DeltaFilter
from ..stars import read_bt_settl_table, read_husser2013_table
from ..orbits import as_from_rhop, i_from_baew
from ..orbits.taylor_z import vajs_from_paiew, find_contact_point
from ..utils.octasphere import octasphere


def bt_settl_table(filters, tmin: float = 1200., tmax: float = 7000., nt: int = 200):
    """Calculates an interpolation table for passband-integrated flux as a function of temperature.
    """
    if tmax > 7000:
        raise ValueError('tmax for BT-SETTL model has to be smaller or equal to 7000 K.')
    if tmin < 1200:
        raise ValueError('tmax for BT-SETTL model has to be larger or equal to 1200 K.')

    temperatures = linspace(tmin, tmax, nt)
    npb = len(filters)

    spectra = read_bt_settl_table()
    wl_table = spectra.columns.values.astype('d')
    t_table = spectra.index.values
    spectra = interp1d(wl_table, spectra, bounds_error=False, fill_value=0.0)
    fluxes = zeros((npb, t_table.size))
    for i, f in enumerate(filters):
        wl, tr = f.sample()
        if wl.ndim == 0:
            fluxes[i] = spectra(wl) * tr
        else:
            fluxes[i] = (spectra(wl) * tr).mean(1)
    fluxes = interp1d(t_table, fluxes)(temperatures)
    fluxes /= fluxes.max(1)[:, newaxis]

    return fluxes, tmin, temperatures[1] - temperatures[0]


def husser2013_table(filters, tmin: float = 2500, tmax: float = 12000, nt: int = 200):
    """Calculates an interpolation table for passband-integrated flux as a function of temperature.
    """
    if tmax > 12000:
        raise ValueError('tmax for the Husser2013 model has to be smaller or equal to 12000 K.')
    if tmin < 2500:
        raise ValueError('tmax for the Husser2013 model has to be larger or equal to 2500 K.')

    temperatures = linspace(tmin, tmax, nt)
    npb = len(filters)

    spectra = read_husser2013_table()
    wl_table = spectra.columns.values.astype('d')
    t_table = spectra.index.values
    spectra = interp1d(wl_table, spectra, bounds_error=False, fill_value=0.0)
    fluxes = zeros((npb, t_table.size))
    for i, f in enumerate(filters):
        wl, tr = f.sample()
        if wl.ndim == 0:
            fluxes[i] = spectra(wl) * tr
        else:
            fluxes[i] = (spectra(wl) * tr).mean(1)
    fluxes = interp1d(t_table, fluxes)(temperatures)
    fluxes /= fluxes.max(1)[:, newaxis]
    return fluxes, tmin, temperatures[1] - temperatures[0]


def blackbody_table(filters, tmin: float = 1200, tmax: float = 12000, nt: int = 200):
    """Calculates an interpolation table for passband-integrated flux as a function of temperature.
    """
    temperatures = linspace(tmin, tmax, nt)
    npb = len(filters)

    fluxes = zeros((npb, nt))
    for i, f in enumerate(filters):
        wl, tr = f.sample()
        wl *= 1e-9
        for j, t in enumerate(temperatures):
            fluxes[i, j] = (planck(wl, t) * tr).mean()
    fluxes /= fluxes.max(1)[:,newaxis]
    return fluxes, tmin, temperatures[1] - temperatures[0]


class GravityDarkenedModel(TransitModel):
    """Transit model for a gravity-darkened fast-rotating oblate star.

    Transit model for a gravity-darkened fast-rotating oblate star following Barnes (ApJ, 2009, 705).
    """

    def __init__(self, filters: Union[float, Filter, Iterable[Filter]], rstar: float = 1.0,
                 sres: int = 80, pres: int = 5, tres: int = 60, model: str = 'blackbody',
                 tmin: float = 5000, tmax: float = 7000, accurate_mu: bool = True):
        """

        Parameters
        ----------
        filters
            Either the effective wavelength [nm], a Filter object, or a list of Filter objects.
        rstar
            Stellar radius [R_Sun]
        sres
            Stellar discretization resolution.
        pres
            Planet discretization resolution.
        tres
            Orbit discretization resolution.
        model
            The spectroscopic model to use. Can be either 'blackbody', 'husser2013', or 'bt-settl'.
        tmin
            Minimum allowed temperature [K].
        tmax
            Maximum allowed temperature [K].
        """
        super().__init__()

        assert model in ('blackbody', 'husser2013', 'bt-settl')
        self.model = model
        self.accurate_mu = accurate_mu

        if isinstance(filters, (float, int)):
            self.filters = (DeltaFilter('default', float(filters)),)
        elif isinstance(filters, Filter):
            self.filters = (filters,)
        else:
            self.filters = filters

        self.rstar = rstar*R_sun.value     # Stellar equator radius  [m]
        self.sres = sres                   # Integration resolution for the star
        self.pres = pres                   # Integration resolution for the planet
        self.tres = tres
        self.tmin = tmin                   # Minimum temperature [K]
        self.tmax = tmax                   # Maximum temperature [K]

        if self.model == 'blackbody':
            self._flux_table, self._teff0, self._dteff = blackbody_table(self.filters, tmin, tmax)
        elif self.model == 'husser2013':
            self._flux_table, self._teff0, self._dteff = husser2013_table(self.filters, tmin, tmax)
        else:
            self._flux_table, self._teff0, self._dteff = bt_settl_table(self.filters, tmin, tmax)

        self._ts, self._xs, self._ys = create_star_xy(sres)
        self._xp, self._yp = create_planet_xy(pres)

    def visualize(self, k, p, rho, b, e, w, alpha, rperiod, tpole, istar, beta, ldc, figsize=(5, 5), ax=None,
                  ntheta=18, vmin: float = 0.0, vmax: float = 1.0, passband: int = 0):
        if ax is None:
            fig, ax = subplots(figsize=figsize)
            ax.set_aspect(1.)
        else:
            fig, ax = None, ax

        ldc = atleast_2d(ldc)
        a = as_from_rhop(rho, p)
        inc = i_from_baew(b, a, e, w)
        mstar, ostar, gpole, f, _ = map_osm(rstar=self.rstar, rho=rho, rperiod=rperiod, tpole=tpole, phi=0.0)

        # Plot the star
        # -------------
        vertices_original, faces = octasphere(4)
        vertices = vertices_original.copy()
        vertices[:, 1] *= (1.0 - f)

        triangles = vertices[faces]
        centers = triangles.mean(1)
        normals = cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        nlength = norm(normals, axis=1)
        normals /= nlength[:, newaxis]

        rotation = Rotation.from_rotvec((0.5 * pi - istar) * array([1, 0, 0]))
        rn = rotation.apply(normals)
        rc = rotation.apply(centers)

        mask = rn[:, 2] < 0.0
        l = luminosity_v2(centers[mask], normals[mask], istar, mstar, self.rstar, ostar, tpole, gpole, beta,
                         ldc, passband, self._flux_table, self._teff0, self._dteff)
        l /= l.max()
        ax.tripcolor(rc[mask, 0], rc[mask, 1], l, shading='gouraud', vmin=vmin, vmax=vmax)

        nphi = 180
        theta = linspace(0 + 0.1, pi - 0.1, ntheta)
        phi = linspace(0, 2 * pi, nphi)
        for i in range(theta.size):
            y = (1.0 - f) * cos(theta[i])
            x = cos(phi) * sin(theta[i])
            z = sin(phi) * sin(theta[i])
            v = rotation.apply(array([x, full(nphi, y), z]).T)
            m = v[:, 2] < 0.0
            ax.plot(where(m, v[:, 0], nan), v[:, 1], 'k--', lw=1.5, alpha=0.25)

        # Plot the orbit
        # --------------
        y0, vx, vy, ax_, ay, jx, jy, sx, sy = vajs_from_paiew(p, a, inc, e, w)
        c1 = find_contact_point(k, 1, y0, vx, vy, ax_, ay, jx, jy, sx, sy)
        c4 = find_contact_point(k, 4, y0, vx, vy, ax_, ay, jx, jy, sx, sy)
        time = linspace(2 * c1, 2 * c4, 100)

        ox, oy = xy_taylor_vt(time, alpha, y0, vx, vy, ax_, ay, jx, jy, sx, sy)
        ax.plot(ox, oy, 'k')

        pxy = xy_taylor_vt(array([0.0]), alpha, y0, vx, vy, ax_, ay, jx, jy, sx, sy)
        ax.add_artist(Circle(pxy, k, zorder=10, fc='k'))

        # Plot the info
        # -------------
        ax.text(0.025, 0.95, f"i$_\star$ = {degrees(istar):.1f}$^\circ$", transform=ax.transAxes)
        ax.text(0.025, 0.90, f"i$_\mathrm{{p}}$ = {degrees(inc):.1f}$^\circ$", transform=ax.transAxes)
        ax.text(1 - 0.025, 0.95, fr"$\alpha$ = {degrees(alpha):.1f}$^\circ$", transform=ax.transAxes, ha='right')
        ax.text(0.025, 0.05, f"f = {f:.1f}", transform=ax.transAxes)

        setp(ax, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[], yticks=[])
        if fig is not None:
            fig.tight_layout()
        return ax

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
        This version of the `evaluate` method is optimized for calculating a single transit model. If you want to
        evaluate the model for a large number of parameters simultaneously, use `evaluate`.

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

        flux = oblate_model_s(self.time, k, t0, p, a, l, i, e, w, ldc, mstar, self.rstar, ostar, tpole, gpole, f, feff,
                              sphi, cphi, beta, self._flux_table, self._teff0, self._dteff,
                              self.tres, self._ts, self._xs, self._ys,
                              self._xp, self._yp, self.lcids, self.pbids, self.nsamples, self.exptimes, self.npb, self.accurate_mu)

        return squeeze(flux)


    def evaluate_brute(self, k: Union[float, ndarray], rho: float, rperiod: float, tpole: float, phi: float,
                       beta: float, ldc: ndarray, t0: float, p: float, a: float, i: float, l: float = 0.0,
                       e: float = 0.0, w: float = 0.0, copy: bool = True, plot: bool = False, res: int = 300) -> ndarray:
        mstar, ostar, gpole, f, feff = map_osm(self.rstar, rho, rperiod, tpole, phi)
        sphi, cphi = sin(phi), cos(phi)
        ldc = atleast_2d(ldc)
        k = atleast_1d(k)

        st, sx, sy = create_star_xy(res)
        fstar = create_star_luminosity(res, sx, sy, mstar, self.rstar, ostar, tpole, gpole, f,
                                       sphi, cphi, beta, ldc, self._flux_table, self._teff0, self._dteff, self.accurate_mu)
        px, py = xy_taylor_vt(self.time - t0, l, *vajs_from_paiew(p, a, i, e, w))

        if plot:
            fig, ax = subplots()
            ax.imshow(fstar[0], extent=(-1, 1, -1, 1))
            ax.plot(px, py)
            fig.tight_layout()

        fnorm = [nansum(fstar[ipb]) for ipb in range(self.npb)]
        rflux = zeros_like(px)
        for j in range(px.size):
            ilc = self.lcids[j]
            ipb = self.pbids[ilc]
            m = ~((sx - px[j]) ** 2 + (sy - py[j]) ** 2 < k[0] ** 2).reshape((res, res))
            rflux[j] = nansum(fstar[ipb] * m) / fnorm[ipb]
        return rflux