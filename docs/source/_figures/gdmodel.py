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

"""Plotting helpers for the gravity-darkened star pages.

The feature page is about what a gravity-darkened star does to a transit that an ordinary
limb-darkened star cannot. Its figure therefore shows the stellar disk with the planet's path beside
the residuals left after fitting a standard transit model.

The model page explains how the model works. `plot_surface_maps` shows the chain from the local
gravity to the brightness in a passband, and `plot_integration` the discretisation of the planet
and its path that the light curve is computed from.
"""

from math import cos, sin

from matplotlib.patches import Arc, Circle, Ellipse
from matplotlib.patheffects import Stroke, Normal
from matplotlib.pyplot import subplots, setp
from numpy import (arccos, array, atleast_2d, degrees, hypot, isfinite, linspace, meshgrid, nanmax, nanmin, pi, sqrt,
                   where)
from scipy.constants import G
from scipy.optimize import least_squares

from meepmeep.backends.numba.point2d import solve2d

from pytransit import GravityDarkenedModel, RoadRunnerModel
from pytransit.contamination import BoxcarFilter
from pytransit.models.numba.gdmodel import (calculate_luminosity_interpolation_table, create_planet_xy,
                                            create_star_luminosity, create_star_xy, luminosity_v, map_osm,
                                            xy_taylor_st, xy_taylor_vt, z_v)

__all__ = ['plot_gravity_darkening_signal', 'plot_surface_maps', 'plot_integration', 'plot_geometry']

# The example star of both pages: an A-type star of 1.8 solar masses and radii rotating once in
# 11 hours, seen with its pole tilted towards the observer, and a planet on a misaligned orbit.
RSTAR, RHO, RPERIOD, TPOLE, PHI, BETA = 1.8, 0.48, 0.47, 8500.0, 0.5, 0.22
K, LDC, P, A, B, LAMBDA = 0.1, (0.3, 0.2), 3.0, 6.0, 0.3, pi / 3


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _standard_fit(time, flux, p, a0):
    """The light curve of the best-fitting spherical, limb-darkened star.

    The radius ratio, scaled semi-major axis, impact parameter, transit centre, and quadratic limb
    darkening coefficients are all free, so whatever remains is what gravity darkening adds.
    """
    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)

    def residuals(x):
        k, a, b, t0, u1, u2 = x
        return tm.evaluate(k, [u1, u2], t0, p, a, arccos(b / a)) - flux

    bounds = ([0.03, 2.0, 0.0, -0.01, -1.0, -1.0], [0.3, 20.0, 0.95, 0.01, 2.0, 2.0])
    best = None
    for bs in (0.05, 0.3, 0.5, 0.7):
        fit = least_squares(residuals, [0.1, a0, bs, 0.0, 0.3, 0.2], bounds=bounds, x_scale='jac',
                            xtol=1e-14, ftol=1e-14)
        if best is None or fit.cost < best.cost:
            best = fit
    return flux + best.fun          # `fun` is the model minus the data


def _draw_star(ax, tm, time, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l, res=300):
    """Draw the gravity-darkened stellar disk and the planet's path, as `evaluate` sees them."""
    mstar, ostar, gpole, f, feff = map_osm(tm.rstar, rho, rperiod, phi)
    st, sx, sy = create_star_xy(res)
    image = create_star_luminosity(res, sx, sy, mstar, tm.rstar, ostar, tpole, gpole, f, sin(phi),
                                   cos(phi), beta, atleast_2d(ldc), tm._flux_table, tm._teff0,
                                   tm._dteff, tm.accurate_mu)[0]
    ax.imshow(image / nanmax(image), extent=(-1, 1, -1, 1), origin='lower', cmap='inferno',
              vmin=0.4, vmax=1.0)
    px, py = xy_taylor_vt(time - t0, l, solve2d(0.0, p, a, i, 0.0, 0.0))
    ax.plot(px, py, c='w', lw=1, ls='--')
    # An arrowhead where the path leaves the star shows the direction of motion.
    j = max(n for n in range(px.size) if px[n] ** 2 + py[n] ** 2 < 1.0)
    ax.annotate('', xy=(px[j], py[j]), xytext=(px[j - 15], py[j - 15]),
                arrowprops=dict(arrowstyle='-|>', color='w', lw=1))
    setp(ax, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    ax.set_aspect('equal')
    ax.axis('off')


def plot_gravity_darkening_signal(time, fluxes, models, rho, rperiod, tpole, phi, beta, ldc,
                                  t0, p, a, i):
    """Draw the stellar disk and the residuals of a standard transit model fit for several orbits.

    Parameters
    ----------
    time : ndarray
        The mid-exposure times the fluxes were computed for.
    fluxes : dict
        Mapping from ``(band, l)`` to the modelled flux, where `band` names a passband and `l` is
        the projected spin-orbit angle in radians.
    models : dict
        Mapping from the passband names to the `GravityDarkenedModel` instances used. The stellar
        disk is drawn in the first passband.
    rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i
        The parameters the fluxes were computed with.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bands = list(models)
    ls = sorted({l for _, l in fluxes})
    colours = {'blue': 'C0', 'red': 'C3'}

    fig, axs = subplots(2, len(ls), figsize=(9, 5.4), height_ratios=(2, 3), sharey='row')
    for l, at, ab in zip(ls, axs[0], axs[1]):
        _draw_star(at, models[bands[0]], time, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l)
        at.set_title(f'$\\lambda$ = {degrees(l):.0f}°')
        for band in bands:
            flux = fluxes[band, l]
            residual = flux - _standard_fit(time, flux, p, a)
            ab.plot(24 * time, 1e6 * residual, c=colours.get(band), label=band)
        ab.axhline(0.0, c='0.8', lw=0.5, zorder=-1)
        ab.autoscale(axis='x', tight=True)
        setp(ab, xlabel='Time - $t_c$ [h]')
        _despine(ab)
    setp(axs[1, 0], ylabel='Residual of a standard fit [ppm]')
    axs[1, 0].legend(fontsize='small', frameon=False, loc='lower left')
    fig.tight_layout()
    return fig


def _example_model(**kwargs):
    return GravityDarkenedModel(filters=BoxcarFilter('blue', 400, 500), rstar=RSTAR, model='blackbody',
                                tmin=5000, tmax=9500, **kwargs)


def plot_surface_maps(res=300):
    """Draw the example star's effective gravity, temperature, and brightness in a blue passband.

    The gravity and temperature are computed here from the equations of the model page, and the
    brightness with the model's own code, so the last panel is the star as `evaluate` sees it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tm = _example_model()
    mstar, ostar, gpole, f, feff = map_osm(tm.rstar, RHO, RPERIOD, PHI)
    sphi, cphi = sin(PHI), cos(PHI)

    st, sx, sy = create_star_xy(res)
    x, y = sx * tm.rstar, sy * tm.rstar
    z = z_v(x, y, tm.rstar, f, sphi, cphi)

    # The surface point in the star's own frame, where the spin axis is the y axis.
    ys, zs = y * cphi + z * sphi, -y * sphi + z * cphi
    r, d = sqrt(x ** 2 + ys ** 2 + zs ** 2), hypot(x, zs)
    gn, gc = -G * mstar / r ** 2, ostar ** 2 * d
    with_axis = where(d > 0.0, d, 1.0)
    gx, gy, gz = gn * x / r + gc * x / with_axis, gn * ys / r, gn * zs / r + gc * zs / with_axis
    gravity = sqrt(gx ** 2 + gy ** 2 + gz ** 2) / gpole
    temperature = TPOLE * gravity ** BETA

    brightness = create_star_luminosity(res, sx, sy, mstar, tm.rstar, ostar, TPOLE, gpole, f, sphi, cphi, BETA,
                                        atleast_2d(LDC), tm._flux_table, tm._teff0, tm._dteff, tm.accurate_mu)[0]
    brightness = brightness.ravel() / nanmax(brightness)

    panels = (('Effective gravity', gravity, '$g / g_\\mathrm{pole}$'),
              ('Effective temperature', temperature, '$T_\\mathrm{eff}$ [K]'),
              ('Brightness, 400 to 500 nm', brightness, 'Relative to the brightest point'))
    fig, axs = subplots(1, 3, figsize=(9, 3.9))
    for ax, (title, image, label) in zip(axs, panels):
        image = image.reshape((res, res))
        im = ax.imshow(image, extent=(-1, 1, -1, 1), origin='lower', cmap='inferno', vmin=nanmin(image),
                       vmax=nanmax(image))
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04, shrink=0.8)
        cb.set_label(label, fontsize='small')
        cb.ax.tick_params(labelsize='small')
        cb.outline.set_visible(False)
        ax.set_title(title, fontsize='medium')
        setp(ax, xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_integration(pres=20, tres=60):
    """Draw the discretisation the light curve is computed from, for the default resolution.

    The panels show the positions along the planet's path where the mean brightness behind the
    planet is tabulated, the planet's sample points at one of them as the planet crosses the limb,
    and the resulting table.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tm = _example_model(pres=pres, tres=tres)
    mstar, ostar, gpole, f, feff = map_osm(tm.rstar, RHO, RPERIOD, PHI)
    sphi, cphi = sin(PHI), cos(PHI)
    sl, cl = sin(LAMBDA), cos(LAMBDA)
    ldc = atleast_2d(LDC)
    c = solve2d(0.0, P, A, arccos(B / A), 0.0, 0.0)

    xp, yp = create_planet_xy(pres)
    times, table = calculate_luminosity_interpolation_table(tres, K, xp, yp, sl, cl, c, mstar, tm.rstar, ostar,
                                                            TPOLE, gpole, f, sphi, cphi, BETA, ldc,
                                                            tm._flux_table, tm._teff0, tm._dteff, tm.accurate_mu)
    centres = array([xy_taylor_st(t, sl, cl, c) for t in times])

    res = 300
    st, sx, sy = create_star_xy(res)
    image = create_star_luminosity(res, sx, sy, mstar, tm.rstar, ostar, TPOLE, gpole, f, sphi, cphi, BETA, ldc,
                                   tm._flux_table, tm._teff0, tm._dteff, tm.accurate_mu)[0]
    scale = nanmax(image)
    table = table[0] / scale

    # The table node shown in the middle panel: the first one whose planet is half on the star.
    inside = (centres[:, 0] ** 2 + (centres[:, 1] / (1.0 - feff)) ** 2) < 1.0
    j = int(where(inside)[0][0])
    cx, cy = centres[j]

    fig, axs = subplots(1, 3, figsize=(9, 3.4), width_ratios=(1.0, 1.0, 1.25))

    ax = axs[0]
    ax.imshow(image / scale, extent=(-1, 1, -1, 1), origin='lower', cmap='inferno', vmin=0.4, vmax=1.0)
    ax.plot(centres[inside, 0], centres[inside, 1], '.', c='w', ms=2.5)
    ax.plot(centres[~inside, 0], centres[~inside, 1], '.', c='0.5', ms=2.5)
    ax.plot(cx, cy, 'o', mfc='none', mec='w', ms=11, mew=1.2)
    ax.set_title(f'{tres} table nodes along the path', fontsize='medium')
    setp(ax, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))

    # The close-up is rendered on its own grid so that the limb stays sharp.
    ax = axs[1]
    w = 1.6 * K
    zx, zy = meshgrid(linspace(cx - w, cx + w, res), linspace(cy - w, cy + w, res))
    zoom = luminosity_v(zx.ravel() * tm.rstar, zy.ravel() * tm.rstar, mstar, tm.rstar, ostar, TPOLE, gpole, f, sphi,
                        cphi, BETA, ldc, 0, tm._flux_table, tm._teff0, tm._dteff, tm.accurate_mu).reshape((res, res))
    ax.imshow(zoom / scale, extent=(cx - w, cx + w, cy - w, cy + w), origin='lower', cmap='inferno', vmin=0.4,
              vmax=1.0)
    px, py = cx + K * xp, cy + K * yp
    on_star = isfinite(z_v(px * tm.rstar, py * tm.rstar, tm.rstar, f, sphi, cphi))
    ax.plot(px[on_star], py[on_star], '.', c='w', ms=3)
    ax.plot(px[~on_star], py[~on_star], '.', c='0.6', ms=3)
    ax.add_patch(Ellipse((cx, cy), 2 * K, 2 * K, fc='none', ec='k', lw=1))
    ax.set_title(f'{xp.size} sample points on the planet', fontsize='medium')
    setp(ax, xlim=(cx - w, cx + w), ylim=(cy - w, cy + w))

    for ax in axs[:2]:
        ax.set_aspect('equal')
        ax.axis('off')

    ax = axs[2]
    ax.plot(24 * times, table, c='0.6', lw=1, zorder=1)
    ax.plot(24 * times, table, '.', c='k', ms=4, zorder=2)
    ax.plot(24 * times[j], table[j], 'o', mfc='none', mec='k', ms=11, mew=1.2)
    ax.set_title('Mean brightness behind the planet', fontsize='medium')
    setp(ax, xlabel='Time - $t_c$ [h]', ylabel='Relative to the brightest point')
    _despine(ax)
    fig.tight_layout()
    return fig


def plot_geometry(res=300):
    """Draw the geometry of the model: the star and the orbit on the sky, and the star from the side.

    The sky view shows the orbital azimuth angle `l` as the rotation of the planet's path about the
    centre of the disk, and the side view the stellar obliquity `phi` as the tilt of the spin axis
    out of the plane of the sky. Both use the example star and orbit of the page.

    Returns
    -------
    matplotlib.figure.Figure
    """
    tm = _example_model()
    mstar, ostar, gpole, f, feff = map_osm(tm.rstar, RHO, RPERIOD, PHI)
    sphi, cphi = sin(PHI), cos(PHI)
    ink, muted = '0.15', '0.45'

    fig, axs = subplots(1, 2, figsize=(9, 3.9), width_ratios=(2.8, 3.6))
    halo = [Stroke(linewidth=3, foreground='w'), Normal()]

    # The sky view
    # ------------
    ax = axs[0]
    st, sx, sy = create_star_xy(res)
    image = create_star_luminosity(res, sx, sy, mstar, tm.rstar, ostar, TPOLE, gpole, f, sphi, cphi, BETA,
                                   atleast_2d(LDC), tm._flux_table, tm._teff0, tm._dteff, tm.accurate_mu)[0]
    ax.imshow(image / nanmax(image), extent=(-1, 1, -1, 1), origin='lower', cmap='inferno', vmin=0.4, vmax=1.0)

    c = solve2d(0.0, P, A, arccos(B / A), 0.0, 0.0)
    time = linspace(-0.105, 0.105, 200)
    x0, y0 = xy_taylor_vt(time, 0.0, c)
    x1, y1 = xy_taylor_vt(time, LAMBDA, c)
    ax.plot(x0, y0, c=muted, lw=1, ls='--', path_effects=halo)
    ax.plot(x1, y1, c='k', lw=1.2, path_effects=halo)
    ax.annotate('', xy=(x1[-1], y1[-1]), xytext=(x1[-12], y1[-12]), arrowprops=dict(arrowstyle='-|>', color=ink, lw=1.2))
    ax.annotate('', xy=(x0[-1], y0[-1]), xytext=(x0[-12], y0[-12]), arrowprops=dict(arrowstyle='-|>', color=muted, lw=1))
    pj = abs(time - 0.03).argmin()
    ax.add_patch(Circle((x1[pj], y1[pj]), K, fc='k', ec='w', lw=0.6, zorder=5))

    # The two paths are the same line rotated about the disk centre, so they cross at an angle l.
    d = B / cos(0.5 * LAMBDA)
    qx, qy = d * sin(0.5 * LAMBDA), -d * cos(0.5 * LAMBDA)
    ax.add_patch(Arc((qx, qy), 0.5, 0.5, theta1=0.0, theta2=degrees(LAMBDA), color='w', lw=1.5, zorder=4))
    ax.text(qx + 0.33 * cos(0.5 * LAMBDA), qy + 0.33 * sin(0.5 * LAMBDA), '$l$', color='w', ha='center', va='center',
            fontsize='x-large')

    # The projected spin axis, drawn outside the disk so that it does not hide the pole.
    top = 1.0 - feff
    ax.plot([0, 0], [-1.3, -top], c=ink, lw=1)
    ax.annotate('', xy=(0, 1.3), xytext=(0, top), arrowprops=dict(arrowstyle='-|>', color=ink, lw=1))
    ax.text(0.05, 1.27, 'Projected spin axis', color=ink, fontsize='small', va='center')
    ax.text(x0[0], y0[0] - 0.06, '$l = 0$', color=muted, fontsize='small', va='top', ha='left')
    ax.set_title('From the observer', fontsize='medium')
    setp(ax, xlim=(-1.4, 1.4), ylim=(-1.4, 1.4))

    # The side view, with the observer on the right
    # ---------------------------------------------
    ax = axs[1]
    ax.add_patch(Ellipse((0, 0), 2 * (1.0 - f), 2.0, angle=-degrees(PHI), fc='0.9', ec='0.5', lw=0.8))
    ax.plot([0, 0], [-1.3, 1.3], c=muted, lw=1, ls='--')
    ax.text(-0.05, -1.27, 'Plane of the sky', color=muted, fontsize='small', va='center', ha='right')
    ax.plot([-1.3 * sphi, 1.0 * sphi], [-1.3 * cphi, 1.0 * cphi], c=ink, lw=1)
    ax.annotate('', xy=(1.3 * sphi, 1.3 * cphi), xytext=(1.0 * sphi, 1.0 * cphi),
                arrowprops=dict(arrowstyle='-|>', color=ink, lw=1))
    ax.text(1.3 * sphi + 0.06, 1.3 * cphi, 'Spin axis', color=ink, fontsize='small', va='center')
    ax.add_patch(Arc((0, 0), 1.2, 1.2, theta1=90.0 - degrees(PHI), theta2=90.0, color=ink, lw=1.2))
    ax.text(0.72 * sin(0.5 * PHI), 0.72 * cos(0.5 * PHI), '$\\phi$', color=ink, ha='center', va='center',
            fontsize='large')
    # The equator, which is the widest and coolest part of the star.
    ax.plot([-cphi, cphi], [sphi, -sphi], c='0.6', lw=0.8, ls=':')
    ax.annotate('', xy=(2.15, 0.0), xytext=(1.2, 0.0), arrowprops=dict(arrowstyle='-|>', color=muted, lw=1))
    ax.text(1.68, 0.08, 'To the observer', color=muted, fontsize='small', ha='center', va='bottom')
    ax.text(-0.8 * cphi, 0.8 * sphi + 0.07, 'Equator', color=muted, fontsize='small', ha='center', va='bottom',
            rotation=-degrees(PHI))
    ax.add_patch(Circle((1.05, -B * cos(LAMBDA)), K, fc='k', ec='none', zorder=5))
    ax.set_title('From the side', fontsize='medium')
    setp(ax, xlim=(-1.4, 2.2), ylim=(-1.4, 1.4))

    for ax in axs:
        ax.set_aspect('equal')
        ax.axis('off')
    # Both panels span the same height in data units, so the margins are set to make the height
    # the limiting dimension, which draws the two stars at the same scale.
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.05)
    return fig
