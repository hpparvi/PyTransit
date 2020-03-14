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

import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sb
from matplotlib.gridspec import GridSpec

from numpy import linspace, median, argmin, percentile

color = sb.color_palette()[0]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap = sb.blend_palette(colors, as_cmap=True)

color = sb.color_palette()[1]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap2 = sb.blend_palette(colors, as_cmap=True)

## Color definitions
## -----------------
c_ob = "#002147"  # Oxford blue
c_bo = "#CC5500"  # Burnt orange


def plot_kdist(samples, side: str = 'right', ax=None, clip: tuple = (0, 1), percentiles: tuple = (16, 84),
               offset: float = 0.02, bw=0.005, gridsize: int = 200):
    assert side in ('left', 'right')
    sign = 1 if side == 'right' else -1
    fig, ax = (None, ax) if ax is not None else pl.subplots()
    p = sb.kdeplot(samples, kernel='cos', bw=bw, gridsize=gridsize, cut=0, clip=clip, vertical=True, ax=ax, color='k',
                   legend=False)
    xd, yd = p.lines[-1].get_xdata(), p.lines[-1].get_ydata()
    m = median(samples)
    my = xd[argmin(abs(yd - m))] / xd.max()
    p.lines[-1].set_xdata(sign * (offset + xd / xd.max()))
    ax.plot((sign * offset, sign * offset), clip, 'k')
    ax.plot((sign * offset, sign * my), (m, m), 'k')
    p = percentile(samples, percentiles)
    mask = (yd > p[0]) & (yd < p[1])
    ax.fill_betweenx(yd[mask], sign * (offset + xd[mask] / xd.max()),  sign*offset, alpha=0.25)
    return fig

def plot_two_sided_kde(left, right, clip: tuple = (0, 1), percentiles: tuple = (16, 84),
                       offset: float = 0.02, bw=0.005, gridsize: int = 200, ax = None):
    fig, ax = (None, ax) if ax is not None else pl.subplots()
    plot_kdist(left, side='left', clip=clip, percentiles=percentiles, offset=offset, bw=bw, gridsize=gridsize, ax=ax)
    plot_kdist(right, side='right', clip=clip, percentiles=percentiles, offset=offset, bw=bw, gridsize=gridsize, ax=ax)
    pl.setp(ax, xlim=(-1.1, 1.1))
    return fig

def _jplot(hte, cte, cnr, imp, rho, fw=10, nb=30, gs=25, simulation=False, **kwargs):
    htelim = kwargs.get('htelim', (2000, 8000))
    ctelim = kwargs.get('ctelim', (4000, 12000))
    blim = kwargs.get('blim', (0, 1))
    rlim = kwargs.get('rlim', (0, 15))
    clim = kwargs.get('clim', (0, 1))

    fig = pl.figure(figsize=(fw, fw / 4))
    gs_tt = GridSpec(2, 1, bottom=0.2, top=1, left=0.1, right=0.3, hspace=0, wspace=0, height_ratios=[0.15, 0.85])
    gs_ct = GridSpec(2, 5, bottom=0.2, top=1, left=0.37, right=1, hspace=0.05, wspace=0.05,
                     height_ratios=[0.15, 0.85],
                     width_ratios=[1, 1, 1, 1, 0.2])

    ax_tt = pl.subplot(gs_tt[1, 0])
    ax_chj = pl.subplot(gs_ct[1, 0])
    ax_ccj = pl.subplot(gs_ct[1, 1])
    ax_cbj = pl.subplot(gs_ct[1, 2])
    ax_crj = pl.subplot(gs_ct[1, 3])
    ax_thm = pl.subplot(gs_ct[0, 0])
    ax_ctm = pl.subplot(gs_ct[0, 1])
    ax_bm = pl.subplot(gs_ct[0, 2])
    ax_rm = pl.subplot(gs_ct[0, 3])
    ax_cnm = pl.subplot(gs_ct[1, 4])

    ax_tt.hexbin(hte, cte, gridsize=gs, cmap=cmap, extent=(htelim[0], htelim[1], ctelim[0], ctelim[1]))
    ax_chj.hexbin(hte, cnr, gridsize=gs, cmap=cmap, extent=(htelim[0], htelim[1], clim[0], clim[1]))
    ax_ccj.hexbin(cte, cnr, gridsize=gs, cmap=cmap, extent=(ctelim[0], ctelim[1], clim[0], clim[1]))
    ax_cbj.hexbin(imp, cnr, gridsize=gs, cmap=cmap, extent=(blim[0], blim[1], clim[0], clim[1]))
    ax_crj.hexbin(rho, cnr, gridsize=gs, cmap=cmap, extent=(rlim[0], rlim[1], clim[0], clim[1]))

    ax_thm.hist(hte, bins=nb, alpha=0.5, range=htelim)
    ax_ctm.hist(cte, bins=nb, alpha=0.5, range=ctelim)
    ax_bm.hist(imp, bins=nb, alpha=0.5, range=blim)
    ax_rm.hist(rho, bins=nb, alpha=0.5, range=rlim)
    ax_cnm.hist(cnr, bins=nb, alpha=0.5, range=clim, orientation='horizontal')

    pl.setp(ax_tt, xlabel='Host $T_\mathrm{Eff}$', ylabel='Contaminant $T_\mathrm{Eff}$')
    pl.setp(ax_chj, xlabel='Host $T_\mathrm{Eff}$', ylabel='Contamination in $i\'$')
    pl.setp(ax_ccj, xlabel='Contaminant $T_\mathrm{Eff}$')
    pl.setp(ax_cbj, xlabel='Impact parameter')
    pl.setp(ax_crj, xlabel='Stellar density')

    pl.setp(ax_thm, xlim=ax_chj.get_xlim())
    pl.setp(ax_ctm, xlim=ax_ccj.get_xlim())
    pl.setp(ax_bm, xlim=ax_cbj.get_xlim())
    pl.setp([ax_ccj, ax_cnm], ylim=ax_chj.get_ylim())
    pl.setp([ax_chj, ax_ccj, ax_cbj, ax_crj, ax_cnm], ylim=clim)

    pl.setp([ax_thm, ax_ctm, ax_cnm, ax_bm, ax_rm], yticks=[], xticks=[])
    pl.setp(ax_ccj.get_yticklabels(), visible=False)
    pl.setp(ax_cbj.get_yticklabels(), visible=False)
    pl.setp(ax_crj.get_yticklabels(), visible=False)
    [sb.despine(ax=ax, left=True, offset=0.1) for ax in [ax_thm, ax_ctm, ax_bm, ax_rm]]
    [sb.despine(ax=ax) for ax in [ax_chj, ax_ccj, ax_cbj, ax_crj]]
    sb.despine(ax=ax_cnm, bottom=True)
    return fig, ax_tt, ax_chj, ax_cbj, ax_ccj, ax_crj

def joint_marginal_plot(df, fw=10, nb=30, gs=25, **kwargs):
    return _jplot(df.teff_h, df.teff_c, df.cnt, df.b, df.rho, fw, nb, gs, **kwargs)[0]
