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

import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sb
from matplotlib.gridspec import GridSpec

from numpy import sqrt, percentile, ceil, linspace, concatenate, histogram

color = sb.color_palette()[0]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap = sb.blend_palette(colors, as_cmap=True)

color = sb.color_palette()[1]
color_rgb = mpl.colors.colorConverter.to_rgb(color)
colors = [sb.utils.set_hls_values(color_rgb, l=l) for l in linspace(1, 0, 12)]
cmap2 = sb.blend_palette(colors, as_cmap=True)

c_ob = "#002147"  # Oxford blue
c_bo = "#CC5500"  # Burnt orange

def marginal_radius_ratio_plot(df, bins: int = 40, klim: tuple = None, figsize: tuple = (6, 5)):
    if klim is None:
        klim = percentile(concatenate([df.k_app.values, df.k_true.values]), [0.0, 99.9])

    v1, e = histogram(df.k_app, range=klim, bins=bins, density=True)
    v2, e = histogram(df.k_true, range=klim, bins=bins, density=True)

    fig, ax = pl.subplots(figsize=figsize, constrained_layout=True)
    ax.fill_betweenx(e[:-1], -10 - v1, -10, step='post', linewidth=1, edgecolor='k')
    ax.fill_betweenx(e[:-1], 10 + v2, 10, step='post', linewidth=1, edgecolor='k')

    ax.text(-0.5 * v1.max(), klim[1], 'Apparent radius ratio', ha='center', va='top')
    ax.text(0.5 * v2.max(), klim[1], 'True radius ratio', ha='center', va='top')

    pl.setp(ax, xlabel='Posterior density', ylabel='Radius ratio', xticks=[], ylim=klim)
    sb.despine(fig, offset=5)
    return fig


def _jplot(hte, cte, cnr, imp, rho, fw=10, nb=30, gs=25, ylabel='Contamination in $i\'$', **kwargs):
    htelim = kwargs.get('htelim', (2000, 8000))
    ctelim = kwargs.get('ctelim', (4000, 12000))
    blim = kwargs.get('blim', (0, 1))
    rlim = kwargs.get('rlim', (0, 15))
    clim = kwargs.get('clim', (0, 1))

    fig = pl.figure(figsize=(fw, fw / 4))
    gs_tt = GridSpec(2, 1, bottom=0.2, top=1, left=0.1, right=0.3, hspace=0, wspace=0, height_ratios=[0.15, 0.85], figure=fig)
    gs_ct = GridSpec(2, 5, bottom=0.2, top=1, left=0.38, right=1, hspace=0.05, wspace=0.05,
                     height_ratios=[0.15, 0.85],
                     width_ratios=[1, 1, 1, 1, 0.2], figure=fig)

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

    ax_thm.hist(hte, bins=nb, alpha=0.5, range=htelim, histtype='stepfilled')
    ax_ctm.hist(cte, bins=nb, alpha=0.5, range=ctelim, histtype='stepfilled')
    ax_bm.hist(imp, bins=nb, alpha=0.5, range=blim, histtype='stepfilled')
    ax_rm.hist(rho, bins=nb, alpha=0.5, range=rlim, histtype='stepfilled')
    ax_cnm.hist(cnr, bins=nb, alpha=0.5, range=clim, histtype='stepfilled', orientation='horizontal')

    pl.setp(ax_tt, xlabel='Host $T_\mathrm{Eff}$', ylabel='Contaminant $T_\mathrm{Eff}$')
    pl.setp(ax_chj, xlabel='Host $T_\mathrm{Eff}$', ylabel=ylabel)
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


def joint_radius_ratio_plot(df, fw=10, nb=30, gs=25, **kwargs):
    return _jplot(df.teff_h, df.teff_c, df.k_true, df.b, df.rho, fw, nb, gs, ylabel='True radius ratio', **kwargs)[0]

def joint_contamination_plot(df, fw=10, nb=30, gs=25, **kwargs):
    return _jplot(df.teff_h, df.teff_c, df.cnt, df.b, df.rho, fw, nb, gs, **kwargs)[0]



