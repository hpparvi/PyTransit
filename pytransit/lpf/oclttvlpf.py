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


from matplotlib.pyplot import subplots, setp
from numpy import sqrt, array, inf, int, s_, percentile, median, mean, round, zeros, atleast_2d, ceil, poly1d, polyfit
from numpy.random.mtrand import permutation
from uncertainties import ufloat, UFloat

try:
    import seaborn as sb
    with_seaborn = True
except ImportError:
    with_seaborn = False

from ..param.parameter import GParameter, LParameter
from ..param.parameter import UniformPrior as UP, NormalPrior as NP
from ..orbits.orbits_py import as_from_rhop, i_from_ba
from .ocllpf import OCLBaseLPF


def plot_estimates(x, p, ax, bwidth=0.8):
    ax.bar(x, p[4, :] - p[3, :], bwidth, p[3, :], alpha=0.25, fc='b')
    ax.bar(x, p[2, :] - p[1, :], bwidth, p[1, :], alpha=0.25, fc='b')
    [ax.plot((xx - 0.47 * bwidth, xx + 0.47 * bwidth), (pp[[0, 0]]), 'k') for xx, pp in zip(x, p.T)]


class OCLTTVLPF(OCLBaseLPF):
    def __init__(self, target: str, zero_epoch: float, period: float, passbands: list,
                 times: list = None, fluxes: list = None, errors: list = None, pbids: list = None, wnids: list = None,
                 nsamples: list = None, exptimes: list = None, cl_ctx=None, cl_queue=None):

        self.zero_epoch = zero_epoch if isinstance(zero_epoch, UFloat) else ufloat(zero_epoch, 1e-5)
        self.period = period if isinstance(period, UFloat) else ufloat(period, 1e-7)
        self.epoch = round((array([t.mean() for t in times]) - self.zero_epoch.n) / self.period.n).astype(int)

        super().__init__(target, passbands, times, fluxes, errors, pbids, nsamples=nsamples, exptimes=exptimes,
                         wnids=wnids, cl_ctx=cl_ctx, cl_queue=cl_queue)


    def _init_p_orbit(self):
        """Orbit parameter initialisation for a TTV model.

        The orbit part of the parameter vector will be [rho b tc_0 tc_1 ... tc_n].

        """

        # Basic orbital parameters
        # ------------------------
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
                  GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]

        # Transit centers
        # ---------------
        self.epoch = round((array([t.mean() for t in self.times]) - self.zero_epoch.n) / self.period.n).astype(int)

        for e in self.epoch:
            tc = self.zero_epoch + e*self.period
            porbit.append(GParameter(f'tc_{e:d}', f'transit_centre_{e:d}', 'd', NP(tc.n, tc.s), (-inf, inf)))

        self.ps.add_global_block('orbit', porbit)
        self._start_tc = 2
        self._sl_tc = s_[self._start_tc:self._start_tc + self.nlc]

    def optimize_times(self, window):
        times, fluxes, pbids = [], [], []
        tcp = self.ps[self._sl_tc]
        for i in range(self.nlc):
            tc = tcp[i].prior.mean
            mask = abs(self.times[i] - tc) < 0.5*window/24.
            times.append(self.times[i][mask])
            fluxes.append(self.fluxes[i][mask])
        self._init_data(times, fluxes, self.pbids)

    def transit_model(self, pvp, copy=False):
        pvp = atleast_2d(pvp)
        pvp_cl = zeros([pvp.shape[0], 6 + self.nlc], "f")  # k tc p a i e w
        uv = zeros([pvp.shape[0], 2], "f")
        tc_end = 1 + self.nlc
        pvp_cl[:, 0:1] = sqrt(pvp[:, self._pid_k2])  # Radius ratio
        pvp_cl[:, 1:tc_end] = pvp[:, self._sl_tc]    # Transit centre and orbital period
        pvp_cl[:, tc_end + 0] = self.period.n
        pvp_cl[:, tc_end + 1] = a = as_from_rhop(pvp[:, 0], self.period.n)
        pvp_cl[:, tc_end + 2] = i_from_ba(pvp[:, 1], a)
        a, b = sqrt(pvp[:, self._sl_ld][:, 0]), 2. * pvp[:, self._sl_ld][:, 1]
        uv[:, 0] = a * b
        uv[:, 1] = a * (1. - b)
        flux = self.tm.evaluate_pv_ttv(pvp_cl, uv, copy=copy)
        return flux.T if copy else None

    def posterior_period(self, burn: int = 0, thin: int = 1) -> float:
        df = self.posterior_samples(burn, thin, False)
        tccols = [c for c in df.columns if 'tc' in c]
        tcs = median(df[tccols], 0)
        return mean((tcs[1:] - tcs[0]) / (self.epoch[1:] - self.epoch[0]))

    def plot_ttvs(self, burn=0, thin=1, axs=None, figsize=None, bwidth=0.8, fmt='h', windows=None, sigma=inf, nsamples=1000):
        assert fmt in ('d', 'h', 'min')
        multiplier = {'d': 1, 'h': 24, 'min': 1440}
        ncol = 1 if windows is None else len(windows)
        fig, axs = (None, axs) if axs is not None else subplots(1, ncol, figsize=figsize, sharey=True)
        df = self.posterior_samples(burn, thin, derived_parameters=False)
        tccols = [c for c in df.columns if 'tc' in c]
        df = df[tccols]
        s = df.std()
        m = (s < median(s) + sigma * s.std()).values
        df = df.iloc[:, m]
        epochs = self.epoch[m]

        samples = []
        for tcs in permutation(df.values)[:nsamples]:
            samples.append(tcs - poly1d(polyfit(epochs, tcs, 1))(epochs))
        samples = array(samples)
        p = multiplier[fmt] * percentile(samples, [50, 16, 84, 0.5, 99.5], 0)
        setp(axs, ylabel='Transit center - linear prediction [{}]'.format(fmt), xlabel='Transit number')
        if windows is None:
            plot_estimates(epochs, p, axs, bwidth)
            if with_seaborn:
                sb.despine(ax=axs, offset=15)
        else:
            setp(axs[1:], ylabel='')
            for ax, w in zip(axs, windows):
                m = (epochs > w[0]) & (epochs < w[1])
                plot_estimates(epochs[m], p[:, m], ax, bwidth)
                setp(ax, xlim=w)
                if with_seaborn:
                    sb.despine(ax=ax, offset=15)
        if fig:
            fig.tight_layout()
        return axs

    def plot_transits(self, ncols=4, figsize=(13, 11), remove=(), ylim=None):
        nt = len(self.times)
        nrows = int(ceil(nt / ncols))
        fig, axs = subplots(nrows, ncols, figsize=figsize, sharey='all', gridspec_kw=dict(hspace=0.01, wspace=0.01))

        if self.de is not None:
            pv = self.de.minimum_location
            fmodel = self.flux_model(pv).ravel()
        else:
            pv = None
            fmodel = None

        for i in range(nt):
            ax = axs.flat[i]

            # The light curve itself
            # ----------------------
            lc = '0.75' if i in remove else 'k'
            ax.plot(self.times[i], self.fluxes[i], '.', c=lc)

            # The fitted model, if available
            # ------------------------------
            if fmodel is not None:
                m = self.lcids == i
                ax.plot(self.times[i], fmodel[m], 'w', lw=4)
                ax.plot(self.times[i], fmodel[m], 'k', lw=1)

            # Transit centre prior
            # --------------------
            p = self.ps[self._start_tc + i].prior
            [ax.axvspan(p.mean - s * p.std, p.mean + s * p.std, alpha=0.15) for s in (3, 2, 1)]

            if self.de is not None:
                ax.axvline(pv[self._start_tc + i], c='k', zorder=-2)

            # The transit index
            # -----------------
            ax.text(0.05, 0.95, self.epoch[i], ha='left', va='top', transform=ax.transAxes)
            ax.text(0.05, 0.05, "({})".format(i), ha='left', va='bottom', transform=ax.transAxes, size='small')
            ax.set_xlim(self.times[i][[0, -1]])

        if ylim is not None:
            setp(axs, ylim=ylim)

        for ax in axs.flat[nt:]:
            ax.set_visible(False)

        setp(axs, xticks=[])
        fig.tight_layout()
