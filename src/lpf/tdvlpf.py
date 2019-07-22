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
from numpy import pi, sign, cos, sqrt, sin, array, arccos, inf, round, int, s_, percentile, concatenate, median, mean, \
    arange

import scipy.ndimage as ndi

from numba import njit, prange
from .ttvlpf import TTVLPF
from ..models.transitmodel import TransitModel
from ..param.parameter import ParameterSet, PParameter, GParameter
from ..param.parameter import UniformPrior as U, NormalPrior as N, GammaPrior as GM
from ..orbits.orbits_py import duration_eccentric, as_from_rhop, p_from_dkaiews

try:
    import seaborn as sb
    with_seaborn = True
except ImportError:
    with_seaborn = False

@njit("f8[:](f8[:], f8[:], f8, f8, f8[:], i8[:])", cache=False, parallel=False)
def z_circular_ttv(t, p, a, i, tc, tcid):
    cosph = cos(2*pi * (t - tc[tcid]) / p[tcid])
    z = sign(cosph) * a * sqrt(1.0 - cosph * cosph * sin(i) ** 2)
    return z


class TDVLPF(TTVLPF):
    """Log posterior function for TDV estimation.

    A log posterior function for TDV estimation. Each light curve represents a single transit, and
    is given a separate free transit centre parameter. The average orbital period and (one) transit
    zero epoch are assumed as known.

    Notes: The number of parameters can grow large with Kepler short-period planets.

    """
    def __init__(self, target: str, zero_epoch: float, period: float, tc_sigma: float, duration_prior: tuple,
                 passbands: list, times: list = None, fluxes: list = None, pbids: list = None, tm: TransitModel = None,
                 nsamples: int = 1, exptime: float = 0.020433598):
        self.zero_epoch = zero_epoch
        self.period = period
        self.tc_sigma = tc_sigma
        self.t14_prior = duration_prior
        super().__init__(target, zero_epoch, period, tc_sigma, passbands, times, fluxes, pbids, tm, nsamples, exptime)

    def _init_p_orbit(self):
        """Orbit parameter initialisation for a TTV model.
           """

        # Basic orbital parameters
        # ------------------------
        porbit = [GParameter('rho', 'stellar_density', 'g/cm^3', U(0.1, 25.0), (0, inf)),
                  GParameter('b', 'impact_parameter', 'R_s', U(0.0, 1.0), (0, 1))]

        # Transit centers
        # ---------------

        def create_tc_prior(t, f, p=5):
            m = f > percentile(f, p)
            m = ~ndi.binary_erosion(m, iterations=6, border_value=1)
            return N(t[m].mean(), 0.25 * t[m].ptp())

        self.tnumber = round((array([t.mean() for t in self.times]) - self.zero_epoch) / self.period).astype(int)
        for t, f, tn in zip(self.times, self.fluxes, self.tnumber):
            prior = create_tc_prior(t, f)
            porbit.append(GParameter(f'tc_{tn:d}', f'transit_centre_{tn:d}', 'd', prior, (-inf, inf)))

        # Transit durations
        # -----------------
        for tn in self.tnumber:
            porbit.append(GParameter(f't14_{tn:d}', f'duration_{tn:d}', 'd', N(*self.t14_prior), (0, inf)))

        self.ps.add_global_block('orbit', porbit)
        self._start_tc = 2
        self._sl_tc = s_[self._start_tc:self._start_tc + self.nlc]
        self._start_d = 2 + self.nlc
        self._sl_d = s_[self._start_d:self._start_d + self.nlc]


    def _compute_z(self, pv):
        a = as_from_rhop(pv[0], self.period)
        if a < 1.:
            return None
        else:
            i = arccos(pv[1] / a)
            tc = pv[self._sl_tc]
            p = p_from_dkaiews(pv[self._sl_d], sqrt(pv[self._pid_k2]), a, i, 0., 0., 1)
            return z_circular_ttv(self.timea, p, a, i, tc, self.lcida)

    def plot_tdvs(self, burn=0, thin=1, ax=None, figsize=None, bwidth=0.8):
        fig, ax = (None, ax) if ax is not None else subplots(figsize=figsize)
        df = self.posterior_samples(burn, thin)
        dcols = [c for c in df.columns if 't14_' in c]
        p = 24 * df[dcols].quantile([0.50, 0.16, 0.84, 0.005, 0.995]).values
        ax.bar(self.tnumber, p[4, :] - p[3, :], bwidth, p[3, :], alpha=0.25, fc='b')
        ax.bar(self.tnumber, p[2, :] - p[1, :], bwidth, p[1, :], alpha=0.25, fc='b')
        [ax.plot((xx - 0.47 * bwidth, xx + 0.47 * bwidth), (pp[[0, 0]]), 'k') for xx, pp in zip(self.tnumber, p.T)]
        setp(ax, ylabel='Transit duration [h]', xlabel='Transit number')
        fig.tight_layout()
        if with_seaborn:
            sb.despine(ax=ax, offset=15)
        return ax