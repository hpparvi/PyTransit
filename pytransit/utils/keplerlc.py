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

"""Light curve container that splits the data into individual transits."""

import numpy as np

from numpy import inf, isfinite, abs, sum, unique, ones, compress, array, mean, round, int64, where, float64, full_like, \
    nan, argsort, isin

from pytransit.orbits import epoch


def fold_orbit_and_phase(time, period, origo, shift):
    """Orbit number and phase of each time.

    Parameters
    ----------
    time : ndarray
        Times to fold.
    period : float
        Orbital period.
    origo : float
        Time of zero phase.
    shift : float
        Phase shift added before the folding, in units of the period.

    Returns
    -------
    orbit : ndarray
        Integer orbit number of each time.
    phase : ndarray
        Phase of each time in the units of the period, between -period/2 and period/2.
    """
    phase  = ((time - origo) / period + shift)
    orbit  = (phase // 1).astype(int64)
    phase  = (phase % 1. - 0.5) * period
    return  orbit, phase

class KeplerLC(object):
    """Light curve split into individual transits.

    A convenience class for working with Kepler and TESS light curves. Keeps the finite points
    within `d_baseline` / 2 of each transit centre, sorted in time, and drops the transits with
    too few points, no out-of-transit points to normalise the flux with, or too large a flux range.
    The remaining transits are numbered consecutively from zero.
    """

    def __init__(self, time, flux, quarter, zero_epoch, period, d_transit, d_baseline, error=None, **kwargs):
        """
        Parameters
        ----------
        time : ndarray
            Times [d].
        flux : ndarray
            Fluxes.
        quarter : ndarray
            Kepler quarter or TESS sector of each point.
        zero_epoch : float
            Transit centre time of the zero epoch [d].
        period : float
            Orbital period [d].
        d_transit : float
            Approximate transit duration [d], used to separate the in-transit and out-of-transit
            points.
        d_baseline : float
            Total duration of the data to include around each transit centre [d].
        error : ndarray, optional
            Flux uncertainties. Default to NaN.
        max_ptp : float, optional
            Maximum peak-to-peak range of the normalised flux in a transit. Defaults to infinity.
        min_pts : int, optional
            Minimum number of points in a transit. Defaults to 5.
        """
        self.t0   = zero_epoch
        self.p    = period
        self.max_ptp = kwargs.get('max_ptp', inf)
        self.min_pts = kwargs.get('min_pts', 5)

        time = array(time, float64)
        flux = array(flux, float64)
        quarter = array(quarter)
        error = full_like(time, nan) if error is None else array(error, float64)

        # The per-transit and per-quarter slices assume that each transit and quarter is a
        # contiguous block, so the data are sorted in time.
        m = isfinite(time) & isfinite(flux)
        sids = argsort(time[m], kind='stable')
        time, flux, quarter, error = time[m][sids], flux[m][sids], quarter[m][sids], error[m][sids]

        orbit, phase = fold_orbit_and_phase(time, period, zero_epoch, 0.5)
        msk_inc = abs(phase) < 0.5*d_baseline

        self.time = time
        self.phase = phase
        self.flux = flux
        self.error = error
        self.qidarr = quarter
        self.tidarr = orbit
        self.msk_oot = abs(phase) > 0.5*d_transit

        # Remove orbits with too few datapoints or no out-of-transit points
        # -----------------------------------------------------------------
        for tid in unique(orbit[msk_inc]):
            m = msk_inc & (orbit == tid)
            if m.sum() < self.min_pts or not (m & self.msk_oot).any():
                msk_inc[orbit == tid] = False
        self._compress_data(msk_inc)
        self._compute_indices()

        # Remove orbits with too big ptp range
        # ------------------------------------
        keep = array([np.ptp(f) <= self.max_ptp for f in self.normalized_flux_per_transit], bool)
        self._compress_data(keep[self.tidarr])
        self._compute_indices()

    def _compress_data(self, mask):
        """Keep only the points where `mask` is True."""
        self.time    = compress(mask, self.time)
        self.phase   = compress(mask, self.phase)
        self.flux    = compress(mask, self.flux)
        self.error   = compress(mask, self.error)
        self.qidarr  = compress(mask, self.qidarr)
        self.tidarr  = compress(mask, self.tidarr)
        self.msk_oot = compress(mask, self.msk_oot)

    def _compute_indices(self): 
        """Update the quarter and transit indices, and renumber the transits consecutively."""
        self.qids = unique(self.qidarr)
        self.tids = unique(self.tidarr)
        self.nt   = len(self.tids)
        self.npt  = self.time.size

        self.qslices = [slice(*where(self.qidarr==qid)[0][[0,-1]]+[0,1]) for qid in self.qids]
        self.qsldict = {qid : slice(*where(self.qidarr==qid)[0][[0,-1]]+[0,1]) for qid in self.qids}

        self.tidarr = unique(self.tidarr, return_inverse=True)[1]
        self.tids = unique(self.tidarr)
        self.tslices = [slice(*where(self.tidarr==tid)[0][[0,-1]]+[0,1]) for tid in self.tids]
        self.orbit_n = array([epoch(t.mean(), self.t0, self.p) for t in self.time_per_transit])

    def get_transit(self, tid, normalize=False, mask_transit=False):
        """Time and flux of a single transit.

        Parameters
        ----------
        tid : int
            Transit index.
        normalize : bool
            Return the flux normalised by the out-of-transit median if True.
        mask_transit : bool
            Not used.

        Returns
        -------
        tuple
            The time and flux arrays of the transit.
        """
        mask = self.tidarr==tid
        return self.time[mask], self.normalized_flux[mask] if normalize else self.flux[mask]

    def remove_common_orbits(self, lc2):
        """Remove the transits whose orbit numbers are also found in another light curve.

        Parameters
        ----------
        lc2 : KeplerLC
            The light curve to compare against.
        """
        keep = ~isin(self.orbit_n, lc2.orbit_n)
        self._compress_data(keep[self.tidarr])
        self._compute_indices()

    @property
    def flux_per_transit(self):
        """Flux of each transit as a list of arrays."""
        return [self.flux[sl] for sl in self.tslices]
    
    @property
    def normalized_flux_per_transit(self):
        """Normalised flux of each transit as a list of arrays."""
        nf = self.normalized_flux
        return [nf[sl] for sl in self.tslices]

    @property
    def normalized_oot_flux_per_transit(self):
        """Normalised out-of-transit flux of each transit as a list of arrays."""
        nf = self.normalized_flux
        return [nf[sl][self.oot_mask[sl]] for sl in self.tslices]

    @property
    def time_per_transit(self):
        """Time of each transit as a list of arrays."""
        return [self.time[sl] for sl in self.tslices]

    @property
    def oot_time_per_transit(self):
        """Out-of-transit time of each transit as a list of arrays."""
        return [self.time[sl][self.oot_mask[sl]] for sl in self.tslices]

    @property
    def error_per_transit(self):
        """Flux uncertainties of each transit as a list of arrays."""
        return [self.error[sl] for sl in self.tslices]

    @property
    def normalized_error(self):
        """Flux uncertainties divided by the out-of-transit flux median of each transit."""
        return self.error / self.flux_baseline

    @property
    def normalized_error_per_transit(self):
        """Normalised flux uncertainties of each transit as a list of arrays."""
        nf = self.normalized_error
        return [nf[sl] for sl in self.tslices]

    @property
    def quarter_per_transit(self):
        """Quarter or sector indices of each transit as a list of arrays."""
        return [self.qidarr[sl] for sl in self.tslices]

    @property
    def oot_mask(self):
        """Boolean mask that is True for the out-of-transit points."""
        return self.msk_oot

    @property
    def normalized_flux(self):
        """Flux divided by the out-of-transit flux median of each transit."""
        return self.flux / self.flux_baseline

    @property
    def flux_baseline(self):
        """Out-of-transit flux median of the transit each point belongs to."""
        bl = np.zeros_like(self.flux)
        for tid in self.tids:
            mask_tid = tid==self.tidarr
            mask_nrm = mask_tid & self.msk_oot
            bl[mask_tid] = np.median(self.flux[mask_nrm])
        return bl

    @property
    def oot_flux_std(self):
        """Standard deviation of the normalised out-of-transit flux."""
        return self.normalized_flux[self.msk_oot].std()
