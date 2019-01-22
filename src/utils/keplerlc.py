import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as pl

from .utils.misc import fold

def fold_orbit_and_phase(time, origo, period, shift):
    phase  = ((time - origo)/period + shift)
    orbit  = (phase-0.5) // 1
    orbit  = (orbit - np.nanmin(orbit)).astype(np.uint)
    phase  = phase % 1.
    return  orbit, phase

class KeplerLC(object):
    def __init__(self, time, flux, zero_epoch, period, d_transit, d_baseline, quarter=None, **kwargs):
        """
        Kepler light curve.

        Parameters
        ----------

          time       : BJD array      [d]
          flux       : Flux array     [-]
          zero_epoch : zero epoch     [d]
          period     : orbital period [d]
          d_transit  : transit duration (approximate) [d]
          d_baseline : total duration to include around each transit
          quarter    : quarter array

          max_ptp    : maximum point-to-point span per transit
          min_pts    : minimum number of points per transit
        """
        self.t0   = zero_epoch
        self.p    = period
        self.max_ptp = kwargs.get('max_ptp', 1e5)
        self.min_pts = kwargs.get('min_pts', 5)

        orbit, phase = fold_orbit_and_phase(time, period, zero_epoch, 0.5)
        msk_phase = np.abs(phase-0.5)*period < 0.5*d_baseline          ## phase inclusion mask
        msk_oot   = np.abs(phase-0.5)*period > 0.5*d_transit           ## out-of-transit mask
        msk_inc   = np.isfinite(time) & np.isfinite(flux) & msk_phase  ## final data inclusion mask

        self.time   = np.array(time)
        self.flux   = np.array(flux)
        self.qidarr = np.array(quarter)                                ## quarter indices
        self.tidarr, nt = orbit, orbit[-1]                             ## transit indices
        self.msk_oot = msk_oot

        ## Remove orbits with too few datapoints
        ## -------------------------------------
        npts = np.array([np.sum((self.tidarr == tid) & msk_inc) for tid in np.unique(self.tidarr)])
        for tid,npt in enumerate(npts):
            if npt < self.min_pts:
                msk_inc[self.tidarr==tid] = 0
        self._compress_data(msk_inc)
        self._compute_indices()

        ## Remove orbits with too big ptp range
        ## ------------------------------------
        msk_inc = np.ones(self.npt, np.bool)
        for tid,ptp in enumerate(list(map(np.ptp, self.normalized_flux_per_transit))):
            if ptp > self.max_ptp:
                msk_inc[self.tidarr==tid] = 0
        self._compress_data(msk_inc)
        self._compute_indices()
 
 
    def _compress_data(self, mask):
        self.time    = np.compress(mask, self.time)
        self.flux    = np.compress(mask, self.flux)
        self.qidarr  = np.compress(mask, self.qidarr)
        self.tidarr  = np.compress(mask, self.tidarr)
        self.msk_oot = np.compress(mask, self.msk_oot)


    def _compute_indices(self): 
        self.qids = np.unique(self.qidarr)
        self.tids = np.unique(self.tidarr)
        self.nt   = len(self.tids)
        self.npt  = self.time.size

        self.qslices = [slice(*np.where(self.qidarr==qid)[0][[0,-1]]+[0,1]) for qid in self.qids]
        self.qsldict = {qid : slice(*np.where(self.qidarr==qid)[0][[0,-1]]+[0,1]) for qid in self.qids}

        for i,tid in enumerate(self.tids):
            self.tidarr[self.tidarr==tid] = i
        self.tids = np.unique(self.tidarr)
        self.tslices = [slice(*np.where(self.tidarr==tid)[0][[0,-1]]+[0,1]) for tid in self.tids]

        self.orbit_n = np.round((np.array(list(map(np.mean, self.time_per_transit))) - self.t0) / self.p).astype(np.int)


    def get_transit(self, tid, normalize=False):
        mask = self.tidarr==tid
        return self.time[mask], self.normalized_flux[mask] if normalize else self.flux[mask]

    def remove_common_orbits(self, lc2):
        is_unique = ~np.in1d(self.orbit_n, lc2.orbit_n)
        mask = np.ones(self.npt, np.bool)
        for tid,include in enumerate(is_unique):
            mask[self.tslices[tid]] = include
        self._compress_data(mask)
        self._compute_indices()

    @property
    def flux_per_transit(self):
        return [self.flux[sl] for sl in self.tslices]
    
    @property
    def normalized_flux_per_transit(self):
        nf = self.normalized_flux
        return [nf[sl] for sl in self.tslices]

    @property
    def time_per_transit(self):
        return [self.time[sl] for sl in self.tslices]

    @property
    def quarter_per_transit(self):
        return [self.qidarr[sl] for sl in self.tslices]

    @property
    def oot_mask(self):
        return self.msk_oot

    @property
    def normalized_flux(self):
        return self.flux / self.flux_baseline

    @property
    def flux_baseline(self):
        bl = np.zeros_like(self.flux)
        for tid in self.tids:
            mask_tid = tid==self.tidarr
            mask_nrm = mask_tid & self.msk_oot
            bl[mask_tid] = np.median(self.flux[mask_nrm])
        return bl

    @property
    def oot_flux_std(self):
        return self.normalized_flux[self.msk_oot].std()
