from numpy import ndarray, squeeze, asarray, atleast_2d, atleast_1d

from .model_full import rr_full
from .model_simple import rr_simple


def rrmodel(times, k, t0, p, a, i, e, w,
            parallelize, nlc, npb, nep,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, istar, weights, dk, kmin, kmax, dg, z_edges):

    if npb > 1 or nep > 1 or isinstance(k, ndarray):
        k, t0, p, a, i, e, w = atleast_2d(k), atleast_2d(t0), atleast_1d(p), atleast_1d(a), atleast_1d(i), atleast_1d(e), atleast_1d(w)
        return squeeze(rr_full(times, k, t0, p, a, i, e, w, parallelize, nlc, npb, nep,
                       lcids, pbids, epids, nsamples, exptimes,
                       ldp, istar, weights, dk, kmin, kmax, dg, z_edges))
    else:
        return rr_simple(times, k, t0, p, a, i, e, w, parallelize, nlc, npb, nep,
                         lcids, pbids, epids, nsamples, exptimes,
                         ldp, istar, weights, dk, kmin, kmax, dg, z_edges)