from numpy import ndarray, squeeze, asarray

from .model_full import rr_full
from .model_simple import rr_simple


def rrmodel(times, k, t0, p, a, i, e, w,
            parallelize, nlc, npb, nep,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, istar, weights, dk, kmin, kmax, dg, z_edges):

    if npb > 1 or nep > 1 or isinstance(k, ndarray):
        k, t0, p, a, i, e, w = asarray(k), asarray(t0), asarray(p), asarray(a), asarray(i), asarray(e), asarray(w)
        return squeeze(rr_full(times, k, t0, p, a, i, e, w, parallelize, nlc, npb, nep,
                       lcids, pbids, epids, nsamples, exptimes,
                       ldp, istar, weights, dk, kmin, kmax, dg, z_edges))
    else:
        return rr_simple(times, k, t0, p, a, i, e, w, parallelize, nlc, npb, nep,
                         lcids, pbids, epids, nsamples, exptimes,
                         ldp, istar, weights, dk, kmin, kmax, dg, z_edges)