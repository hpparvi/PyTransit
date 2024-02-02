from numpy import ndarray, squeeze, asarray, atleast_2d, atleast_1d

from .model_full import rr_full
from .model_simple import rr_simple


def rrmodel(times, k, t0, p, a, i, e, w,
            parallelize, nlc, npb, nep,
            lcids, pbids, epids, nsamples, exptimes,
            ldp, istar, weights, dk, kmin, kmax, dg, z_edges):

    k, t0, p, a, i, e, w = (atleast_2d(k), atleast_2d(t0), atleast_1d(p), atleast_1d(a),
                            atleast_1d(i), atleast_1d(e), atleast_1d(w))

    if nlc > 1:
        return squeeze(rr_full(times, k, t0, p, a, i, e, w, parallelize, nlc, npb, nep,
                       lcids, pbids, epids, nsamples, exptimes,
                       ldp, istar, weights, dk, kmin, kmax, dg, z_edges))
    else:
        return rr_simple(times, k[0, 0], t0[0, 0], p[0], a[0], i[0], e[0], w[0], parallelize, nsamples[0], exptimes[0],
                         ldp[0, 0, :], istar[0, 0], weights, dk, kmin, kmax, dg, z_edges)