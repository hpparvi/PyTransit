/*  PyTransit: fast and easy exoplanet transit modelling in Python.
 *  Copyright (C) 2010-2026  Hannu Parviainen
 *
 *  Device-side white-noise log-likelihood and gradient reduction.
 *
 *  The model flux and its Jacobian are by far the largest arrays in a fit, and
 *  reading the Jacobian back dominates the runtime of a gradient-based sampler.
 *  This kernel reduces both on the device, so an evaluation returns a handful of
 *  doubles instead of an (npt, ngrad) array.
 *
 *  Noise model: independent Gaussians with a fixed per-point `err` and a sampled
 *  `jitter`, so e2 = err^2 + jitter^2. A constant `err` gives the i.i.d. case.
 *
 *      lnL          = -N/2 log(2 pi) - 1/2 sum(log e2) - 1/2 sum(r^2/e2)
 *      dlnL/dtheta  = sum (r/e2) dm/dtheta
 *      dlnL/djitter = jitter [ sum(r^2/e2^2) - sum(1/e2) ]
 *
 *  with r = obs - model. The four scalar sums and every gradient column are
 *  produced by one kernel: each work-group owns one output slot and reduces the
 *  whole time series into it, so `s` is uniform within a group and none of the
 *  branching below diverges.
 *
 *  Gradient slots carry a rule saying which data points they accept, which makes
 *  the output land directly in the *expanded* Jacobian layout that `evaluate`
 *  returns -- a parameter shared across passbands or epochs accepts every point
 *  and collapses to one column, while a per-passband or per-epoch parameter gets
 *  one slot per group. No host-side scatter is needed.
 */

#define SLOT_CHI2       0   /* sum r^2/e2   */
#define SLOT_LOG_E2     1   /* sum log(e2)  */
#define SLOT_CHI2_E2    2   /* sum r^2/e2^2 */
#define SLOT_INV_E2     3   /* sum 1/e2     */
#define NSCALAR         4

#define MATCH_ALL       0   /* parameter is shared: accept every point       */
#define MATCH_PASSBAND  1   /* accept points whose passband is `grp`         */
#define MATCH_EPOCH     2   /* accept points whose epoch is `grp`            */


__kernel void wn_loglikelihood(__global const REAL *obs,        /* (npt,)             */
                               __global const REAL *err,        /* (npt,)             */
                               __global const REAL *flux,       /* (npv, npt)         */
                               __global const REAL *dflux,      /* (npv, npt, ngrad)  */
                               __global const int  *lcids,      /* (npt,)             */
                               __global const int  *pbids,      /* (nlc,)             */
                               __global const int  *epids,      /* (nlc,)             */
                               __global const int  *slot_col,   /* (nslot,)           */
                               __global const int  *slot_kind,  /* (nslot,)           */
                               __global const int  *slot_grp,   /* (nslot,)           */
                               const int npt, const int ngrad, const int nslot,
                               const REAL jitter,
                               __local REAL *scratch,
                               __global REAL *out)              /* (npv, nslot)       */
{
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int s   = get_group_id(0);      /* output slot; uniform in this group */
    const int ipv = get_global_id(1);

    const REAL j2 = jitter * jitter;
    const int kind = slot_kind[s];
    const int col  = slot_col[s];
    const int grp  = slot_grp[s];

    /* Kahan-compensated accumulation. Each work-item sums npt/lsz terms serially,
       and naive summation there loses O(n) ulps -- which in single precision costs
       the sum(log e2) slot several nats at large npt, biasing the jitter posterior
       because that slot moves with jitter. Compensation makes the error independent
       of the term count. It survives compilation because the backend is built
       without -cl-fast-relaxed-math; do not add that flag. */
    REAL acc = (REAL)0.0, comp = (REAL)0.0;
    for (int ipt = lid; ipt < npt; ipt += lsz) {
        REAL e = err[ipt];
        REAL e2 = e * e + j2;
        REAL term;

        if (s == SLOT_LOG_E2) {
            term = log(e2);                       /* model-independent */
        } else if (s == SLOT_INV_E2) {
            term = (REAL)1.0 / e2;                /* model-independent */
        } else {
            REAL r = obs[ipt] - flux[(size_t)ipv * npt + ipt];
            if (s == SLOT_CHI2) {
                term = r * r / e2;
            } else if (s == SLOT_CHI2_E2) {
                term = r * r / (e2 * e2);
            } else {
                int g = grp;
                if (kind != MATCH_ALL) {
                    int ilc = lcids[ipt];
                    g = (kind == MATCH_PASSBAND) ? pbids[ilc] : epids[ilc];
                }
                if (g != grp) continue;
                term = (r / e2) * dflux[((size_t)ipv * npt + ipt) * ngrad + col];
            }
        }

        REAL y = term - comp;
        REAL t = acc + y;
        comp = (t - acc) - y;
        acc = t;
    }

    /* Tree reduction across the work-group; the host guarantees a power-of-two
       local size so no odd-size handling is needed. NaN from an invalid
       parameter vector propagates through, as it should. */
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) out[ipv * nslot + s] = scratch[0];
}
