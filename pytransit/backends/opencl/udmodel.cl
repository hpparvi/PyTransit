/*  PyTransit: fast and easy exoplanet transit modelling in Python.
 *  Copyright (C) 2010-2026  Hannu Parviainen
 *
 *  OpenCL uniform-disk transit model kernels.
 *
 *  Both kernels reproduce `backends.numba.udmodel` / `udmodel_grad` step for
 *  step. Two conventions are easy to get wrong and are load-bearing here:
 *
 *  1. `flux` is a flux *deficit*: it starts at zero and is only ever
 *     decremented, so it is 0 out of transit and -k^2 at the centre of a
 *     non-grazing transit. It is not normalised to one.
 *  2. The transit window comes from the host-side `bounding_box` contact
 *     points, evaluated with passband 0's radius ratio and epoch 0's
 *     coefficients for every point. That is what the Numba kernel does.
 *
 *  One work item handles one (parameter vector, time) pair, looping over its
 *  own supersamples. Dimension 0 is the time index so that neighbouring work
 *  items write neighbouring elements of the (npv, npt) output.
 */


__kernel void udmodel(__global const REAL *times,      /* (npt,)          */
                      __global const REAL *k,          /* (npv, npb)      */
                      __global const REAL *t0,         /* (npv, ntc)      */
                      __global const REAL *p,          /* (npv, nor)      */
                      __global const REAL *xyc,        /* (npv, nor, 10)  */
                      __global const REAL *bt1,        /* (npv,)          */
                      __global const REAL *bt4,        /* (npv,)          */
                      __global const int  *lcids,      /* (npt,)          */
                      __global const int  *pbids,      /* (nlc,)          */
                      __global const int  *epids,      /* (nlc,)          */
                      __global const int  *nsamples,   /* (nlc,)          */
                      __global const REAL *exptimes,   /* (nlc,)          */
                      const int npt, const int npb, const int ntc, const int nor,
                      __global REAL *flux)             /* (npv, npt)      */
{
    const int ipt = get_global_id(0);
    const int ipv = get_global_id(1);
    if (ipt >= npt) return;

    const int ilc = lcids[ipt];
    const int ipb = pbids[ilc];
    const int itc = epids[ilc];
    const int iep = (nor > 1) ? epids[ilc] : 0;

    const REAL t0v = t0[ipv * ntc + itc];
    const REAL pv  = p[ipv * nor + iep];
    const REAL et  = exptimes[ilc];

    REAL epoch;
    const REAL t = folded_time(times[ipt], t0v, pv, &epoch);

    REAL f = (REAL)0.0;
    if (((bt1[ipv] - et) <= t) && (t <= (bt4[ipv] + et))) {
        const REAL kv = k[ipv * npb + ipb];
        const int  ns = nsamples[ilc];
        __global const REAL *cf = xyc + (ipv * nor + iep) * 10;

        for (int is = 1; is <= ns; is++) {
            REAL toff = et * (((REAL)is - (REAL)0.5) / (REAL)ns - (REAL)0.5);
            REAL z = sep_c(t + toff, cf);
            if (z <= (REAL)1.0 + kv)
                f -= ccia((REAL)1.0, kv, z) / PI_R;
        }
        f /= (REAL)ns;
    }
    flux[ipv * npt + ipt] = f;
}


__kernel void udmodel_grad(__global const REAL *times,      /* (npt,)              */
                           __global const REAL *k,          /* (npv, npb)          */
                           __global const REAL *t0,         /* (npv, ntc)          */
                           __global const REAL *p,          /* (npv, nor)          */
                           __global const REAL *xyc,        /* (npv, nor, 10)      */
                           __global const REAL *dxyc,       /* (npv, nor, NORB, 10)*/
                           __global const REAL *bt1,        /* (npv,)              */
                           __global const REAL *bt4,        /* (npv,)              */
                           __global const int  *lcids,      /* (npt,)              */
                           __global const int  *pbids,      /* (nlc,)              */
                           __global const int  *epids,      /* (nlc,)              */
                           __global const int  *nsamples,   /* (nlc,)              */
                           __global const REAL *exptimes,   /* (nlc,)              */
                           const int npt, const int npb, const int ntc, const int nor,
                           __global REAL *flux,             /* (npv, npt)          */
                           __global REAL *dflux)            /* (npv, npt, 7)       */
{
    const int ipt = get_global_id(0);
    const int ipv = get_global_id(1);
    if (ipt >= npt) return;

    const int ilc = lcids[ipt];
    const int ipb = pbids[ilc];
    const int itc = epids[ilc];
    const int iep = (nor > 1) ? epids[ilc] : 0;

    const REAL t0v = t0[ipv * ntc + itc];
    const REAL pv  = p[ipv * nor + iep];
    const REAL et  = exptimes[ilc];

    REAL epoch;
    const REAL t = folded_time(times[ipt], t0v, pv, &epoch);

    /* Compact gradient slots, in the order [k, t0, p, a, i, e, w]. Out-of-window
       points keep the zeros they are initialised with, as in the Numba kernel. */
    REAL f = (REAL)0.0;
    REAL df[7];
    for (int m = 0; m < 7; m++) df[m] = (REAL)0.0;

    if (((bt1[ipv] - et) <= t) && (t <= (bt4[ipv] + et))) {
        const REAL kv = k[ipv * npb + ipb];
        const int  ns = nsamples[ilc];
        __global const REAL *cf  = xyc  + (ipv * nor + iep) * 10;
        __global const REAL *dcf = dxyc + (ipv * nor + iep) * NORB * 10;

        for (int is = 1; is <= ns; is++) {
            REAL toff = et * (((REAL)is - (REAL)0.5) / (REAL)ns - (REAL)0.5);
            REAL dz[NORB];
            REAL z = sep_cd(t + toff, cf, dcf, dz);
            if (z <= (REAL)1.0 + kv) {
                REAL dadk, dadz;
                REAL area = ccia_and_grad((REAL)1.0, kv, z, &dadk, &dadz);
                f     -= area / PI_R;
                df[0] -= dadk / PI_R;
                for (int m = 0; m < NORB; m++)
                    df[m + 1] -= dadz * dz[m] / PI_R;
            }
        }

        f /= (REAL)ns;
        for (int m = 0; m < 7; m++) df[m] /= (REAL)ns;

        /* Period folding correction. The folded time t - t0 - epoch*p depends on
           the period through the -epoch*p term, so the total period derivative
           (slot 2) gains epoch times the transit-centre derivative (slot 1). It
           must use the already-averaged slot 1, so it comes after the division;
           it vanishes at epoch 0 and grows with the transit number. */
        df[2] += epoch * df[1];
    }

    flux[ipv * npt + ipt] = f;
    __global REAL *out = dflux + (ipv * npt + ipt) * 7;
    for (int m = 0; m < 7; m++) out[m] = df[m];
}
