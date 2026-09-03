/*  PyTransit: fast and easy exoplanet transit modelling in Python.
 *  Copyright (C) 2010-2026  Hannu Parviainen
 *
 *  OpenCL RoadRunner transit model kernels (Parviainen 2020).
 *
 *  Both kernels reproduce `backends.numba.rrmodel` / `rrmodel_grad` step for
 *  step. Three conventions differ from the uniform-disk kernels and are easy to
 *  carry over wrongly:
 *
 *  1. The flux is *normalised*: it is 1.0 out of transit, and each supersample
 *     contributes `(ldi - Ip*A)/ldi`. It is not a deficit accumulated from zero.
 *  2. There is a per-parameter-vector validity guard. An invalid vector yields
 *     an all-NaN flux and gradient row.
 *  3. Limb darkening reaches the device only as split cubic tables of the mean
 *     intensity under the planet disk against the grazing parameter g = z/(1+k),
 *     built on the host by `backends.numba.ldquad.rr_tables`. The device code is
 *     therefore independent of which limb-darkening profile was used.
 *
 *  NLDC (the number of limb-darkening coefficients) is a build option, so the
 *  private gradient accumulator has a static size.
 */


/* Split cubic table lookup at the grazing parameter g.

   Port of `backends.numba.ldquad.ldm_lookup`. The table has ng/2 uniform nodes
   from 0 to the limb contact gc and ng - ng/2 from gc to 1, stored as per-interval
   cubic coefficients (ng - 2 rows of 4). `ldm_locate` returns the row, the local
   coordinate a in [0, 1) and the interval width h.

   g > 1 returns 0 so that an untransiting sample degenerates to (ldi-0)/ldi = 1;
   the numba kernels guard the same case with `z >= 1 + k`.

   NaN g needs its own handling and cannot be folded into the range tests: NaN
   compares false against both, and `(int)NaN` is INT_MIN on at least NVIDIA,
   which would index the table at a -16 GB offset. That is a memory fault, and it
   takes the shared context down for the rest of the process. `k` is not covered
   by the parameter-vector validity guard, so a NaN radius ratio reaches here
   routinely from a sampler. The `isnan` test and the lower clamp both guard this;
   both are kept deliberately. */
inline int ldm_locate(REAL g, REAL gc, int ng, REAL *a, REAL *h) {
    const int n1 = ng / 2;
    const int n2 = ng - n1;
    REAL x;
    int i;
    if (g < gc) {
        x = g / gc * (REAL)(n1 - 1);
        i = (int)x;
        if (i < 0) i = 0;
        if (i > n1 - 2) i = n1 - 2;
        *a = x - (REAL)i;
        *h = gc / (REAL)(n1 - 1);
        return i;
    }
    x = (g - gc) / ((REAL)1.0 - gc) * (REAL)(n2 - 1);
    i = (int)x;
    if (i < 0) i = 0;
    if (i > n2 - 2) i = n2 - 2;
    *a = x - (REAL)i;
    *h = ((REAL)1.0 - gc) / (REAL)(n2 - 1);
    return i + n1 - 1;
}

inline REAL ldm_lookup(REAL g, REAL gc, int ng, __global const REAL *coef) {
    if (isnan(g)) return NAN;
    if (g < (REAL)0.0) return NAN;
    if (g > (REAL)1.0) return (REAL)0.0;
    REAL a, h;
    const int i = ldm_locate(g, gc, ng, &a, &h);
    __global const REAL *c = coef + i * 4;
    return c[0] + a * (c[1] + a * (c[2] + a * c[3]));
}

/* As `ldm_lookup`, also returning the derivative with respect to g: the analytic
   derivative of the interval's cubic. */
inline REAL ldm_lookup_and_grad(REAL g, REAL gc, int ng, __global const REAL *coef, REAL *dv) {
    if (isnan(g)) { *dv = (REAL)0.0; return NAN; }
    if (g < (REAL)0.0) { *dv = (REAL)0.0; return NAN; }
    if (g > (REAL)1.0) { *dv = (REAL)0.0; return (REAL)0.0; }
    REAL a, h;
    const int i = ldm_locate(g, gc, ng, &a, &h);
    __global const REAL *c = coef + i * 4;
    *dv = (c[1] + a * ((REAL)2.0 * c[2] + (REAL)3.0 * a * c[3])) / h;
    return c[0] + a * (c[1] + a * (c[2] + a * c[3]));
}

__kernel void rrmodel(__global const REAL *times,      /* (npt,)          */
                      __global const REAL *k,          /* (npv, npb)      */
                      __global const REAL *t0,         /* (npv, ntc)      */
                      __global const REAL *p,          /* (npv, nor)      */
                      __global const REAL *xyc,        /* (npv, nor, 10)  */
                      __global const REAL *gcs,        /* (npv, npb)             */
                      __global const REAL *coef,       /* (npv, npb, ng - 2, 4)  */
                      __global const REAL *ldi,        /* (npv, npb)      */
                      __global const REAL *bt1,        /* (npv,)          */
                      __global const REAL *bt4,        /* (npv,)          */
                      __global const int  *valid,      /* (npv,)          */
                      __global const int  *lcids,      /* (npt,)          */
                      __global const int  *pbids,      /* (nlc,)          */
                      __global const int  *epids,      /* (nlc,)          */
                      __global const int  *nsamples,   /* (nlc,)          */
                      __global const REAL *exptimes,   /* (nlc,)          */
                      const int npt, const int npb, const int ntc, const int nor,
                      const int ng,
                      __global REAL *flux)             /* (npv, npt)      */
{
    const int ipt = get_global_id(0);
    const int ipv = get_global_id(1);
    if (ipt >= npt) return;

    if (!valid[ipv]) {
        flux[ipv * npt + ipt] = NAN;
        return;
    }

    const int ilc = lcids[ipt];
    const int ipb = pbids[ilc];
    const int itc = epids[ilc];
    const int iep = (nor > 1) ? epids[ilc] : 0;

    const REAL t0v = t0[ipv * ntc + itc];
    const REAL pv  = p[ipv * nor + iep];
    const REAL et  = exptimes[ilc];

    REAL epoch;
    const REAL t = folded_time(times[ipt], t0v, pv, &epoch);

    REAL f = (REAL)1.0;
    if (((bt1[ipv] - et) <= t) && (t <= (bt4[ipv] + et))) {
        const REAL kv  = k[ipv * npb + ipb];
        const REAL ldiv = ldi[ipv * npb + ipb];
        const int  ns  = nsamples[ilc];
        __global const REAL *cf  = xyc + (ipv * nor + iep) * 10;
        const REAL gcv = gcs[ipv * npb + ipb];
        __global const REAL *coef_pb = coef + (size_t)(ipv * npb + ipb) * (ng - 2) * 4;

        f = (REAL)0.0;
        for (int is = 1; is <= ns; is++) {
            REAL toff = et * (((REAL)is - (REAL)0.5) / (REAL)ns - (REAL)0.5);
            REAL z = sep_c(t + toff, cf);
            /* No `z <= 1+k` guard: ccia returns 0 and the lookup returns 0
               for g > 1, so an untransiting sample degenerates to (ldi-0)/ldi = 1. */
            REAL ip = ldm_lookup(z / ((REAL)1.0 + kv), gcv, ng, coef_pb);
            REAL ap = ccia((REAL)1.0, kv, z);
            f += (ldiv - ip * ap) / ldiv;
        }
        f /= (REAL)ns;
    }
    flux[ipv * npt + ipt] = f;
}


__kernel void rrmodel_grad(__global const REAL *times,      /* (npt,)                */
                           __global const REAL *k,          /* (npv, npb)            */
                           __global const REAL *t0,         /* (npv, ntc)            */
                           __global const REAL *p,          /* (npv, nor)            */
                           __global const REAL *xyc,        /* (npv, nor, 10)        */
                           __global const REAL *dxyc,       /* (npv, nor, NORB, 10)  */
                           __global const REAL *gcs,        /* (npv, npb)                  */
                           __global const REAL *coef,       /* (npv, npb, ng - 2, 4)       */
                           __global const REAL *kcoef,      /* (npv, npb, ng - 2, 4)       */
                           __global const REAL *dcoef,      /* (npv, npb, NLDC, ng - 2, 4) */
                           __global const REAL *ldi,        /* (npv, npb)            */
                           __global const REAL *dldi,       /* (npv, npb, NLDC)      */
                           __global const REAL *bt1,        /* (npv,)                */
                           __global const REAL *bt4,        /* (npv,)                */
                           __global const int  *valid,      /* (npv,)                */
                           __global const int  *lcids,      /* (npt,)                */
                           __global const int  *pbids,      /* (nlc,)                */
                           __global const int  *epids,      /* (nlc,)                */
                           __global const int  *nsamples,   /* (nlc,)                */
                           __global const REAL *exptimes,   /* (nlc,)                */
                           const int npt, const int npb, const int ntc, const int nor,
                           const int ng,
                           __global REAL *flux,             /* (npv, npt)            */
                           __global REAL *dflux)            /* (npv, npt, 7+npb*NLDC)*/
{
    const int ipt = get_global_id(0);
    const int ipv = get_global_id(1);
    if (ipt >= npt) return;

    const int ngrad = 7 + npb * NLDC;
    __global REAL *out = dflux + (size_t)(ipv * npt + ipt) * ngrad;

    if (!valid[ipv]) {
        flux[ipv * npt + ipt] = NAN;
        for (int m = 0; m < ngrad; m++) out[m] = NAN;
        return;
    }

    const int ilc = lcids[ipt];
    const int ipb = pbids[ilc];
    const int itc = epids[ilc];
    const int iep = (nor > 1) ? epids[ilc] : 0;

    const REAL t0v = t0[ipv * ntc + itc];
    const REAL pv  = p[ipv * nor + iep];
    const REAL et  = exptimes[ilc];

    REAL epoch;
    const REAL t = folded_time(times[ipt], t0v, pv, &epoch);

    /* Only this point's own passband block is ever non-zero, so the private
       accumulator holds just the 7 orbital slots plus one passband's NLDC
       coefficients; the row is scattered on write-out. */
    REAL f = (REAL)1.0;
    REAL df[7 + NLDC];
    for (int m = 0; m < 7 + NLDC; m++) df[m] = (REAL)0.0;

    if (((bt1[ipv] - et) <= t) && (t <= (bt4[ipv] + et))) {
        const REAL kv   = k[ipv * npb + ipb];
        const REAL ldiv = ldi[ipv * npb + ipb];
        const int  ns   = nsamples[ilc];
        __global const REAL *cf  = xyc  + (ipv * nor + iep) * 10;
        __global const REAL *dcf = dxyc + (ipv * nor + iep) * NORB * 10;
        const REAL gcv = gcs[ipv * npb + ipb];
        /* Rate of change of the limb contact with k. */
        const REAL dgc_dk = (REAL)-2.0 / (((REAL)1.0 + kv) * ((REAL)1.0 + kv));
        __global const REAL *coef_pb  = coef  + (size_t)(ipv * npb + ipb) * (ng - 2) * 4;
        __global const REAL *kcoef_pb = kcoef + (size_t)(ipv * npb + ipb) * (ng - 2) * 4;
        __global const REAL *dcoef_pb = dcoef + (size_t)(ipv * npb + ipb) * NLDC * (ng - 2) * 4;

        f = (REAL)0.0;
        for (int is = 1; is <= ns; is++) {
            REAL toff = et * (((REAL)is - (REAL)0.5) / (REAL)ns - (REAL)0.5);
            REAL dz[NORB];
            REAL z = sep_cd(t + toff, cf, dcf, dz);

            REAL dadk, dadz;
            REAL ap = ccia_and_grad((REAL)1.0, kv, z, &dadk, &dadz);

            REAL g = z / ((REAL)1.0 + kv);
            REAL dip_dg;
            REAL ip = ldm_lookup_and_grad(g, gcv, ng, coef_pb, &dip_dg);

            f += (ldiv - ip * ap) / ldiv;

            /* dg/dz = 1/(1+k) and dg/dk = -z/(1+k)^2 at fixed z. The tabulated k
               derivative follows the node trajectory g(k); removing the part due
               to the node moving gives the derivative at fixed g. */
            REAL dip_dz = dip_dg / ((REAL)1.0 + kv);
            REAL dg_dk = (g < gcv) ? g * dgc_dk / gcv : dgc_dk * ((REAL)1.0 - g) / ((REAL)1.0 - gcv);
            REAL dip_dk_fixed_g = ldm_lookup(g, gcv, ng, kcoef_pb) - dip_dg * dg_dk;
            REAL dip_dk = dip_dg * (-z / (((REAL)1.0 + kv) * ((REAL)1.0 + kv))) + dip_dk_fixed_g;
            df[0] += -(dip_dk * ap + ip * dadk) / ldiv;

            REAL dflux_dz = -(dip_dz * ap + ip * dadz) / ldiv;
            for (int m = 0; m < NORB; m++)
                df[m + 1] += dflux_dz * dz[m];

            /* Period folding correction: the folded time depends on the period
               through -epoch*p, which dz does not see. Zero at epoch 0, growing
               with the transit number. */
            df[2] += epoch * dflux_dz * dz[0];

            for (int j = 0; j < NLDC; j++)
                df[7 + j] += -ap * ldm_lookup(g, gcv, ng, dcoef_pb + j * (ng - 2) * 4) / ldiv;
        }

        f /= (REAL)ns;
        for (int m = 0; m < 7 + NLDC; m++) df[m] /= (REAL)ns;

        /* The disk-integrated intensity also depends on the LD coefficients, and
           this term uses the already-averaged flux, so it comes after the
           division rather than inside the sample loop. */
        __global const REAL *dldi_pb = dldi + (ipv * npb + ipb) * NLDC;
        for (int j = 0; j < NLDC; j++)
            df[7 + j] += ((REAL)1.0 - f) * dldi_pb[j] / ldiv;
    }

    flux[ipv * npt + ipt] = f;
    for (int m = 0; m < 7; m++) out[m] = df[m];
    for (int b = 0; b < npb; b++)
        for (int j = 0; j < NLDC; j++)
            out[7 + b * NLDC + j] = (b == ipb) ? df[7 + j] : (REAL)0.0;
}
