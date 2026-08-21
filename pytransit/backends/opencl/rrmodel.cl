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
 *  3. Limb darkening reaches the device only as `ldm`, the mean intensity under
 *     the planet disk tabulated against the grazing parameter g = z/(1+k), and
 *     contracted on the host. The device code is therefore independent of which
 *     limb-darkening profile was used.
 *
 *  NLDC (the number of limb-darkening coefficients) is a build option, so the
 *  private gradient accumulator has a static size.
 */


/* Linear interpolation of a mean limb-darkening table at the grazing parameter.

   Port of `backends.numba.rrmodel.interpolate_mean_limb_darkening`, with one
   deliberate difference: the index is clamped to `ng - 2`. Because
   dg = (1 - 1e-7)/(ng - 1), a g within ~1e-7 of 1.0 gives floor(g/dg) == ng - 1
   and the reference reads `lda[ng]`, one past the end. Numba returns whatever
   follows the array; on a device that is an out-of-bounds global read, so it is
   clamped here. The JAX backend clamps likewise.

   NaN g needs its own handling and cannot be folded into the range tests: NaN
   compares false against both, and `(int)floor(NaN)` is INT_MIN on at least
   NVIDIA, which slips under the upper clamp and indexes the table at a -16 GB
   offset. That is a memory fault, and it takes the shared context down for the
   rest of the process, not just the one evaluation. `k` is not covered by the
   parameter-vector validity guard, so a NaN radius ratio reaches here routinely
   from a sampler.

   Both the `isnan` test and the lower clamp guard this, and either alone is
   sufficient -- the clamp turns INT_MIN into 0 and the NaN then propagates
   through the interpolation anyway. Both are kept deliberately: the `isnan`
   states the intent, the clamp is a structural backstop for any other route to
   a negative index. Neither is dead code; removing both faults the device. */
inline REAL interp_ldm(REAL g, REAL dg, int ng, __global const REAL *ldm) {
    if (isnan(g)) return NAN;
    if (g < (REAL)0.0) return NAN;
    if (g > (REAL)1.0) return (REAL)0.0;
    int i = (int)floor(g / dg);
    if (i < 0) i = 0;
    if (i > ng - 2) i = ng - 2;
    REAL a = (g - (REAL)i * dg) / dg;
    return ((REAL)1.0 - a) * ldm[i] + a * ldm[i + 1];
}


/* As `interp_ldm`, also returning the derivative with respect to g.

   The interpolant is piecewise linear, so its derivative is piecewise constant:
   (ldm[i+1] - ldm[i]) / dg. */
inline REAL interp_ldm_and_grad(REAL g, REAL dg, int ng, __global const REAL *ldm, REAL *dv) {
    if (isnan(g)) { *dv = (REAL)0.0; return NAN; }   /* see `interp_ldm` */
    if (g < (REAL)0.0) { *dv = (REAL)0.0; return NAN; }
    if (g > (REAL)1.0) { *dv = (REAL)0.0; return (REAL)0.0; }
    int i = (int)floor(g / dg);
    if (i < 0) i = 0;
    if (i > ng - 2) i = ng - 2;
    REAL a = (g - (REAL)i * dg) / dg;
    *dv = (ldm[i + 1] - ldm[i]) / dg;
    return ((REAL)1.0 - a) * ldm[i] + a * ldm[i + 1];
}


__kernel void rrmodel(__global const REAL *times,      /* (npt,)          */
                      __global const REAL *k,          /* (npv, npb)      */
                      __global const REAL *t0,         /* (npv, ntc)      */
                      __global const REAL *p,          /* (npv, nor)      */
                      __global const REAL *xyc,        /* (npv, nor, 10)  */
                      __global const REAL *ldm,        /* (npv, npb, ng)  */
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
                      const int ng, const REAL dg,
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
        __global const REAL *ldm_pb = ldm + (ipv * npb + ipb) * ng;

        f = (REAL)0.0;
        for (int is = 1; is <= ns; is++) {
            REAL toff = et * (((REAL)is - (REAL)0.5) / (REAL)ns - (REAL)0.5);
            REAL z = sep_c(t + toff, cf);
            /* No `z <= 1+k` guard: ccia returns 0 and the interpolation returns 0
               for g > 1, so an untransiting sample degenerates to (ldi-0)/ldi = 1. */
            REAL ip = interp_ldm(z / ((REAL)1.0 + kv), dg, ng, ldm_pb);
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
                           __global const REAL *ldm,        /* (npv, npb, ng)        */
                           __global const REAL *dldm,       /* (npv, npb, NLDC, ng)  */
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
                           const int ng, const REAL dg,
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
        __global const REAL *ldm_pb  = ldm  + (ipv * npb + ipb) * ng;
        __global const REAL *dldm_pb = dldm + (size_t)(ipv * npb + ipb) * NLDC * ng;

        f = (REAL)0.0;
        for (int is = 1; is <= ns; is++) {
            REAL toff = et * (((REAL)is - (REAL)0.5) / (REAL)ns - (REAL)0.5);
            REAL dz[NORB];
            REAL z = sep_cd(t + toff, cf, dcf, dz);

            REAL dadk, dadz;
            REAL ap = ccia_and_grad((REAL)1.0, kv, z, &dadk, &dadz);

            REAL g = z / ((REAL)1.0 + kv);
            REAL dip_dg;
            REAL ip = interp_ldm_and_grad(g, dg, ng, ldm_pb, &dip_dg);

            f += (ldiv - ip * ap) / ldiv;

            /* dg/dz = 1/(1+k) and dg/dk = -z/(1+k)^2 at fixed z. The k derivative
               neglects dldm/dk, i.e. the weight table's own k-dependence; the
               Numba tests allow rtol=0.1 on this column for that reason. */
            REAL dip_dz = dip_dg / ((REAL)1.0 + kv);
            REAL dip_dk = dip_dg * (-z / (((REAL)1.0 + kv) * ((REAL)1.0 + kv)));
            df[0] += -(dip_dk * ap + ip * dadk) / ldiv;

            REAL dflux_dz = -(dip_dz * ap + ip * dadz) / ldiv;
            for (int m = 0; m < NORB; m++)
                df[m + 1] += dflux_dz * dz[m];

            /* Period folding correction: the folded time depends on the period
               through -epoch*p, which dz does not see. Zero at epoch 0, growing
               with the transit number. */
            df[2] += epoch * dflux_dz * dz[0];

            for (int j = 0; j < NLDC; j++)
                df[7 + j] += -ap * interp_ldm(g, dg, ng, dldm_pb + j * ng) / ldiv;
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
