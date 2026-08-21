/*  PyTransit: fast and easy exoplanet transit modelling in Python.
 *  Copyright (C) 2010-2026  Hannu Parviainen
 *
 *  Shared device code for the OpenCL backends.
 *
 *  `REAL` is set to `double` or `float` by a -DREAL= build option, and
 *  USE_FP64 is defined alongside -DREAL=double. Every floating-point literal
 *  is cast to REAL so the single-precision build does not silently promote.
 */

#ifdef USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define PI_R ((REAL)3.14159265358979323846)

/* Number of orbital parameters carried through the gradient: (tc, p, a, i, e, w).
   MeepMeep also returns a `lan` row, which the transit model ignores; it is
   dropped host-side so it never reaches the device. */
#define NORB 6


/* Sort three values into descending order (*x >= *y >= *z).

   Mirrors `backends.numba.ccintersection._tsort`. The ordering is what keeps
   the kite-area product below well conditioned near the contact points, so it
   matters more in single precision, not less. */
inline void tsort3(REAL r1, REAL r2, REAL b, REAL *x, REAL *y, REAL *z) {
    if (r1 > r2) {
        if (r1 > b) {
            *x = r1;
            if (r2 > b) { *y = r2; *z = b;  }
            else        { *y = b;  *z = r2; }
        } else {
            *x = b; *y = r1; *z = r2;
        }
    } else {
        if (r2 > b) {
            *x = r2;
            if (r1 > b) { *y = r1; *z = b;  }
            else        { *y = b;  *z = r1; }
        } else {
            *x = b; *y = r2; *z = r1;
        }
    }
}


/* Area of intersection of two circles (Agol et al. 2020).

   Port of `backends.numba.ccintersection.ccia`. */
inline REAL ccia(REAL r1, REAL r2, REAL b) {
    if (r1 + r2 <= b)
        return (REAL)0.0;

    if (fabs(r1 - r2) < b && b <= r1 + r2) {
        REAL x, y, z;
        tsort3(r1, r2, b, &x, &y, &z);
        REAL a_kite = (REAL)0.5 * sqrt((x + (y + z)) * (z - (x - y)) * (z + (x - y)) * (x + (y - z)));
        REAL k0 = atan2((REAL)2.0 * a_kite, (r2 - r1) * (r2 + r1) + b * b);
        REAL k1 = atan2((REAL)2.0 * a_kite, (r1 - r2) * (r1 + r2) + b * b);
        return r1 * r1 * k1 + r2 * r2 * k0 - a_kite;
    }

    if (b <= r1 - r2) return PI_R * r2 * r2;
    if (b <= r2 - r1) return PI_R * r1 * r1;
    return NAN;
}


/* Intersection area together with its derivatives w.r.t. `r2` and `b`.

   Port of `backends.numba.ccintersection.ccia_and_grad`.

   Note on `a_kite`: the Python source leaves it unbound in the
   full-containment branch (`b <= r1 - r2`) and Numba happens to zero-fill the
   slot, which yields the physically correct `dadb = 0`. That branch covers
   most of a non-grazing transit, and an uninitialised read is undefined
   behaviour in C, so the zero is made explicit here. */
inline REAL ccia_and_grad(REAL r1, REAL r2, REAL b, REAL *dadr2, REAL *dadb) {
    REAL a_lens, k0, a_kite = (REAL)0.0;

    if (r1 + r2 <= b) {
        *dadr2 = (REAL)0.0;
        *dadb  = (REAL)0.0;
        return (REAL)0.0;
    } else if (fabs(r1 - r2) < b && b <= r1 + r2) {
        REAL x, y, z;
        tsort3(r1, r2, b, &x, &y, &z);
        a_kite = (REAL)0.5 * sqrt((x + (y + z)) * (z - (x - y)) * (z + (x - y)) * (x + (y - z)));
        k0 = atan2((REAL)2.0 * a_kite, (r2 - r1) * (r2 + r1) + b * b);
        REAL k1 = atan2((REAL)2.0 * a_kite, (r1 - r2) * (r1 + r2) + b * b);
        a_lens = r1 * r1 * k1 + r2 * r2 * k0 - a_kite;
    } else if (b <= r1 - r2) {
        a_lens = PI_R * r2 * r2;
        k0 = PI_R;
    } else if (b <= r2 - r1) {
        a_lens = PI_R * r1 * r1;
        k0 = (REAL)0.0;
    } else {
        *dadr2 = NAN;
        *dadb  = NAN;
        return NAN;
    }

    *dadr2 = (REAL)2.0 * k0 * r2;
    *dadb  = (fabs(b) > (REAL)1e-8) ? -(REAL)2.0 * a_kite / b : (REAL)0.0;
    return a_lens;
}


/* Sky-projected planet-star separation at an expansion-point-centred time.

   `cf` points at the 10 contiguous coefficients of a MeepMeep (2, 5) Taylor
   matrix: row 0 is x, row 1 is y. Port of `meepmeep.numba2d.sep_c`. */
inline REAL sep_c(REAL t, __global const REAL *cf) {
    REAL px = cf[0] + t * (cf[1] + t * (cf[2] + t * (cf[3] + t * cf[4])));
    REAL py = cf[5] + t * (cf[6] + t * (cf[7] + t * (cf[8] + t * cf[9])));
    return sqrt(px * px + py * py);
}


/* Separation and its derivatives w.r.t. (tc, p, a, i, e, w).

   `dcf` points at NORB blocks of 10 coefficients, one per parameter, and the
   NORB derivatives are written into `dz`. Port of `meepmeep`'s `_sep_cd_w`
   truncated to the six parameters the transit model uses. */
inline REAL sep_cd(REAL t, __global const REAL *cf, __global const REAL *dcf, REAL *dz) {
    REAL px = cf[0] + t * (cf[1] + t * (cf[2] + t * (cf[3] + t * cf[4])));
    REAL py = cf[5] + t * (cf[6] + t * (cf[7] + t * (cf[8] + t * cf[9])));
    REAL d = sqrt(px * px + py * py);

    for (int m = 0; m < NORB; m++) {
        __global const REAL *r = dcf + 10 * m;
        REAL dpx = r[0] + t * (r[1] + t * (r[2] + t * (r[3] + t * r[4])));
        REAL dpy = r[5] + t * (r[6] + t * (r[7] + t * (r[8] + t * r[9])));
        dz[m] = (px * dpx + py * dpy) / d;
    }
    return d;
}
