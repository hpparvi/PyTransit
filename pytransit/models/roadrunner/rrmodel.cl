/*  PyTransit: fast and easy exoplanet transit modelling in Python.
 *  Copyright (C) 2010-2026  Hannu Parviainen
 *
 *  OpenCL RoadRunner transit model (Parviainen 2020).
 *
 *  The evaluation runs as three kernels:
 *
 *  1. `calculate_ldm` integrates the tabulated stellar intensity profile over the planet's
 *     footprint by Gauss quadrature matched to the transit geometry, giving the mean intensity
 *     under the planet at every node of a grazing parameter table split at the limb contact.
 *     One work item per (parameter vector, passband, node). This is a port of
 *     `common.ldm_nodes` and `common.ldm_table`.
 *  2. `calculate_coefficients` turns each table into per-interval cubic coefficients, one work
 *     item per (parameter vector, passband, interval). A port of `common.split_cubic_coefficients`.
 *  3. `rr_flux` evaluates the model, one work item per (parameter vector, time sample), reading
 *     the mean intensity with the split cubic lookup of `common.ldm_lookup`. The projected
 *     separation comes from MeepMeep's `sep_c2`, the device twin of the `sep_c` the Numba model
 *     uses, so the two backends share the orbit to the last bit.
 *
 *  MeepMeep's `point2d.cl` and its `common.cl` are prepended to this source by the host, and
 *  supply `taylor5`, `sep_c2`, the fp64 pragma, and `PI_R` / `TWO_PI_R_R`.
 *
 *  The floating point type is set by a -DREAL= build option, with USE_FP64 defined alongside
 *  -DREAL=double. Every floating point literal is cast to REAL: a bare literal is a double in
 *  C, which would silently promote the single precision build to double arithmetic where the
 *  device supports it and fail to compile where it does not.
 */


REAL circle_circle_intersection_area(REAL r1, REAL r2, REAL b){
    if (r1 < b - r2){
        return (REAL)0.0;
    }
    else if (r1 >= b + r2){
        return PI_R * r2 * r2;
    }
    else if (b - r2 <= -r1){
        return PI_R * r1 * r1;
    }
    else{
        return r2*r2 * acos((b*b + r2*r2 - r1*r1) / (2 * b * r2)) +
               r1*r1 * acos((b*b + r1*r1 - r2*r2) / (2 * b * r1)) -
               (REAL)0.5 * sqrt((-b + r2 + r1) * (b + r2 - r1) * (b - r2 + r1) * (b + r2 + r1));
    }
}


/* ------------------------------------------------------------------------------------------ */
/* Mean intensity under the planet by quadrature                                              */
/* ------------------------------------------------------------------------------------------ */

/* The grazing parameter at which the planet first touches the stellar limb. The mean intensity
   under the planet has a kink here, so the g table is split at it. */
inline REAL split_point(const REAL k){
    return ((REAL)1.0 - k) / ((REAL)1.0 + k);
}

/* Number of table nodes in the first segment, [0, gc]: proportional to its length, but at least
   four nodes in either segment so that a cubic can be fitted. Port of `common.g_nodes`. */
inline int first_segment_size(const REAL gc, const int ng){
    int n1 = (int) rint(ng * gc);
    if (n1 < 4) n1 = 4;
    if (ng - n1 < 4) n1 = ng - 4;
    return n1;
}

/* Grazing parameter of table node `ig`: two uniform segments meeting at gc, which is stored
   twice, as the last node of the first segment and the first node of the second. */
inline REAL g_node(const int ig, const REAL gc, const int n1, const int ng){
    if (ig < n1){
        return gc * (REAL) ig / (REAL) (n1 - 1);
    }
    else{
        return gc + ((REAL)1.0 - gc) * (REAL) (ig - n1) / (REAL) (ng - n1 - 1);
    }
}

/* Angular extent of the planet disk at stellar radius z for a planet at separation b. */
inline REAL planet_angular_extent(const REAL z, const REAL b, const REAL k){
    if (b < (REAL)1e-7){
        return (z < k) ? TWO_PI_R : (REAL)0.0;
    }
    if (z <= k - b){
        return TWO_PI_R;
    }
    if (z < b - k || z > b + k){
        return (REAL)0.0;
    }
    REAL c = (z * z + b * b - k * k) / ((REAL)2.0 * z * b);
    c = clamp(c, -(REAL)1.0, (REAL)1.0);
    return (REAL)2.0 * acos(c);
}

/* Cubic interpolation of a profile tabulated on the grid mu = (t0 + dt * i)**2, i = 0..n-1.
   Port of `common.profile_at`. */
inline REAL profile_at(const REAL m, const REAL t0, const REAL dt, __global const REAL *ldp, const int n){
    REAL x = (sqrt(m) - t0) / dt;
    if (x <= (REAL)0.0) return ldp[0];
    if (x >= (REAL) (n - 1)) return ldp[n - 1];
    int i = (int) x - 1;
    if (i < 0) i = 0;
    else if (i > n - 4) i = n - 4;
    REAL u = x - (REAL) i;
    return (-(u - (REAL)1.0) * (u - (REAL)2.0) * (u - (REAL)3.0) / (REAL)6.0 * ldp[i]
            + u * (u - (REAL)2.0) * (u - (REAL)3.0) / (REAL)2.0 * ldp[i + 1]
            - u * (u - (REAL)1.0) * (u - (REAL)3.0) / (REAL)2.0 * ldp[i + 2]
            + u * (u - (REAL)1.0) * (u - (REAL)2.0) / (REAL)6.0 * ldp[i + 3]);
}

/* Mu from z without the rounding of z*z past one producing a NaN. */
inline REAL mu_from_z(const REAL z){
    return sqrt(fmax((REAL)0.0, (REAL)1.0 - z * z));
}

/* The mean intensity under the planet at every node of the grazing parameter table.

   Global size (npv, npb, ng). Port of `common.ldm_nodes` followed by `common.ldm_table`: the
   quadrature nodes and geometric factors for this node's geometry are generated and consumed
   on the fly. `rules` holds the Gauss-Legendre nodes and weights followed by the Gauss-Jacobi
   (0.5, 0) nodes and weights, nq of each. The work item of node 0 also stores the limb contact
   and the first segment size for the other kernels. A NaN radius ratio gives a NaN table. */
__kernel void calculate_ldm(__global const REAL *ks,    /* (npv, npb)      */
                            __global const REAL *ldp,   /* (npv, npb, nmu) */
                            __global const REAL *rules, /* (4, nq)         */
                            const REAL pt0, const REAL pdt, const int nmu, const int nq,
                            __global REAL *gcs,         /* (npv, npb)      */
                            __global int   *n1s,        /* (npv, npb)      */
                            __global REAL *ldm)         /* (npv, npb, ng)  */
{
    const int ipv = get_global_id(0);
    const int ipb = get_global_id(1);
    const int npb = get_global_size(1);
    const int ig  = get_global_id(2);
    const int ng  = get_global_size(2);
    const int ipp = ipv * npb + ipb;

    const REAL k  = ks[ipp];
    const REAL gc = split_point(k);

    if (isnan(k)){
        if (ig == 0){
            gcs[ipp] = NAN;
            n1s[ipp] = 4;
        }
        ldm[ipp * ng + ig] = NAN;
        return;
    }

    const int n1 = first_segment_size(gc, ng);
    if (ig == 0){
        gcs[ipp] = gc;
        n1s[ipp] = n1;
    }

    __global const REAL *prof = ldp + ipp * nmu;
    __global const REAL *t_gl = rules;
    __global const REAL *w_gl = rules + nq;
    __global const REAL *t_gj = rules + 2 * nq;
    __global const REAL *w_gj = rules + 3 * nq;

    const REAL g = g_node(ig, gc, n1, ng);

    /* At g = 1 the overlap vanishes and the mean intensity is the profile at the limb. The Numba
       code evaluates the geometry a hair inside the limb instead, where every quadrature node
       falls below the first profile node and `profile_at` returns the same value; in single
       precision that hair would round away, leaving the mean 0/0. */
    if (g >= (REAL)1.0){
        ldm[ipp * ng + ig] = prof[0];
        return;
    }

    const REAL b = g * ((REAL)1.0 + k);

    REAL num = (REAL)0.0;
    REAL den = (REAL)0.0;
    REAL c, h, m, z, sq, w;

    if (b + k <= (REAL)1.0){
        if (b < k){
            /* z in [0, k - b]: theta = 2 pi. Integrated in mu with Gauss-Legendre. */
            const REAL mu_mid = mu_from_z(k - b);
            c = (REAL)0.5 * (mu_mid + (REAL)1.0);
            h = (REAL)0.5 * ((REAL)1.0 - mu_mid);
            for (int q = 0; q < nq; q++){
                m = c + h * t_gl[q];
                z = mu_from_z(m);
                w = w_gl[q] * planet_angular_extent(z, b, k) * m * h;
                num += w * profile_at(m, pt0, pdt, prof, nmu);
                den += w;
            }
            /* z in [k - b, b + k] with z = (k - b) + s**2: regular at s = 0, sqrt zero at the end. */
            c = h = (REAL)0.5 * sqrt((REAL)2.0 * b);
            for (int q = 0; q < nq; q++){
                sq = c + h * t_gj[q];
                z = k - b + sq * sq;
                m = mu_from_z(z);
                w = w_gj[q] * planet_angular_extent(z, b, k) * z * (REAL)2.0 * sq * h;
                num += w * profile_at(m, pt0, pdt, prof, nmu);
                den += w;
            }
        }
        else{
            /* z in [b - k, b + k] with z = (b + k) - s**2: the limb end is regular, and the sqrt
               zero of theta at z = b - k sits at s_max, where the Jacobi weight takes it. */
            c = h = (REAL)0.5 * sqrt((REAL)2.0 * k);
            for (int q = 0; q < nq; q++){
                sq = c + h * t_gj[q];
                z = b + k - sq * sq;
                m = mu_from_z(z);
                w = w_gj[q] * planet_angular_extent(z, b, k) * z * (REAL)2.0 * sq * h;
                num += w * profile_at(m, pt0, pdt, prof, nmu);
                den += w;
            }
        }
    }
    else{
        /* z in [b - k, 1]: integrated in mu from the limb, where the profile is regular, to
           mu_lo, where theta has its sqrt zero. The distance of the inner contact from the limb,
           1 - (b - k) = (1 - g)(1 + k), is formed without the cancellation of b - k near 1. */
        const REAL dlimb = ((REAL)1.0 - g) * ((REAL)1.0 + k);
        const REAL mu_lo = sqrt(dlimb * ((REAL)2.0 - dlimb));
        c = h = (REAL)0.5 * mu_lo;
        for (int q = 0; q < nq; q++){
            m = c + h * t_gj[q];
            z = mu_from_z(m);
            w = w_gj[q] * planet_angular_extent(z, b, k) * m * h;
            num += w * profile_at(m, pt0, pdt, prof, nmu);
            den += w;
        }
    }
    ldm[ipp * ng + ig] = num / den;
}

/* Per-interval cubic coefficients of the split mean intensity tables.

   Global size (npv, npb, ng - 2). Port of `common.split_cubic_coefficients`: interval `ii`
   belongs to the first segment if ii < n1 - 1 and to the second otherwise, and its cubic is
   the four-point Lagrange interpolant on a stencil shifted inwards at the segment ends. `cm`
   holds the three (4, 4) stencil matrices of `common.CUBIC_MATRICES`. */
__kernel void calculate_coefficients(__global const REAL *ldm,  /* (npv, npb, ng)         */
                                     __global const int   *n1s, /* (npv, npb)             */
                                     __global const REAL *cm,   /* (3, 4, 4)              */
                                     const int ng,
                                     __global REAL *coef)       /* (npv, npb, ng - 2, 4)  */
{
    const int ipv = get_global_id(0);
    const int ipb = get_global_id(1);
    const int npb = get_global_size(1);
    const int ii  = get_global_id(2);
    const int ipp = ipv * npb + ipb;

    const int n1 = n1s[ipp];
    __global const REAL *tab = ldm + ipp * ng;
    int nseg, i;
    if (ii < n1 - 1){
        nseg = n1;
        i = ii;
    }
    else{
        nseg = ng - n1;
        i = ii - (n1 - 1);
        tab += n1;
    }
    int j = i - 1;
    if (j < 0) j = 0;
    else if (j > nseg - 4) j = nseg - 4;
    const int s = i - j;

    __global REAL *c = coef + (ipp * (ng - 2) + ii) * 4;
    for (int r = 0; r < 4; r++){
        __global const REAL *row = cm + (s * 4 + r) * 4;
        c[r] = row[0] * tab[j] + row[1] * tab[j + 1] + row[2] * tab[j + 2] + row[3] * tab[j + 3];
    }
}

/* The mean intensity under the planet at grazing parameter g from the split cubic table.

   Port of `common.ldm_lookup`. The NaN test is not decorative: NaN compares false against
   both range tests, and `(int) NAN` is INT_MIN on at least NVIDIA, which would index the table
   far outside its buffer. */
inline REAL ldm_lookup(const REAL g, const REAL gc, const int n1, const int ng, __global const REAL *coef){
    if (isnan(g) || isnan(gc)) return NAN;
    if (g < (REAL)0.0) return NAN;
    if (g > (REAL)1.0) return (REAL)0.0;
    REAL x;
    int i;
    if (g < gc){
        x = g / gc * (REAL) (n1 - 1);
        i = (int) x;
        if (i < 0) i = 0;
        if (i > n1 - 2) i = n1 - 2;
    }
    else{
        const int n2 = ng - n1;
        x = (g - gc) / ((REAL)1.0 - gc) * (REAL) (n2 - 1);
        i = (int) x;
        if (i < 0) i = 0;
        if (i > n2 - 2) i = n2 - 2;
        x += (REAL) (n1 - 1);
        i += n1 - 1;
    }
    const REAL a = x - (REAL) i;
    __global const REAL *c = coef + i * 4;
    return c[0] + a * (c[1] + a * (c[2] + a * c[3]));
}

/* Normalised flux for one sample at projected distance z, which is out of transit beyond the
   last contact. The far side of the orbit is excluded by the caller's bounding box, so z is a
   separation here and never the negative flag the Keplerian solver used to return. */
inline REAL rr_flux_sample(const REAL z, const REAL k, const REAL istar,
                           const REAL gc, const int n1, const int ng, __global const REAL *coef){
    if (z >= (REAL)1.0 + k){
        return (REAL)1.0;
    }
    const REAL iplanet = ldm_lookup(z / ((REAL)1.0 + k), gc, n1, ng, coef);
    const REAL aplanet = circle_circle_intersection_area((REAL)1.0, k, z);
    return (istar - iplanet * aplanet) / istar;
}

/* The transit model for a population of parameter vectors.

   Global size (npv, npt). The parameter vectors are rows of `pv_pop` laid out as
   [k_0, ..., k_{nk-1}, t0, p, a, i, e, w]; the radius ratios are read from `ks`, which the host
   has broadcast to one per passband. */
__kernel void rr_flux(__global const REAL *times,    /* (npt,)                */
                      __global const REAL *ks,       /* (npv, npb)            */
                      __global const REAL *istar,    /* (npv, npb)            */
                      __global const REAL *gcs,      /* (npv, npb)            */
                      __global const int   *n1s,     /* (npv, npb)            */
                      __global const REAL *coef,     /* (npv, npb, ng - 2, 4) */
                      __global const int   *valid,   /* (npv,)                */
                      __global const REAL *xyc,      /* (npv, 2, 5)           */
                      __global const REAL *bbs,      /* (npv, nlc, 2)         */
                      const int ng,
                      __global const uint *lcids,    /* (npt,)                */
                      __global const uint *pbids,    /* (nlc,)                */
                      __global const REAL *pv_pop,   /* (npv, pv_length)      */
                      __global const uint *nss,      /* (nlc,)                */
                      __global const REAL *exptimes, /* (nlc,)                */
                      const uint pv_length, const uint nlc, const uint npb,
                      __global REAL *flux)           /* (npv, npt)            */
{
    const uint i_pv = get_global_id(0);
    const uint i_tm = get_global_id(1);
    const uint n_tm = get_global_size(1);
    const uint gid  = i_pv * n_tm + i_tm;

    if (!valid[i_pv]){
        flux[gid] = NAN;
        return;
    }

    const uint lcid = lcids[i_tm];
    const uint pbid = pbids[lcid];
    const uint ipp  = i_pv * npb + pbid;
    const uint nks  = pv_length - 6;

    __global const REAL *pv = &pv_pop[i_pv * pv_length + nks];
    __global const REAL *coef_pb = coef + ipp * (ng - 2) * 4;

    const uint ns = nss[lcid];
    const REAL exptime = exptimes[lcid];
    const REAL k = ks[ipp];
    const REAL gc = gcs[ipp];
    const int n1 = n1s[ipp];
    const REAL ist = istar[ipp];

    /* Fold the sample into the epoch around the transit centre. As in the Numba model the fold
       uses the unshifted sample time and the supersampling offsets are added to the centred
       time, so a sample never lands in a neighbouring epoch. */
    const REAL epoch = floor((times[i_tm] - pv[0] + (REAL)0.5 * pv[1]) / pv[1]);
    const REAL tcen  = times[i_tm] - (pv[0] + epoch * pv[1]);

    /* The Taylor expansion of the position is only valid near the transit, so anything outside
       the bounding box is out of transit by construction and must not be evaluated. */
    __global const REAL *bb = bbs + (i_pv * nlc + lcid) * 2;
    if (tcen < bb[0] || tcen > bb[1]){
        flux[gid] = (REAL)1.0;
        return;
    }

    __global const REAL *c = xyc + i_pv * 10;

    REAL f = (REAL)0.0;
    for (uint i = 1; i < ns + 1; i++){
        const REAL toffset = exptime * (((REAL) i - (REAL)0.5) / (REAL) ns - (REAL)0.5);
        f += rr_flux_sample(sep_c2(tcen + toffset, c), k, ist, gc, n1, ng, coef_pb);
    }
    flux[gid] = f / (REAL) ns;
}
