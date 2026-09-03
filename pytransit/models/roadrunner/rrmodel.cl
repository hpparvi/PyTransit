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
 *     the mean intensity with the split cubic lookup of `common.ldm_lookup`.
 *
 *  Everything is single precision.
 */

__constant float TWO_PI = 2*M_PI_F;
__constant float HALF_PI = M_PI_2_F;

float mean_anomaly_offset(const float e, const float w){
    float offset = atan2(sqrt(1.0f-e*e) * sin(HALF_PI - w), e + cos(HALF_PI - w));
    return offset - e*sin(offset);
}

/* Projected star-planet distance, or -1 on the far side of the orbit. */
float z_iter(const float t, const float t0, const float p, const float a,
             const float i, const float e,  const float w, const float ma_offset,
             const float eclipse){
    float Ma, ec, ect, Ea, sta, cta, Ta, z;

    Ma = fmod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI);
    ec = e*sin(Ma)/(1.f - e*cos(Ma));

    for(int i=0; i<15; i++){
        ect = ec;
        ec = e*sin(Ma+ec);
        if (fabs(ect-ec) < 1e-4){
            break;
        }
    }
    Ea  = Ma + ec;
    sta = sqrt(1.f-e*e) * sin(Ea)/(1.f-e*cos(Ea));
    cta = (cos(Ea)-e)/(1.f-e*cos(Ea));
    Ta  = atan2(sta, cta);

    if (eclipse * sign(sin(w+Ta)) > 0.0f){
        return a*(1.f-e*e)/(1.f+e*cos(Ta)) * sqrt(1.f - pow(sin(w+Ta)*sin(i), 2));
    }
    else{
        return -1.f;
    }
}


float circle_circle_intersection_area(float r1, float r2, float b){
    if (r1 < b - r2){
        return 0.0f;
    }
    else if (r1 >= b + r2){
        return M_PI_F * r2 * r2;
    }
    else if (b - r2 <= -r1){
        return M_PI_F * r1 * r1;
    }
    else{
        return r2*r2 * acos((b*b + r2*r2 - r1*r1) / (2 * b * r2)) +
               r1*r1 * acos((b*b + r1*r1 - r2*r2) / (2 * b * r1)) -
               0.5f * sqrt((-b + r2 + r1) * (b + r2 - r1) * (b - r2 + r1) * (b + r2 + r1));
    }
}


/* ------------------------------------------------------------------------------------------ */
/* Mean intensity under the planet by quadrature                                              */
/* ------------------------------------------------------------------------------------------ */

/* The grazing parameter at which the planet first touches the stellar limb. The mean intensity
   under the planet has a kink here, so the g table is split at it. */
inline float split_point(const float k){
    return (1.0f - k) / (1.0f + k);
}

/* Number of table nodes in the first segment, [0, gc]: proportional to its length, but at least
   four nodes in either segment so that a cubic can be fitted. Port of `common.g_nodes`. */
inline int first_segment_size(const float gc, const int ng){
    int n1 = (int) rint(ng * gc);
    if (n1 < 4) n1 = 4;
    if (ng - n1 < 4) n1 = ng - 4;
    return n1;
}

/* Grazing parameter of table node `ig`: two uniform segments meeting at gc, which is stored
   twice, as the last node of the first segment and the first node of the second. */
inline float g_node(const int ig, const float gc, const int n1, const int ng){
    if (ig < n1){
        return gc * (float) ig / (float) (n1 - 1);
    }
    else{
        return gc + (1.0f - gc) * (float) (ig - n1) / (float) (ng - n1 - 1);
    }
}

/* Angular extent of the planet disk at stellar radius z for a planet at separation b. */
inline float planet_angular_extent(const float z, const float b, const float k){
    if (b < 1e-7f){
        return (z < k) ? TWO_PI : 0.0f;
    }
    if (z <= k - b){
        return TWO_PI;
    }
    if (z < b - k || z > b + k){
        return 0.0f;
    }
    float c = (z * z + b * b - k * k) / (2.0f * z * b);
    c = clamp(c, -1.0f, 1.0f);
    return 2.0f * acos(c);
}

/* Cubic interpolation of a profile tabulated on the grid mu = (t0 + dt * i)**2, i = 0..n-1.
   Port of `common.profile_at`. */
inline float profile_at(const float m, const float t0, const float dt, __global const float *ldp, const int n){
    float x = (sqrt(m) - t0) / dt;
    if (x <= 0.0f) return ldp[0];
    if (x >= (float) (n - 1)) return ldp[n - 1];
    int i = (int) x - 1;
    if (i < 0) i = 0;
    else if (i > n - 4) i = n - 4;
    float u = x - (float) i;
    return (-(u - 1.0f) * (u - 2.0f) * (u - 3.0f) / 6.0f * ldp[i]
            + u * (u - 2.0f) * (u - 3.0f) / 2.0f * ldp[i + 1]
            - u * (u - 1.0f) * (u - 3.0f) / 2.0f * ldp[i + 2]
            + u * (u - 1.0f) * (u - 2.0f) / 6.0f * ldp[i + 3]);
}

/* Mu from z without the rounding of z*z past one producing a NaN. */
inline float mu_from_z(const float z){
    return sqrt(fmax(0.0f, 1.0f - z * z));
}

/* The mean intensity under the planet at every node of the grazing parameter table.

   Global size (npv, npb, ng). Port of `common.ldm_nodes` followed by `common.ldm_table`: the
   quadrature nodes and geometric factors for this node's geometry are generated and consumed
   on the fly. `rules` holds the Gauss-Legendre nodes and weights followed by the Gauss-Jacobi
   (0.5, 0) nodes and weights, nq of each. The work item of node 0 also stores the limb contact
   and the first segment size for the other kernels. A NaN radius ratio gives a NaN table. */
__kernel void calculate_ldm(__global const float *ks,       /* (npv, npb)      */
                            __global const float *ldp,      /* (npv, npb, nmu) */
                            __global const float *rules,    /* (4, nq)         */
                            const float pt0, const float pdt, const int nmu, const int nq,
                            __global float *gcs,            /* (npv, npb)      */
                            __global int   *n1s,            /* (npv, npb)      */
                            __global float *ldm)            /* (npv, npb, ng)  */
{
    const int ipv = get_global_id(0);
    const int ipb = get_global_id(1);
    const int npb = get_global_size(1);
    const int ig  = get_global_id(2);
    const int ng  = get_global_size(2);
    const int ipp = ipv * npb + ipb;

    const float k  = ks[ipp];
    const float gc = split_point(k);

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

    __global const float *prof = ldp + ipp * nmu;
    __global const float *t_gl = rules;
    __global const float *w_gl = rules + nq;
    __global const float *t_gj = rules + 2 * nq;
    __global const float *w_gj = rules + 3 * nq;

    const float g = g_node(ig, gc, n1, ng);

    /* At g = 1 the overlap vanishes and the mean intensity is the profile at the limb. The Numba
       code evaluates the geometry a hair inside the limb instead, where every quadrature node
       falls below the first profile node and `profile_at` returns the same value; in single
       precision that hair would round away, leaving the mean 0/0. */
    if (g >= 1.0f){
        ldm[ipp * ng + ig] = prof[0];
        return;
    }

    const float b = g * (1.0f + k);

    float num = 0.0f;
    float den = 0.0f;
    float c, h, m, z, sq, w;

    if (b + k <= 1.0f){
        if (b < k){
            /* z in [0, k - b]: theta = 2 pi. Integrated in mu with Gauss-Legendre. */
            const float mu_mid = mu_from_z(k - b);
            c = 0.5f * (mu_mid + 1.0f);
            h = 0.5f * (1.0f - mu_mid);
            for (int q = 0; q < nq; q++){
                m = c + h * t_gl[q];
                z = mu_from_z(m);
                w = w_gl[q] * planet_angular_extent(z, b, k) * m * h;
                num += w * profile_at(m, pt0, pdt, prof, nmu);
                den += w;
            }
            /* z in [k - b, b + k] with z = (k - b) + s**2: regular at s = 0, sqrt zero at the end. */
            c = h = 0.5f * sqrt(2.0f * b);
            for (int q = 0; q < nq; q++){
                sq = c + h * t_gj[q];
                z = k - b + sq * sq;
                m = mu_from_z(z);
                w = w_gj[q] * planet_angular_extent(z, b, k) * z * 2.0f * sq * h;
                num += w * profile_at(m, pt0, pdt, prof, nmu);
                den += w;
            }
        }
        else{
            /* z in [b - k, b + k] with z = (b + k) - s**2: the limb end is regular, and the sqrt
               zero of theta at z = b - k sits at s_max, where the Jacobi weight takes it. */
            c = h = 0.5f * sqrt(2.0f * k);
            for (int q = 0; q < nq; q++){
                sq = c + h * t_gj[q];
                z = b + k - sq * sq;
                m = mu_from_z(z);
                w = w_gj[q] * planet_angular_extent(z, b, k) * z * 2.0f * sq * h;
                num += w * profile_at(m, pt0, pdt, prof, nmu);
                den += w;
            }
        }
    }
    else{
        /* z in [b - k, 1]: integrated in mu from the limb, where the profile is regular, to
           mu_lo, where theta has its sqrt zero. The distance of the inner contact from the limb,
           1 - (b - k) = (1 - g)(1 + k), is formed without the cancellation of b - k near 1. */
        const float dlimb = (1.0f - g) * (1.0f + k);
        const float mu_lo = sqrt(dlimb * (2.0f - dlimb));
        c = h = 0.5f * mu_lo;
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
__kernel void calculate_coefficients(__global const float *ldm,   /* (npv, npb, ng)         */
                                     __global const int   *n1s,   /* (npv, npb)             */
                                     __global const float *cm,    /* (3, 4, 4)              */
                                     const int ng,
                                     __global float *coef)        /* (npv, npb, ng - 2, 4)  */
{
    const int ipv = get_global_id(0);
    const int ipb = get_global_id(1);
    const int npb = get_global_size(1);
    const int ii  = get_global_id(2);
    const int ipp = ipv * npb + ipb;

    const int n1 = n1s[ipp];
    __global const float *tab = ldm + ipp * ng;
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

    __global float *c = coef + (ipp * (ng - 2) + ii) * 4;
    for (int r = 0; r < 4; r++){
        __global const float *row = cm + (s * 4 + r) * 4;
        c[r] = row[0] * tab[j] + row[1] * tab[j + 1] + row[2] * tab[j + 2] + row[3] * tab[j + 3];
    }
}

/* The mean intensity under the planet at grazing parameter g from the split cubic table.

   Port of `common.ldm_lookup`. The NaN test is not decorative: NaN compares false against
   both range tests, and `(int) NAN` is INT_MIN on at least NVIDIA, which would index the table
   far outside its buffer. */
inline float ldm_lookup(const float g, const float gc, const int n1, const int ng, __global const float *coef){
    if (isnan(g) || isnan(gc)) return NAN;
    if (g < 0.0f) return NAN;
    if (g > 1.0f) return 0.0f;
    float x;
    int i;
    if (g < gc){
        x = g / gc * (float) (n1 - 1);
        i = (int) x;
        if (i < 0) i = 0;
        if (i > n1 - 2) i = n1 - 2;
    }
    else{
        const int n2 = ng - n1;
        x = (g - gc) / (1.0f - gc) * (float) (n2 - 1);
        i = (int) x;
        if (i < 0) i = 0;
        if (i > n2 - 2) i = n2 - 2;
        x += (float) (n1 - 1);
        i += n1 - 1;
    }
    const float a = x - (float) i;
    __global const float *c = coef + i * 4;
    return c[0] + a * (c[1] + a * (c[2] + a * c[3]));
}

/* Normalised flux for one sample at projected distance z. A negative z is the far side of the
   orbit, and z beyond the last contact is out of transit. */
inline float rr_flux_sample(const float z, const float k, const float istar,
                            const float gc, const int n1, const int ng, __global const float *coef){
    if (z < 0.0f || z >= 1.0f + k){
        return 1.0f;
    }
    const float iplanet = ldm_lookup(z / (1.0f + k), gc, n1, ng, coef);
    const float aplanet = circle_circle_intersection_area(1.0f, k, z);
    return (istar - iplanet * aplanet) / istar;
}

/* The transit model for a population of parameter vectors.

   Global size (npv, npt). The parameter vectors are rows of `pv_pop` laid out as
   [k_0, ..., k_{nk-1}, t0, p, a, i, e, w]; the radius ratios are read from `ks`, which the host
   has broadcast to one per passband. */
__kernel void rr_flux(__global const float *times,       /* (npt,)                */
                      __global const float *ks,          /* (npv, npb)            */
                      __global const float *istar,       /* (npv, npb)            */
                      __global const float *gcs,         /* (npv, npb)            */
                      __global const int   *n1s,         /* (npv, npb)            */
                      __global const float *coef,        /* (npv, npb, ng - 2, 4) */
                      __global const int   *valid,       /* (npv,)                */
                      const int ng,
                      __global const uint *lcids,        /* (npt,)                */
                      __global const uint *pbids,        /* (nlc,)                */
                      __global const float *pv_pop,      /* (npv, pv_length)      */
                      __global const uint *nss,          /* (nlc,)                */
                      __global const float *exptimes,    /* (nlc,)                */
                      const uint pv_length, const uint nlc, const uint npb,
                      __global float *flux)              /* (npv, npt)            */
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

    __global const float *pv = &pv_pop[i_pv * pv_length + nks];
    __global const float *coef_pb = coef + ipp * (ng - 2) * 4;

    const uint ns = nss[lcid];
    const float exptime = exptimes[lcid];
    const float ma_offset = mean_anomaly_offset(pv[4], pv[5]);
    const float k = ks[ipp];
    const float gc = gcs[ipp];
    const int n1 = n1s[ipp];
    const float ist = istar[ipp];

    float f = 0.0f;
    for (uint i = 1; i < ns + 1; i++){
        const float toffset = exptime * (((float) i - 0.5f) / (float) ns - 0.5f);
        const float z = z_iter(times[i_tm] + toffset, pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], ma_offset, 1.0f);
        f += rr_flux_sample(z, k, ist, gc, n1, ng, coef_pb);
    }
    flux[gid] = f / (float) ns;
}
