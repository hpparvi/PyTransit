__constant float TWO_PI = 2*M_PI_F;
__constant float HALF_PI = M_PI_2_F;

float mean_anomaly_offset(const float e, const float w){
    float offset = atan2(sqrt(1.0f-e*e) * sin(HALF_PI - w), e + cos(HALF_PI - w));
    return offset - e*sin(offset);
}

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

__kernel void calculate_weights(__global const float *ks, __global const float *ze, __global const float *gs, __global float *weights){
    uint ik = get_global_id(0);    // z vector index
    uint nk = get_global_size(0);  // z vector size
    uint ig = get_global_id(1);    // g vector index
    uint ng = get_global_size(1);  // g vector size
    uint iz = get_global_id(2);    // z vector index
    uint nz = get_global_size(2);  // z vector size
    uint gid  = ik*ng*nz + ig*nz + iz; // global linear index

    float k = ks[ik];
    float b = gs[ig] * (1.0f + k);
    float n = circle_circle_intersection_area(1.0f, k, b);

    if (iz==0){
        weights[gid] = circle_circle_intersection_area(ze[iz], k, b) / n;
    }
    else{
        float a0 = circle_circle_intersection_area(ze[iz-1], k, b);
        float a1 = circle_circle_intersection_area(ze[iz], k, b);
        weights[gid] = (a1 - a0) / n;
    }
}

__kernel void calculate_ldw(const uint nz, __global const float *ldp, __global const float *weights, __global float *ldw){
    int ipv = get_global_id(0);    // parameter vector index
    int npv = get_global_size(0);  // parameter vector size
    int ipb = get_global_id(1);    // passband index
    int npb = get_global_size(1);  // number of passbands
    int ig  = get_global_id(2);    // g vector index
    int ng  = get_global_size(2);  // g vector size
    uint gid  = ipv*npb*ng + ipb*ng + ig; // global linear index

    uint iw = ipv*ng*nz + ig*nz;
    uint il = ipv*npb*nz + ipb*nz;

    ldw[gid] = 0.0f;
    for(uint i=0; i<nz; i++){
        ldw[gid] += weights[iw + i] * ldp[il+i];
    }
}

float swiftmodel_flux(const float g, const float k, const float istar, __global const float *ldw, const float dg){
    if (g < 1.0f){
      float ag = g / dg;
      uint ig = floor(ag);
      ag -= ig;
      float iplanet = (1.0f - ag) * ldw[ig] + ag * ldw[ig+1];
      float aplanet = circle_circle_intersection_area(1.0f, k, g*(1.0f+k));
      return (istar - iplanet*aplanet) / istar;
    }
    else{
      return 1.0f;
    }
}


__kernel void ptmodel_z(__global const float *ks, const float istar, __global const float *zs, __global const int *pbids,
                        __global const float *ldw, const float dg, const int npb, const int ng, __global float *flux){
    int ipv = get_global_id(0);    // parameter vector index
    int npv = get_global_size(0);  // parameter vector size
    int iz  = get_global_id(1);    // z vector index
    int nz  = get_global_size(1);  // z vector size
    uint gid  = ipv*nz + iz;       // global linear index

    int ipb = pbids[iz];
    float k = ks[ipv];
    float g = zs[iz] / (1.0f+k);

    flux[gid] = swiftmodel_flux(g, k, istar, &ldw[ipv*npb*ng+ipb*ng], dg);
}

__kernel void swift_pop(__global const float *times, __global const float *istar, __global const float *ldw,
         const uint ng, const float dg, __global const uint *lcids, __global const uint *pbids,
         __global const float *pv_pop, __global const uint *nss, __global const float *exptimes,
         const uint pv_length, const uint nlc, const uint npb,
         __global float *flux){

      uint i_tm = get_global_id(1);    // time vector index
      uint n_tm = get_global_size(1);  // time vector size
      uint i_pv = get_global_id(0);    // parameter vector index
      uint n_pv = get_global_size(0);  // parameter vector population size
      uint gid  = i_pv*n_tm + i_tm;    // global linear index
      uint lcid = lcids[i_tm];         // light curve index
      uint pbid = pbids[lcid];         // passband index
      uint nks = pv_length - 6;

      __global const float *ks  = &pv_pop[i_pv*pv_length];
      __global const float *pv  = &pv_pop[i_pv*pv_length + nks];

      uint ns = nss[lcid];
      float exptime = exptimes[lcid];
      float toffset, z;
      float ma_offset = mean_anomaly_offset(pv[4], pv[5]);
      float k = ks[0];

      if (nks > 1){
          if (pbid < nks){
            k = ks[pbid];
          }
          else{
              flux[gid] = NAN;
              return;
          }
      }

      flux[gid] = 0.0f;
      for(int i=1; i<ns+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) ns - 0.5f);
        z = z_iter(times[i_tm]+toffset, pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], ma_offset, 1.0f);
        flux[gid] += swiftmodel_flux(z/(1.0f+k), k, istar[i_pv*npb+pbid], &ldw[i_pv*npb*ng+pbid*ng], dg);
      }
      flux[gid] /= (float) ns;
}