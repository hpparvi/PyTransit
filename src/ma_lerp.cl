__constant float TWO_PI = 2*M_PI_F;
__constant float HALF_PI = M_PI_2_F;

float mean_anomaly_offset(const float e, const float w){
    float offset = atan2(sqrt(1.0f-e*e) * sin(HALF_PI - w), e + cos(HALF_PI - w));
    return offset - e*sin(offset);
}

float mean_anomaly(const float t, const float t0, const float p, const float offset){
    return fmod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI);
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

float z_newton(const float t, const float t0, const float p, const float a,
               const float i, const float  e, const float w, const float ma_offset,
               const float eclipse){
    float Ma = fmod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI);
    float Ea = Ma;
    float err = 0.05f;
    int k = 0;
    while ((fabs(err) > 1.f-8) & (k<1000)){
        err   = Ea - e*sin(Ea) - Ma;
        Ea = Ea - err/(1.f-e*cos(Ea));
        k += 1;
    }
    float sta = sqrt(1.f-e*e) * sin(Ea)/(1.f-e*cos(Ea));
    float cta = (cos(Ea)-e)/(1.f-e*cos(Ea));
    float Ta  = atan2(sta, cta);

    if (eclipse * sign(sin(w+Ta)) > 0.0f){
        return a*(1.f-e*e)/(1.f+e*cos(Ta)) * sqrt(1.f - pow(sin(w+Ta)*sin(i), 2));
    }
    else{
        return -1.f;
    }
}


float z_circular(const float t, const float t0, const float p, const float a, const float i, const float eclipse){
    // eclipse should be -1.f for eclipse and 1.0 for transit
    float cosph = cos(TWO_PI*(t-t0)/p);
    float sini = sin(i);
    if (eclipse * sign(cosph) > 0.0f){
        return sign(cosph) * a*sqrt(1.0f - cosph*cosph*sini*sini);
    }
    else{
        return -1.f;
    }
}

float eval_ma_ip(const float z, __global const float *u,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 const float k, const float k0, const float k1,
		 const uint nk, const uint nz, const float dk, const float dz){

  float ak, az, omega, ed, le, ld;
  uint ik, iz, it;
  uint i00,i10,i01,i11;

  const float u1 = u[0];
  const float u2 = u[1];

  if (sign(z) < 0.f){
    return 1.0f;
  }

  if (k<k0 | k>k1 | z >= 1.f+k){
    return 1.0f;
  }
  else {
    omega = 1.f - u1/3.f - u2/6.f;

    ik = floor((k-k0)/dk);
    ak = (k-(k0+ik*dk))/dk;
    iz = floor(z/dz);
    az = (z-(iz*dz))/dz;

    i00 = (ik  )*nz + iz;
    i10 = (ik+1)*nz + iz;
    i01 = (ik  )*nz + (iz+1);
    i11 = (ik+1)*nz + (iz+1);

    ed = edt[i00]*(1.f-ak)*(1.f-az)
       + edt[i10]*(1.f-az)*ak
       + edt[i01]*(1.f-ak)*az
       + edt[i11]*ak*az;

    le = let[i00]*(1.f-ak)*(1.f-az)
       + let[i10]*(1.f-az)*ak
       + let[i01]*(1.f-ak)*az
       + let[i11]*ak*az;

    ld = ldt[i00]*(1.f-ak)*(1.f-az)
       + ldt[i10]*(1.f-az)*ak
       + ldt[i01]*(1.f-ak)*az
       + ldt[i11]*ak*az;

    return 1.f-((1.f-u1-2.f*u2)*le + (u1+2.f*u2)*ld + u2*ed)/omega;
  }
}

__kernel void ma_circular(__global const float *times, __global const float *pv, __global const float *u,
        __global const float *edt, __global const float *let, __global const float *ldt,
        const int nss, const float exptime, const float k0, const float k1,
        const uint nk, const uint nz, const float dk, const float dz, __global float *flux){
      int gid = get_global_id(0);
      float toffset = 0.0f;
      float ma_offset = mean_anomaly_offset(pv[5], pv[6]);
      float z = 0.0f;

      flux[gid] = 0.0f;
      for(int i=1; i<nss+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) nss - 0.5f);
        z = z_circular(times[gid]+toffset, pv[1], pv[2], pv[3], pv[4], 1.0f);
        flux[gid] += eval_ma_ip(z, u, edt, let, ldt, pv[0], k0, k1, nk, nz, dk, dz);
      }
      flux[gid] /= (float) nss;
}

__kernel void ma_eccentric(__global const float *times, __global const float *pv, __global const float *u,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 const int nss, const float exptime, const float k0, const float k1,
		 const uint nk, const uint nz, const float dk, const float dz, __global float *flux){
      int gid = get_global_id(0);
      float toffset = 0.0f;
      float ma_offset = mean_anomaly_offset(pv[5], pv[6]);
      float z = 0.0f;

      flux[gid] = 0.0f;
      for(int i=1; i<nss+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) nss - 0.5f);
        z = z_iter(times[gid]+toffset, pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], ma_offset, 1.0f);
//        z = z_newton(times[gid]+toffset, pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], ma_offset, 1.0f);
        flux[gid] += eval_ma_ip(z, u, edt, let, ldt, pv[0], k0, k1, nk, nz, dk, dz);
      }
      flux[gid] /= (float) nss;
}


__kernel void ma_eccentric_pop(__global const float *times, __global const float *pv_pop, __global const float *ldc_pop,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 const int nss, const float exptime, const float k0, const float k1,
		 const uint nk, const uint nz, const float dk, const float dz, const uint pv_length, __global float *flux){

      uint i_tm = get_global_id(1);    // time vector index
      uint n_tm = get_global_size(1);  // time vector size
      uint i_pv = get_global_id(0);    // parameter vector index
      uint n_pv = get_global_size(0);  // parameter vector population size
      uint gid  = i_pv*n_tm + i_tm;    // global linear index

      __global float *pv  = &pv_pop[i_pv*pv_length];
      __global float *ldc = &ldc_pop[2*i_pv];

      float toffset, z;
      float ma_offset = mean_anomaly_offset(pv[5], pv[6]);

      flux[gid] = 0.0f;
      for(int i=1; i<nss+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) nss - 0.5f);
        z = z_iter(times[i_tm]+toffset, pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], ma_offset, 1.0f);
        flux[gid] += eval_ma_ip(z, ldc, edt, let, ldt, pv[0], k0, k1, nk, nz, dk, dz);
      }
      flux[gid] /= (float) nss;
}

__kernel void ma_eccentric_pop_ttv(__global const float *times, __global const float *pv_pop,
         __global const float *ldc_pop, __global const int *tid, const int ntr,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 const int nss, const float exptime, const float k0, const float k1,
		 const uint nk, const uint nz, const float dk, const float dz, const uint pv_length, __global float *flux){

      uint i_tm = get_global_id(1);    // time vector index
      uint n_tm = get_global_size(1);  // time vector size
      uint i_pv = get_global_id(0);    // parameter vector index
      uint n_pv = get_global_size(0);  // parameter vector population size
      uint gid  = i_pv*n_tm + i_tm;    // global linear index

      __global float *pv  = &pv_pop[i_pv*pv_length];
      __global float *ldc = &ldc_pop[2*i_pv];
      __global float *pvo = &pv[1+ntr];

      float toffset, z;
      float k  = pv[0];
      float tc = pv[1 + tid[i_tm]];
      float ma_offset = mean_anomaly_offset(pvo[3], pvo[4]);

      flux[gid] = 0.0f;
      for(int i=1; i<nss+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) nss - 0.5f);
        z = z_circular(times[i_tm]+toffset, tc, pvo[0], pvo[1], pvo[2], 1.0f);
        flux[gid] += eval_ma_ip(z, ldc, edt, let, ldt, k, k0, k1, nk, nz, dk, dz);
      }
      flux[gid] /= (float) nss;
}

__kernel void ma_eccentric_pop_tdv(__global const float *times, __global const float *pv_pop,
         __global const float *ldc_pop, __global const int *tid, const int ntr,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 const int nss, const float exptime, const float k0, const float k1,
		 const uint nk, const uint nz, const float dk, const float dz, const uint pv_length, __global float *flux){

      uint i_tm = get_global_id(1);    // time vector index
      uint n_tm = get_global_size(1);  // time vector size
      uint i_pv = get_global_id(0);    // parameter vector index
      uint n_pv = get_global_size(0);  // parameter vector population size
      uint gid  = i_pv*n_tm + i_tm;    // global linear index

      __global const float *pv  = &pv_pop[i_pv*pv_length];
      __global const float *ldc = &ldc_pop[2*i_pv];

      float toffset, z;
      float k   = pv[0];
      float tc  = pv[1 + tid[i_tm]];
      float pr  = pv[1 + ntr + tid[i_tm]];
      float as  = pv[1 + 2*ntr];
      float inc = pv[1 + 2*ntr + 1];

      flux[gid] = 0.0f;
      for(int i=1; i<nss+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) nss - 0.5f);
        z = z_circular(times[i_tm]+toffset, tc, pr, as, inc, 1.0f);
        flux[gid] += eval_ma_ip(z, ldc, edt, let, ldt, k, k0, k1, nk, nz, dk, dz);
      }
      flux[gid] /= (float) nss;
}