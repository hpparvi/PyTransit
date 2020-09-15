float z_circular(const float t, const float t0, const float p, const float a, const float i, const float eclipse){
    // ``eclipse`` should be -1.f for an eclipse and 1.0 for a transit
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


__kernel void ma_eccentric_pop(__global const float *times, __global const uint *lcids, __global const uint *pbids,
         __global const float *pv_pop, __global const float *ldc_pop,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 __global const uint *nss, __global const float *exptimes,  __global const float *vajs,
         float k0, float k1, uint nk, uint nz, float dk, float dz,
         uint pv_length, uint nlc, uint npb,
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
    __global const float *ldc = &ldc_pop[2*npb*i_pv + 2*pbid];
    __global const float *tt = &vajs[i_pv*9];

    uint ns = nss[lcid];
    float exptime = exptimes[lcid];
    float toffset, z;
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

    float half_window_width = fmax(0.125f, (2.0f + k) / tt[1]);
    float epoch = floor((times[i_tm] - pv[0] + 0.5f * pv[1]) / pv[1]);
    float t = times[i_tm] - (pv[0] + epoch * pv[1]);

    if (fabs(t) > half_window_width){
        flux[gid] = 1.0f;
    }
    else{
        flux[gid] = 0.0f;
        for(int i=1; i<ns+1; i++){
            toffset = exptime * (((float) i - 0.5f)/ (float) ns - 0.5f);
            z = z_taylor_tc(t+toffset, tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], tt[6], tt[7], tt[8]);
            flux[gid] += eval_ma_ip(z, ldc, edt, let, ldt, k, k0, k1, nk, nz, dk, dz);
        }
        flux[gid] /= (float) ns;
    }
}


__kernel void ma_eccentric_pop_ttv(__global const float *times, __global const uint *lcids, __global const uint *pbids,
         __global const float *pv_pop, __global const float *ldc_pop,
		 __global const float *edt, __global const float *let, __global const float *ldt,
         __global const uint *nss, __global const float *exptimes,
        float k0, float k1, uint nk, uint nz, float dk, float dz,
        uint pv_length, uint nlc, uint npb,
        __global float *flux){

      uint i_tm = get_global_id(1);    // time vector index
      uint n_tm = get_global_size(1);  // time vector size
      uint i_pv = get_global_id(0);    // parameter vector index
      uint n_pv = get_global_size(0);  // parameter vector population size
      uint gid  = i_pv*n_tm + i_tm;    // global linear index
      uint lcid = lcids[i_tm];         // light curve index
      uint pbid = pbids[lcid];         // passband index

      __global const float *pv  = &pv_pop[i_pv*pv_length];
      __global const float *ldc = &ldc_pop[2*i_pv];
      __global const float *pvo = &pv[1+nlc];

      float toffset, z;
      uint ns = nss[lcid];
      float exptime = exptimes[lcid];
      float k  = pv[0];
      float tc = pv[1 + lcids[i_tm]];

      flux[gid] = 0.0f;
      for(int i=1; i<ns+1; i++){
        toffset = exptime * (((float) i - 0.5f)/ (float) ns - 0.5f);
        z = z_circular(times[i_tm]+toffset, tc, pvo[0], pvo[1], pvo[2], 1.0f);
        flux[gid] += eval_ma_ip(z, ldc, edt, let, ldt, k, k0, k1, nk, nz, dk, dz);
      }
      flux[gid] /= (float) ns;
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