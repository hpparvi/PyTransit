float ma_uniform(float z, float k){
    float flux, kap0, kap1, lambdae;

    if(fabs(k - 0.5f) < 0.001f){
        k = 0.5f;
    }

    if ((z < 0.0f) || (z > 1.0f + k)){
        flux = 1.0f;
    }
    else if ((z > fabs(1.0f - k)) && (z < 1.0f + k)){
        if (k < 0.002f){
            flux = 1.0f;
        }
        else{
            kap1 = acos(min((1.0f - k*k + z*z) / 2.0f / z, 1.0f));
            kap0 = acos(min((k*k + z*z - 1.0f) / 2.0f / k / z, 1.0f));
            lambdae = k * k * kap0 + kap1;
            lambdae = (lambdae - 0.5f * sqrt(max(4.0f*z*z - pown(1.0f + z*z - k*k, 2), 0.0f))) / M_PI_F;
            flux = 1.0f - lambdae;
        }
    }
    else if (z < 1.0f - k){
        flux = 1.0f - k*k;
    }
    else if ((k > 1.0f) && (z < k - 1.0f)){
        flux = 0.0f;
    }
    return flux;
}

__kernel void uniform_eccentric_pop(__global const float *times, __global const uint *lcids, __global const uint *pbids,
         __global const float *pv_pop, __global const uint *nss, __global const float *exptimes, __global const float *vajs,
         const uint pv_length, const uint nlc, const uint npb,
         __global float *flux){

      uint i_tm = get_global_id(1);    // time vector index
      uint n_tm = get_global_size(1);  // time vector size
      uint i_pv = get_global_id(0);    // parameter vector index
      uint n_pv = get_global_size(0);  // parameter vector population size
      uint gid  = i_pv*n_tm + i_tm;    // global linear index
      uint lcid = lcids[i_tm];         // light curve index
      uint pbid = pbids[lcid];         // passband index
      uint nks = pv_length - 6;        // Number of radius ratios

      __global const float *ks = &pv_pop[i_pv*pv_length];
      __global const float *pv = &pv_pop[i_pv*pv_length + nks];
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
            flux[gid] += ma_uniform(z, k);
          }
          flux[gid] /= (float) ns;
      }
}