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
        flux[gid] += ma_uniform(z, k);
      }
      flux[gid] /= (float) ns;
}