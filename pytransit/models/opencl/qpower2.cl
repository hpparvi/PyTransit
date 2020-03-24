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

float q1n(float z, float p, float c, float alpha, float g, float I_0){
    float s = 1.0f-z*z;
    float c0 = (1.0f-c+c*pow(s,g));
    float c2 = 0.5f*alpha*c*pow(s, g-2.0f)*((alpha-1.0f)*z*z - 1.f);
    return 1.f - I_0*M_PI_F*p*p*(c0 + 0.25f*p*p*c2 - 0.125f*alpha*c*p*p*pow(s, g-1));
}

float q2n(float z, float p, float c, float alpha, float g, float I_0){
    float d = (z*z - p*p + 1.0f)/(2.0f*z);
    float ra = 0.5f*(z-p+d);
    float rb = 0.5f*(1+d);
    float sa = 1.0f - ra*ra;
    float sb = 1.0 - rb*rb;
    float q = (z-d)/p;
    float w2 = p*p-pown(d-z, 2);
    float w = sqrt(w2);
    float b0 = 1.0f - c + c*pow(sa, g);
    float b1 = -alpha*c*ra*pow(sa, g-1.0f);
    float b2 = 0.5f*alpha*c*pow(sa, g-2.0f)*((alpha-1)*ra*ra-1.0f);
    float a0 = b0 + b1*(z-ra) + b2*pown(z-ra, 2);
    float a1 = b1+2*b2*(z-ra);
    float aq = acos(q);
    float J1 = (a0*(d-z)-(2.0f/3.0f)*a1*w2 + 0.25f*b2*(d-z)*(2*pown(d-z, 2)-p*p))*w + (a0*p*p + 0.25f*b2*pown(p, 4))*aq;
    float J2 = alpha*c*pow(sa, g-1.0f)*pown(p, 4)*(0.125f*aq + (1.0f/12.0f)*q*(q*q-2.5f)*sqrt(1.0f-q*q) );
    float d0 = 1 - c + c*pow(sb, g);
    float d1 = -alpha*c*rb* pow(sb, g-1.0f);
    float K1 = ((d0-rb*d1)*acos(d) + ((rb*d+(2.0f/3.0f)*(1.0f-d*d))*d1 - d*d0)*sqrt(1.0f-d*d));
    float K2 = (1.0f/3.0f)*c*alpha*pow(sb, g+0.5f)*(1.0f-d);
    return 1.0f - I_0*(J1 - J2 + K1 - K2);
}

float eval_qpower2(float z, float k, __global const float *u){
    float I_0 = (u[1] + 2.0f) / (M_PI_F * (u[1] - u[0] * u[1] + 2.0f));
    float g = 0.5f * u[1];
    float flux = 1.0f;
    if (z >= 0.0f){
        if (z <= 1.0f - k){
            flux = q1n(z, k, u[0], u[1], g, I_0);
        }
        else if(fabs(z - 1.0f) < k){
            flux = q2n(z, k, u[0], u[1], g, I_0);
        }
    }
    return flux;
}


__kernel void qpower2_eccentric_pop(__global const float *times, __global const uint *lcids, __global const uint *pbids,
         __global const float *pv_pop, __global const float *ldc_pop,
		 __global const uint *nss, __global const float *exptimes,
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
        flux[gid] += eval_qpower2(z, k, ldc);
      }
      flux[gid] /= (float) ns;
}
