__kernel void ma(__global const float *z, __global const float *u,
		 __global const float *edt, __global const float *let, __global const float *ldt,
		 const float k, const float k0, const float k1, 
		 const uint nk, const uint nz, const float dk, const float dz,
		 __global float *flux){

  float ak, az, omega, ed, le, ld;
  uint ik, iz, it;
  uint i00,i10,i01,i11;

  uint npt = get_global_size(0);
  uint npb = get_global_size(1);
  uint ipt = get_global_id(0);
  uint ipb = get_global_id(1);

  int oid = ipt*npb + ipb;

  const float u1 = u[  2*ipb];
  const float u2 = u[1+2*ipb];

  //const float u1 = u[ipb];
  //const float u2 = u[ipb + 2*npb];

  if (k<k0 | k>k1 | z[ipt] >= 1.f+k){
    flux[oid] = 1.0f;
  }
  else {
    omega = 1.f - u1/3.f - u2/6.f;

    ik = floor((k-k0)/dk);
    ak = (k-(k0+ik*dk))/dk;
    iz = floor(z[ipt]/dz);
    az = (z[ipt]-(iz*dz))/dz;

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

    flux[oid] = 1.f-((1.f-u1-2.f*u2)*le + (u1+2.f*u2)*ld + u2*ed)/omega;
  }
}
