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

float ptmodel_flux(const float g, const float k, const float istar, __global const float *ldw, const float dg){
    if (g < 1.0f){
      float ag = g / dg;
      uint ig = floor(ag);
      ag -= ig;
      float iplanet = (1.0f - ag) * ldw[ig] + ag * ldw[ig+1];
      float aplanet = circle_circle_intersection_area(1.0f, k, g*(1.0f+k));
      return (M_PI_F*istar - iplanet*aplanet) / (M_PI_F*istar);
    }
    else{
      return 1.0f;
    }
}

__kernel void calculate_weights(const float k, __global const float *ze, __global const float *gs, __global float *weights){
    uint iz = get_global_id(1);    // z vector index
    uint nz = get_global_size(1);  // z vector size
    uint ig = get_global_id(0);    // g vector index
    uint ng = get_global_size(0);  // g vector size
    uint gid  = ig*nz + iz;        // global linear index

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
    uint gid = get_global_id(0);   // g vector index
    ldw[gid] = 0.0f;
    for(int i=0; i<nz; i++){
        ldw[gid] += weights[gid*nz+i] * ldp[i];
    }
}


__kernel void ptmodel_z(const float k, const float istar, __global const float *zs, __global const float *ldw, const float dg, __global float *flux){
    int gid = get_global_id(0);
    float g = zs[gid] / (1.0+k);
    flux[gid] = ptmodel_flux(g, k, istar, ldw, dg);
}