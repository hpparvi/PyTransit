__constant float TWO_PI = 2*M_PI_F;
__constant float HALF_PI = M_PI_2_F;

float mean_anomaly_offset(float e, float w){
    float offset = atan2(sqrt(1.0f-e*e) * sin(HALF_PI - w), e + cos(HALF_PI - w));
    return offset - e*sin(offset);
}

float ta_newton(float t, float t0, float p, float e, float w, float ma_offset){
    float ma, ea, err, sta, cta;
    ma = fmod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI);

    ea = ma;
    err = 0.05f;
    for(int i=0; i<100; i++){
        err = ea - e*sin(ea) - ma;
        ea = ea - err/(1.0-e*cos(ea));
        if (fabs(err) < 1e-4){
            break;
        }
    }

    sta = sqrt(1.f-e*e) * sin(ea)/(1.f-e*cos(ea));
    cta = (cos(ea)-e)/(1.f-e*cos(ea));
    return atan2(sta, cta);
}

void vajs_from_paiew(float p, float a, float i, float e, float w, __global float *cc){
    float f0, f1, f2, f3, f4, f5, f6;
    float r0, r1, r2, r3, r4, r5, r6;
    float x0, x1, x2, x3, x4, x5, x6;
    float y0, y1, y2, y3, y4, y5, y6;
    float g, b, c, d;

    float dt = 0.02f;
    float ae = a*(1.0f - e*e);
    float ci = cos(i);

    float mao = mean_anomaly_offset(e, w);

    f0 = ta_newton(-3*dt, 0.0f, p, e, w, mao);
    f1 = ta_newton(-2*dt, 0.0f, p, e, w, mao);
    f2 = ta_newton(  -dt, 0.0f, p, e, w, mao);
    f3 = ta_newton( 0.0f, 0.0f, p, e, w, mao);
    f4 = ta_newton(   dt, 0.0f, p, e, w, mao);
    f5 = ta_newton( 2*dt, 0.0f, p, e, w, mao);
    f6 = ta_newton( 3*dt, 0.0f, p, e, w, mao);

    r0 = ae/(1.0f + e*cos(f0));
    r1 = ae/(1.0f + e*cos(f1));
    r2 = ae/(1.0f + e*cos(f2));
    r3 = ae/(1.0f + e*cos(f3));
    r4 = ae/(1.0f + e*cos(f4));
    r5 = ae/(1.0f + e*cos(f5));
    r6 = ae/(1.0f + e*cos(f6));

    x0 = -r0*cos(w + f0);
    x1 = -r1*cos(w + f1);
    x2 = -r2*cos(w + f2);
    x3 = -r3*cos(w + f3);
    x4 = -r4*cos(w + f4);
    x5 = -r5*cos(w + f5);
    x6 = -r6*cos(w + f6);

    y0 = -r0*sin(w + f0)*ci;
    y1 = -r1*sin(w + f1)*ci;
    y2 = -r2*sin(w + f2)*ci;
    y3 = -r3*sin(w + f3)*ci;
    y4 = -r4*sin(w + f4)*ci;
    y5 = -r5*sin(w + f5)*ci;
    y6 = -r6*sin(w + f6)*ci;

    cc[0] = y3;

    // First time derivative of position: velocity
    // -------------------------------------------
    g = 1.0f/60.f; b = 9.0f/60.0f; c = 45.0f/60.0f;

    cc[1] = (g*(x6 - x0) + b*(x1 - x5) + c*(x4 - x2))/dt;
    cc[2] = (g*(y6 - y0) + b*(y1 - y5) + c*(y4 - y2))/dt;

    // Second time derivative of position: acceleration
    // ------------------------------------------------
    g = 1.0f/90.0f; b = 3.0f/20.0f; c = 3.0f/2.0f; d = 49.0f/18.0f;
    cc[3] = (g*(x0 + x6) - b*(x1 + x5) + c*(x2 + x4) - d*x3)/pown(dt, 2);
    cc[4] = (g*(y0 + y6) - b*(y1 + y5) + c*(y2 + y4) - d*y3)/pown(dt, 2);

    // Third time derivative of position: jerk
    // ---------------------------------------
    g = 1.0f/8.0f; b = 1.0f; c = 13.0f/8.0f;
    cc[5] = (g*(x0 - x6) + b*(x5 - x1) + c*(x2 - x4))/pown(dt, 3);
    cc[6] = (g*(y0 - y6) + b*(y5 - y1) + c*(y2 - y4))/pown(dt, 3);

    // Fourth time derivative of position: snap
    // ----------------------------------------
    g = 1.0f/6.0f; b = 2.0f; c = 13.0f/2.0f; d = 28.0f/3.0f;
    cc[7] = (-g*(x0 + x6) + b*(x1 + x5) - c*(x2 + x4) + d*x3)/pown(dt, 4);
    cc[8] = (-g*(y0 + y6) + b*(y1 + y5) - c*(y2 + y4) + d*y3)/pown(dt, 4);
}

__kernel void vajs_from_paiew_v(int pv_length, __global const float *pv_pop, __global float *vajs){
    uint ipv = get_global_id(0);     // parameter vector index
    uint npv = get_global_size(0);   // parameter vector population size
    uint nks = pv_length - 6;        // Number of radius ratios

    __global const float *pv = &pv_pop[ipv*pv_length + nks];
    __global float *c = &vajs[ipv*9];

    vajs_from_paiew(pv[1], pv[2], pv[3], pv[4], pv[5], c);
}

float z_taylor_tc(const float t, const float y0, const float vx, const float vy,
               const float ax, const float ay, const float jx, const float jy, const float sx, const float sy){
    float epoch, t1, t2, t3, t4;
    float2 pos;

    t1 = t;
    t2 = t1 * t1;
    t3 = t2 * t1;
    t4 = t3 * t1;
    pos.x =      vx * t1 + 0.5f * ax * t2 + jx * t3 / 6.0f + sx * t4 / 24.0f;
    pos.y = y0 + vy * t1 + 0.5f * ay * t2 + jy * t3 / 6.0f + sy * t4 / 24.0f;
    return length(pos);
}
