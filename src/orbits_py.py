from numpy import pi, arctan2, sin, cos, sqrt, sign, copysign, mod, zeros_like, zeros, linspace, floor, arcsin
from numba import jit, njit, prange

HALF_PI = 0.5 * pi
TWO_PI = 2.0 * pi

# Utilities
# =========

cache = False

@njit("f8(f8, f8)", cache=cache)
def mean_anomaly_offset(e, w):
    mean_anomaly_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    mean_anomaly_offset -= e*sin(mean_anomaly_offset)
    return mean_anomaly_offset

@njit("f8(f8, f8, f8, f8, f8)", cache=cache)
def z_from_ta_s(Ta, a, i, e, w):
    z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
    z *= copysign(1.0, sin(w+Ta))
    return z

@njit("f8[:](f8[:], f8, f8, f8, f8)", parallel=True)
def z_from_ta_v(Ta, a, i, e, w):
    z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
    z *= sign(1.0, sin(w+Ta))
    return z

@njit
def rclip(v, vmin, vmax):
    return min(max(v, vmin), vmax)

@njit
def iclip(v, vmin, vmax):
    return int(min(max(v, vmin), vmax))

# Mean Anomaly
# ============

@njit(cache=cache)
def mean_anomaly(t, t0, p, e, w):
    offset = mean_anomaly_offset(e, w)
    Ma = mod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI)
    return Ma

@njit(parallel=True)
def mean_anomaly_p(t, t0, p, e, w):
    offset = mean_anomaly_offset(e, w)
    Ma = mod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI)
    return Ma


# Ecccentric anomaly
# ==================

@njit("f8[:](f8[:], f8, f8, f8, f8)", cache=cache)
def ea_newton_v(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ea = Ma.copy()
    for j in range(len(t)):
        err = 0.05
        k = 0
        while abs(err) > 1e-8 and k<1000:
            err   = Ea[j] - e*sin(Ea[j]) - Ma[j]
            Ea[j] = Ea[j] - err/(1.0-e*cos(Ea[j]))
            k += 1
    return Ea

@njit("f8(f8, f8, f8, f8, f8)", cache=cache)
def ea_newton_s(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ea = Ma
    err = 0.05
    k = 0
    while abs(err) > 1e-8 and k<1000:
        err   = Ea - e*sin(Ea) - Ma
        Ea = Ea - err/(1.0-e*cos(Ea))
        k += 1
    return Ea

@njit("f8(f8,f8,f8,f8,f8)", cache=cache)
def ea_iter_s(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    ec = e*sin(Ma)/(1.0 - e*cos(Ma))
    for k in range(15):
        ect = ec
        ec  = e*sin(Ma+ec)
        if (abs(ect-ec) < 1e-4):
            break
    Ea  = Ma + ec
    return Ea

@njit("f8[:](f8[:],f8,f8,f8,f8)", parallel=True)
def ea_iter_v(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    ec = e*sin(Ma)/(1.0 - e*cos(Ma))
    for j in prange(len(t)):
        for k in range(15):
            ect   = ec[j]
            ec[j] = e*sin(Ma[j]+ec[j])
            if (abs(ect-ec[j]) < 1e-4):
                break
    Ea  = Ma + ec
    return Ea

# True Anomaly
# ============

@njit("f8[:](f8[:],f8)", parallel=True)
def ta_from_ea_v(Ea, e):
    sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
    cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
    Ta  = arctan2(sta, cta)
    return Ta

@njit("f8(f8,f8)", cache=cache)
def ta_from_ea_s(Ea, e):
    sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
    cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
    Ta  = arctan2(sta, cta)
    return Ta

@njit("f8(f8, f8, f8, f8, f8)", cache=cache)
def ta_newton_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_newton_s(t, t0, p, e, w), e)

@njit("f8[:](f8[:], f8, f8, f8, f8)", parallel=True)
def ta_newton_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_newton_v(t, t0, p, e, w), e)

@njit("f8(f8,f8,f8,f8,f8)", cache=cache)
def ta_iter_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_iter_s(t, t0, p, e, w), e)

@njit("f8[:](f8[:],f8,f8,f8,f8)", parallel=True)
def ta_iter_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_iter_v(t, t0, p, e, w), e)

@njit(parallel=True)
def ta_ps3(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = (Ma + (2.0*e - 0.25*e**3)*sin(Ma)
             + 1.25*e**2*sin(2.0*Ma)
             + 13.0/12.0*e**3*sin(3.0*Ma))
    return Ta

@njit(parallel=True)
def ta_ps5(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = (Ma + (2.0*e - 0.25*e**3 + 5.0/96.0*e**5) * sin(Ma)
             + (1.25*e**2 - 11.0/24.0*e**4) * sin(2.0*Ma)
             + (13.0/12.0 * e**3 - 43.0/64.0 * e**5) * sin(3.0*Ma)
             + 103.0/96.0 * e**4 * sin(4.0*Ma)
             + 1097.0/960.0 * e**5 * sin(5.0*Ma))
    return Ta

@njit
def ta_from_ma(Ma, e):
    Ta = zeros_like(Ma)
    for j in range(len(Ma)):
        ec = e*sin(Ma[j])/(1.0 - e*cos(Ma[j]))
        for k in range(15):
            ect = ec
            ec  = e*sin(Ma[j]+ec)
            if abs(ect-ec) < 1e-4:
                break
        Ea = Ma[j] + ec
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta[j] = arctan2(sta, cta)
    return Ta

@njit
def ta_ip_calculate_table(ne=256, nm=512):
    es = linspace(0, 0.95, ne)
    ms = linspace(0,   pi, nm)
    tae = zeros((ne, nm))
    for i,e in enumerate(es):
        tae[i,:]  = ta_from_ma(ms, e)
        tae[i,:] -= ms
    return tae, es, ms

@njit(parallel=True)
def ta_ip(t, t0, p, e, w, es, ms, tae):
    ne = es.size
    nm = ms.size
    de = es[1] - es[0]
    dm = ms[1] - ms[0]

    ie = iclip(         floor(e/de),      0,   ne-1)
    ae = rclip((e - de*(ie-1)) / de,    0.0,    1.0)
    tae2 = tae[ie:ie+2,:]

    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = zeros_like(Ma)

    for i in range(len(t)):
        if Ma[i] < pi:
            im = iclip(floor(Ma[i]/dm), 0, nm-1)
            am = rclip((Ma[i] - dm*(im-1)) / dm, 0.0, 1.0)
            s = 1.0
        else:
            im = iclip(floor((TWO_PI - Ma[i])/dm), 0, nm-1)
            am = rclip((TWO_PI - (Ma[i] - dm*(im-1))) / dm, 0.0, 1.0)
            s = -1.0

        Ta[i] = ( tae2[0,im  ]*(1.0-ae)*(1.0-am)
                + tae2[1,im  ]*     ae *(1.0-am)
                + tae2[0,im+1]*(1.0-ae)*     am
                + tae2[1,im+1]*     ae *     am  )
        Ta[i] = Ma[i] + s * Ta[i]

        if (Ta[i] < 0.0):
            Ta[i] = Ta[i] + TWO_PI

    return Ta

# Projected distance Z
# ====================
# These functions calculate the projected distance (z) using different ways to calculate the
# tue anomaly (Ta). The functions have different versions optimized for different use-cases
#
#  - z_*_s  : scalar time
#  - z_*_v  : vector time
#  - z_*_p  : vector time, parallelized and usually fastest
#  - z_*_mp : vector time and two-dimensional parameter array, can be faster than the others
#

# Z: Newton's method
# ------------------

@njit(cache=cache)
def z_circular(t, pv):
    t0, p, a, i, e, w = pv
    cosph = cos(TWO_PI*(t-t0)/p)
    z = sign(cosph) * a*sqrt(1.0 - cosph*cosph*sin(i)**2)
    return z

@njit(cache=cache)
def z_newton_s(t, pv):
    """Normalized projected distance for scalar t.

    pv = [t0, p, a, i, e, w]
    """
    t0, p, a, i, e, w = pv
    Ta = ta_newton_s(t, t0, p, e, w)
    return z_from_ta_s(Ta, a, i, e, w)

@njit("f8[:](f8[:], f8[:])", cache=cache)
def z_newton_v(ts, pv):
    t0, p, a, i, e, w = pv
    Ta = ta_newton_v(ts, t0, p, e, w)
    return z_from_ta_v(Ta, a, i, e, w)

@njit("f8[:](f8[:], f8[:])", parallel=True, fastmath=True)
def z_newton_p(ts, pv):
    t0, p, a, i, e, w = pv
    zs = zeros_like(ts)
    for j in prange(len(ts)):
        t = ts[j]
        ma_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
        ma_offset -= e*sin(ma_offset)
        Ma = mod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI)
        Ea = Ma
        err = 0.05
        k = 0
        while abs(err) > 1e-8 and k<1000:
            err   = Ea - e*sin(Ea) - Ma
            Ea = Ea - err/(1.0-e*cos(Ea))
            k += 1
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta  = arctan2(sta, cta)
        z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
        z *= copysign(1.0, sin(w+Ta))
        zs[j] = z
    return zs

@njit("f8[:,:](f8[:], f8[:,:])", parallel=True, fastmath=True)
def z_newton_mp(ts, pvs):
    zs = zeros((pvs.shape[0], ts.size))
    for j in prange(zs.size):
        ipv, ipt = j // ts.size, j % ts.size
        t0, p, a, i, e, w = pvs[ipv]
        t = ts[ipt]
        ma_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
        ma_offset -= e*sin(ma_offset)
        Ma = mod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI)
        Ea = Ma
        err = 0.05
        k = 0
        while abs(err) > 1e-8 and k<1000:
            err   = Ea - e*sin(Ea) - Ma
            Ea = Ea - err/(1.0-e*cos(Ea))
            k += 1
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta  = arctan2(sta, cta)
        z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
        z *= copysign(1.0, sin(w+Ta))
        zs[ipv,ipt] = z
    return zs



# Z: Iteration
# ------------

@njit(cache=cache)
def z_iter_s(t, pv):
    t0, p, a, i, e, w = pv
    Ta = ta_iter_s(t, t0, p, e, w)
    return z_from_ta_s(Ta, a, i, e, w)

@njit("f8[:](f8[:], f8[:])", cache=cache)
def z_iter_v(ts, pv):
    t0, p, a, i, e, w = pv
    Ta = ta_iter_v(ts, t0, p, e, w)
    return z_from_ta_v(Ta, a, i, e, w)

@njit("f8[:](f8[:], f8[:])", parallel=True, fastmath=True)
def z_iter_p(ts, pv):
    t0, p, a, i, e, w = pv
    ma_offset = arctan2(sqrt(1.0 - e ** 2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    ma_offset -= e * sin(ma_offset)
    zs = zeros_like(ts)
    for j in prange(len(ts)):
        t = ts[j]
        Ma = mod(TWO_PI * (t - (t0 - ma_offset * p / TWO_PI)) / p, TWO_PI)
        ec = e*sin(Ma)/(1.0 - e*cos(Ma))
        for k in range(15):
            ect   = ec
            ec = e*sin(Ma+ec)
            if (abs(ect-ec) < 1e-4):
                break
        Ea  = Ma + ec
        sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
        cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
        Ta  = arctan2(sta, cta)
        z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
        z *= copysign(1.0, sin(w+Ta))
        zs[j] = z
    return zs

# Z: Series expansion
# -------------------

@njit("f8[:](f8[:], f8[:])", cache=cache)
def z_ps3(t, pv):
    t0, p, a, i, e, w = pv
    return z_from_ta_v(ta_ps3(t, t0, p, e, w), a, i, e, w)

@njit("f8[:](f8[:], f8[:])", cache=cache)
def z_ps5(t, pv):
    t0, p, a, i, e, w = pv
    return z_from_ta_v(ta_ps5(t, t0, p, e, w), a, i, e, w)

@njit
def impact_parameter(a, i):
    return a * cos(i)

@njit
def impact_parameter_ec(a, i, e, w, tr_sign):
    return a * cos(i) * ((1.-e**2) / (1.+tr_sign*e*sin(w)))

@njit
def duration_eccentric(p, k, a, i, e, w, tr_sign):
    b  = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = sqrt(1.-e**2)/(1.+tr_sign*e*sin(w))
    return p/pi  * arcsin(sqrt((1.+k)**2-b**2)/(a*sin(i))) * ae
