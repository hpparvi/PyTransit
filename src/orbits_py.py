from numpy import pi, arctan2, sin, cos, sqrt, sign, copysign, mod, zeros_like
from numba import jit, prange


HALF_PI = 0.5 * pi
TWO_PI = 2.0 * pi

# Utilities
# =========

@jit("f8(f8, f8)", cache=True, nopython=True)
def mean_anomaly_offset(e, w):
    mean_anomaly_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    mean_anomaly_offset -= e*sin(mean_anomaly_offset)
    return mean_anomaly_offset

@jit("f8(f8, f8, f8, f8, f8)", cache=True, nopython=True)
def z_from_ta_s(Ta, a, i, e, w):
    z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
    z *= copysign(1.0, sin(w+Ta))
    return z

@jit("f8[:](f8[:], f8, f8, f8, f8)", parallel=True, nopython=True)
def z_from_ta_v(Ta, a, i, e, w):
    z  = a*(1.0-e**2)/(1.0+e*cos(Ta)) * sqrt(1.0 - sin(w+Ta)**2 * sin(i)**2)
    z *= sign(1.0, sin(w+Ta))
    return z

# Mean Anomaly
# ============

@jit(cache=True, nopython=True)
def mean_anomaly(t, t0, p, e, w):
    offset = mean_anomaly_offset(e, w)
    Ma = mod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI)
    return Ma

@jit(parallel=True, nopython=True)
def mean_anomaly_p(t, t0, p, e, w):
    offset = mean_anomaly_offset(e, w)
    Ma = mod(TWO_PI * (t - (t0 - offset * p / TWO_PI)) / p, TWO_PI)
    return Ma


# Ecccentric anomaly
# ==================

@jit("f8[:](f8[:], f8, f8, f8, f8)", cache=True, nopython=True)
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

@jit("f8(f8, f8, f8, f8, f8)", cache=True, nopython=True)
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

@jit("f8(f8,f8,f8,f8,f8)", cache=True, nopython=True)
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

@jit("f8[:](f8[:],f8,f8,f8,f8)", parallel=True, nopython=True)
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

@jit("f8[:](f8[:],f8)", parallel=True, nopython=True)
def ta_from_ea_v(Ea, e):
    sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
    cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
    Ta  = arctan2(sta, cta)
    return Ta

@jit("f8(f8,f8)", cache=True, nopython=True)
def ta_from_ea_s(Ea, e):
    sta = sqrt(1.0-e**2) * sin(Ea)/(1.0-e*cos(Ea))
    cta = (cos(Ea)-e)/(1.0-e*cos(Ea))
    Ta  = arctan2(sta, cta)
    return Ta

@jit("f8(f8, f8, f8, f8, f8)", cache=True, nopython=True)
def ta_newton_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_newton_s(t, t0, p, e, w), e)

@jit("f8[:](f8[:], f8, f8, f8, f8)", parallel=True, nopython=True)
def ta_newton_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_newton_v(t, t0, p, e, w), e)

@jit("f8(f8,f8,f8,f8,f8)", cache=True, nopython=True)
def ta_iter_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_iter_s(t, t0, p, e, w), e)

@jit("f8[:](f8[:],f8,f8,f8,f8)", parallel=True, nopython=True)
def ta_iter_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_iter_v(t, t0, p, e, w), e)

@jit(parallel=True, nopython=True)
def ta_ps3(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = (Ma + (2.0*e - 0.25*e**3)*sin(Ma)
             + 1.25*e**2*sin(2.0*Ma)
             + 13.0/12.0*e**3*sin(3.0*Ma))
    return Ta

@jit(parallel=True, nopython=True)
def ta_ps5(t, t0, p, e, w):
    Ma = mean_anomaly(t, t0, p, e, w)
    Ta = (Ma + (2.0*e - 0.25*e**3 + 5.0/96.0*e**5) * sin(Ma)
             + (1.25*e**2 - 11.0/24.0*e**4) * sin(2.0*Ma)
             + (13.0/12.0 * e**3 - 43.0/64.0 * e**5) * sin(3.0*Ma)
             + 103.0/96.0 * e**4 * sin(4.0*Ma)
             + 1097.0/960.0 * e**5 * sin(5.0*Ma))
    return Ta


# Projected distance Z
# ====================

# Z: Newton's method
# ------------------

@jit(cache=True, nopython=True)
def z_newton_s(t, t0, p, a, i, e, w):
    Ta = ta_newton_s(t, t0, p, e, w)
    return z_from_ta_s(Ta, a, i, e, w)

@jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8)", cache=True, nopython=True)
def z_newton_v(ts, t0, p, a, i, e, w):
    Ta = ta_newton_v(ts, t0, p, e, w)
    return z_from_ta_v(Ta, a, i, e, w)

@jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8)", parallel=True, nopython=True, fastmath=True)
def z_newton_p(ts, t0, p, a, i, e, w):
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

@jit("f8[:,:](f8[:], f8[:,:])", parallel=True, nopython=True, fastmath=True)
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

@jit(cache=True, nopython=True)
def z_iter_s(t, t0, p, a, i, e, w):
    Ta = ta_iter_s(t, t0, p, e, w)
    return z_from_ta_s(Ta, a, i, e, w)

@jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8)", cache=True, nopython=True)
def z_iter_v(ts, t0, p, a, i, e, w):
    Ta = ta_iter_v(ts, t0, p, e, w)
    return z_from_ta_v(Ta, a, i, e, w)

@jit("f8[:](f8[:],f8,f8,f8,f8, f8, f8)", parallel=True, nopython=True, fastmath=True)
def z_iter_p(ts, t0, p, a, i, e, w):
    zs = zeros_like(ts)
    for j in prange(len(ts)):
        t = ts[j]
        ma_offset = arctan2(sqrt(1.0-e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
        ma_offset -= e*sin(ma_offset)
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

@jit("f8[:](f8[:],f8,f8,f8,f8, f8, f8)", cache=True, nopython=True)
def z_ps3(t, t0, p, a, i, e, w):
    return z_from_ta_v(ta_ps3(t, t0, p, e, w), a, i, e, w)

@jit("f8[:](f8[:],f8,f8,f8,f8, f8, f8)", cache=True, nopython=True)
def z_ps5(t, t0, p, a, i, e, w):
    return z_from_ta_v(ta_ps5(t, t0, p, e, w), a, i, e, w)