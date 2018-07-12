from numpy import pi, sqrt, arccos, abs, log, ones_like, zeros, zeros_like, linspace, array, atleast_2d, floor
from numba import jit, prange

HALF_PI = 0.5 * pi
FOUR_PI = 4.0 * pi
INV_PI = 1 / pi

@jit(["f4(f4,f4)", "f8(f8,f8)"], cache=True, nopython=True)
def ellpicb(n, k):
    """The complete elliptical integral of the third kind

    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353

    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""

    kc = sqrt(1.0 - k * k)
    e = kc
    p = sqrt(n + 1.0)
    ip = 1.0 / p
    m0 = 1.0
    c = 1.0
    d = 1.0 / p

    for nit in range(1000):
        f = c
        c = d / p + c
        g = e / p
        d = 2.0 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0

        if (abs(1.0 - kc / g) > 1e-8):
            kc = 2.0 * sqrt(e)
            e = kc * m0
        else:
            return HALF_PI * (c * m0 + d) / (m0 * (m0 + p))
    return 0.0


@jit(cache=True, nopython=True)
def ellec(k):
    a1 = 0.443251414630
    a2 = 0.062606012200
    a3 = 0.047573835460
    a4 = 0.017365064510
    b1 = 0.249983683100
    b2 = 0.092001800370
    b3 = 0.040696975260
    b4 = 0.005264496390

    m1 = 1.0 - k * k
    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1.0 / m1)
    return ee1 + ee2


@jit(cache=True, nopython=True)
def ellk(k):
    a0 = 1.386294361120
    a1 = 0.096663442590
    a2 = 0.035900923830
    a3 = 0.037425637130
    a4 = 0.014511962120
    b0 = 0.50
    b1 = 0.124985935970
    b2 = 0.068802485760
    b3 = 0.033283553460
    b4 = 0.004417870120

    m1 = 1.0 - k * k
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1)
    return ek1 - ek2

@jit(["f4[:](f4[:],f4,f4)", "f8[:](f8[:],f8,f8)"], cache=True, nopython=True)
def eval_uniform(zs, k, c):
    flux = zeros_like(zs)

    if abs(k - 0.5) < 1e-3:
        k = 0.5

    for i in range(len(zs)):
        z = zs[i]
        if z < 0.0 or z > 1.0 + k:
            flux[i] = 1.0
        elif k > 1.0 and z < k - 1.0:
            flux[i] = 0.0
        elif (z > abs(1.0 - k) and z < 1.0 + k):
            kap1 = arccos(min((1.0 - k * k + z * z) / 2.0 / z, 1.0))
            kap0 = arccos(min((k * k + z * z - 1.0) / 2.0 / k / z, 1.0))
            lambdae = k * k * kap0 + kap1
            lambdae = (lambdae - 0.5 * sqrt(max(4.0 * z * z - (1.0 + z * z - k * k) ** 2, 0.0))) / pi
            flux[i] = 1.0 - lambdae
        elif (z < 1.0 - k):
            flux[i] = 1.0 - k * k
        flux[i] = c + (1.0 - c) * flux[i]
    return flux


@jit("Tuple((f8[:,:], f8[:], f8[:], f8[:]))(f8[:], f8, f8[:], f8)", cache=True, nopython=True, parallel=False)
def eval_quad(z0, k, u, c):
    if abs(k - 0.5) < 1.0e-4:
        k = 0.5

    npt = len(z0)
    npb = 1

    k2 = k ** 2
    omega = zeros(npb)
    flux = zeros((npt, npb))
    le = zeros(npt)
    ld = zeros(npt)
    ed = zeros(npt)

    for i in range(npb):
        omega[i] = 1.0 - u[2 * i - 1] / 3.0 - u[2 * i] / 6.0

    for i in prange(npt):
        z = z0[i]

        if abs(z - k) < 1e-6:
            z += 1e-6

        # The source is unocculted
        if z > 1.0 + k or z < 0.0:
            flux[i, :] = 1.0
            le[i] = 0.0
            ld[i] = 0.0
            ed[i] = 0.0
            continue

        # The source is completely occulted
        elif (k >= 1.0 and z <= k - 1.0):
            flux[i, :] = 0.0
            le[i] = 1.0
            ld[i] = 1.0
            ed[i] = 1.0
            continue

        z2 = z ** 2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k ** 2 - z ** 2

        # The source is partially occulted and the occulting object crosses the limb
        # Equation (26):
        if z >= abs(1.0 - k) and z <= 1.0 + k:
            kap1 = arccos(min((1.0 - k2 + z2) / (2.0 * z), 1.0))
            kap0 = arccos(min((k2 + z2 - 1.0) / (2.0 * k * z), 1.0))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4.0 * z2 - (1.0 + z2 - k2) ** 2, 0.0))) * INV_PI

        # The occulting object transits the source star (but doesn't completely cover it):
        if z <= 1.0 - k:
            le[i] = k2

        # The edge of the occulting star lies at the origin- special expressions in this case:
        if abs(z - k) < 1.e-4 * (z + k):
            # ! Table 3, Case V.:
            if (k == 0.5):
                ld[i] = 1.0 / 3.0 - 4.0 * INV_PI / 9.0
                ed[i] = 3.0 / 32.0
            elif (z > 0.5):
                q = 0.50 / k
                Kk = ellk(q)
                Ek = ellec(q)
                ld[i] = 1.0 / 3.0 + 16.0 * k / 9.0 * INV_PI * (2.0 * k2 - 1.0) * Ek - (
                                                                                       32.0 * k ** 4 - 20.0 * k2 + 3.0) / 9.0 * INV_PI / k * Kk
                ed[i] = 1.0 / 2.0 * INV_PI * (
                    kap1 + k2 * (k2 + 2.0 * z2) * kap0 - (1.0 + 5.0 * k2 + z2) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)))
            elif (z < 0.5):
                # ! Table 3, Case VI.:
                q = 2.0 * k
                Kk = ellk(q)
                Ek = ellec(q)
                ld[i] = 1.0 / 3.0 + 2.0 / 9.0 * INV_PI * (4.0 * (2.0 * k2 - 1.0) * Ek + (1.0 - 4.0 * k2) * Kk)
                ed[i] = k2 / 2.0 * (k2 + 2.0 * z2)

        # The occulting star partly occults the source and crosses the limb:
        # Table 3, Case III:
        if ((z > 0.5 + abs(k - 0.5) and z < 1.0 + k) or (k > 0.50 and z > abs(1.0 - k) and z < k)):
            q = sqrt((1.0 - (k - z) ** 2) / 4.0 / z / k)
            Kk = ellk(q)
            Ek = ellec(q)
            n = 1.0 / x1 - 1.0
            Pk = ellpicb(n, q)
            ld[i] = 1.0 / 9.0 * INV_PI / sqrt(k * z) * (
                ((1.0 - x2) * (2.0 * x2 + x1 - 3.0) - 3.0 * x3 * (x2 - 2.0)) * Kk + 4.0 * k * z * (
                    z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
            if (z < k):
                ld[i] = ld[i] + 2.0 / 3.0
            ed[i] = 1.0 / 2.0 * INV_PI * (
                kap1 + k2 * (k2 + 2.0 * z2) * kap0 - (1.0 + 5.0 * k2 + z2) / 4.0 * sqrt((1.0 - x1) * (x2 - 1.0)))

        # The occulting star transits the source:
        # Table 3, Case IV.:
        if k <= 1.0 and z <= (1.0 - k):
            q = sqrt((x2 - x1) / (1.0 - x1))
            Kk = ellk(q)
            Ek = ellec(q)
            n = x2 / x1 - 1.0
            Pk = ellpicb(n, q)
            ld[i] = 2.0 / 9.0 * INV_PI / sqrt(1.0 - x1) * (
                (1.0 - 5.0 * z2 + k2 + x3 * x3) * Kk + (1.0 - x1) * (z2 + 7.0 * k2 - 4.0) * Ek - 3.0 * x3 / x1 * Pk)
            if (z < k):
                ld[i] = ld[i] + 2.0 / 3.0
            if (abs(k + z - 1.0) < 1.e-4):
                ld[i] = 2.0 / 3.0 * INV_PI * arccos(1.0 - 2.0 * k) - 4.0 / 9.0 * INV_PI * sqrt(k * (1.0 - k)) * (
                    3.0 + 2.0 * k - 8.0 * k2)
            ed[i] = k2 / 2.0 * (k2 + 2.0 * z2)

        for j in range(npb):
            iu = 2 * j - 1
            iv = 2 * j
            flux[i, j] = 1.0 - ((1.0 - u[iu] - 2.0 * u[iv]) * le[i] + (u[iu] + 2.0 * u[iv]) * ld[i] + u[iv] * ed[i]) / omega[j]
        flux[i, :] = c + (1.0 - c) * flux[i, :]

    return flux, ld, le, ed

@jit(cache=True, nopython=True)
def eval_chromosphere(zs, k, c):
    """Optically thin chromosphere model presented in
       Schlawin, Agol, Walkowicz, Covey & Lloyd (2010)"""
    nz = len(zs)
    flux = ones_like(zs)

    for i in range(nz):
        z = zs[i]
        if ((z > 0.0) and (z - k < 1.0)):
            zmk2 = (z - k) ** 2
            if (z + k < 1.0):
                t = sqrt(4.0 * z * k / (1.0 - zmk2))
                flux[i] = (4.0 / sqrt(1.0 - zmk2) *
                           ((zmk2 - 1.0) * ellec(t)
                            - (z ** 2 - k ** 2) * ellk(t)
                            + (z + k) / (z - k) * ellpicb(4.0 * z * k / zmk2, t)))

            elif (z + k > 1.0):
                t = sqrt((1.0 - zmk2) / (4.0 * z * k))
                flux[i] = (2.0 / (z - k) / sqrt(z * k) *
                           (4.0 * z * k * (k - z) * ellec(t)
                            + (-z + 2.0 * z ** 2 * k + k - 2.0 * k ** 3) * ellk(t)
                            + (z + k) * ellpicb(1.0 / zmk2 - 1.0, t)))
            if (k > z):
                flux[i] += FOUR_PI
            flux[i] = 1.0 - flux[i] / FOUR_PI
            flux[i] = c + (1.0 - c) * flux[i]
    return flux


def calculate_interpolation_tables(kmin=0.05, kmax=0.2, nk=512, nz=512):
    zs = linspace(0, 1 + 1.001 * kmax, nz)
    ks = linspace(kmin, kmax, nk)

    ld = zeros((nk, nz))
    le = zeros((nk, nz))
    ed = zeros((nk, nz))

    for ik, k in enumerate(ks):
        _, ld[ik, :], le[ik, :], ed[ik, :] = eval_quad(zs, k, array([0.0, 0.0]), 0.0)

    return ed, le, ld, ks, zs


@jit(cache=False, nopython=True, parallel=True, fastmath=True)
def eval_quad_ip(zs, k, u, c, edt, ldt, let, kt, zt):
    u = atleast_2d(u)
    npb = u.shape[0]
    flux = zeros((len(zs), npb))
    omega = zeros(npb)
    dk = kt[1] - kt[0]
    dz = zt[1] - zt[0]

    for i in range(npb):
        omega[i] = 1.0 - u[i, 0] / 3.0 - u[i, 1] / 6.0

    ik = int(floor((k - kt[0]) / dk))
    ak1 = (k - kt[ik]) / dk
    ak2 = 1.0 - ak1

    ed2 = edt[ik:ik + 2, :]
    ld2 = ldt[ik:ik + 2, :]
    le2 = let[ik:ik + 2, :]

    for i in prange(len(zs)):
        z = zs[i]
        if (z >= 1.0 + k) or (z < 0.0):
            flux[i, :] = 1.0
        else:
            iz = int(floor((z - zs[0]) / dz))
            az1 = (z - zt[iz]) / dz
            az2 = 1.0 - az1

            ed = (ed2[0, iz] * ak2 * az2
                  + ed2[1, iz] * ak1 * az2
                  + ed2[0, iz + 1] * ak2 * az1
                  + ed2[1, iz + 1] * ak1 * az1)

            ld = (ld2[0, iz] * ak2 * az2
                  + ld2[1, iz] * ak1 * az2
                  + ld2[0, iz + 1] * ak2 * az1
                  + ld2[1, iz + 1] * ak1 * az1)

            le = (le2[0, iz] * ak2 * az2
                  + le2[1, iz] * ak1 * az2
                  + le2[0, iz + 1] * ak2 * az1
                  + le2[1, iz + 1] * ak1 * az1)

            for j in range(npb):
                flux[i, j] = 1.0 - ((1.0 - u[j, 0] - 2.0 * u[j, 1]) * le + (u[j, 0] + 2.0 * u[j, 1]) * ld + u[
                    j, 1] * ed) / omega[j]
                flux[i, j] = c + (1.0 - c) * flux[i, j]

    return flux