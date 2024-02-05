from numba import njit, prange
from numpy import zeros, floor, sqrt, pi


@njit
def trilinear_interpolation_unit_cube(data, x, y, z):
    """Trilinear interpolation inside a unit cube.

    Implementation taken from Paul Bourke's notes
    https://paulbourke.net/miscellaneous/interpolation
    """
    n1 = data.shape[3]
    n2 = data.shape[4]
    ldp = zeros((n1, n2))
    rx, ry, rz = (1.0 - x), (1.0 - y), (1.0 - z)

    a1 = rx * ry * rz
    a2 = x * ry * rz
    b1 = rx * y * rz
    b2 = rx * ry * z
    c1 = x * ry * z
    c2 = rx * y * z
    d1 = x * y * rz
    d2 = x * y * z

    for i1 in range(n1):
        for i2 in range(n2):
            ldp[i1, i2] = (data[0, 0, 0, i1, i2] * a1 +
                           data[1, 0, 0, i1, i2] * a2 +
                           data[0, 1, 0, i1, i2] * b1 +
                           data[0, 0, 1, i1, i2] * b2 +
                           data[1, 0, 1, i1, i2] * c1 +
                           data[0, 1, 1, i1, i2] * c2 +
                           data[1, 1, 0, i1, i2] * d1 +
                           data[1, 1, 1, i1, i2] * d2)
    return ldp


@njit
def trilinear_interpolation_single(data, x, y, z, x0, dx, nx, y0, dy, ny, z0, dz, nz):
    x = min(max(x, x0), x0 + nx * dx)
    y = min(max(y, y0), y0 + ny * dy)
    z = min(max(z, z0), z0 + nz * dz)
    ix = int(floor((x - x0) / dx))
    iy = int(floor((y - y0) / dy))
    iz = int(floor((z - z0) / dz))
    ax = (x - x0 - ix * dx) / dx
    ay = (y - y0 - iy * dy) / dy
    az = (z - z0 - iz * dz) / dz
    return trilinear_interpolation_unit_cube(data[ix:ix + 2, iy:iy + 2, iz:iz + 2, :, :], ax, ay, az)


@njit(parallel=True)
def trilinear_interpolation_set(data, xs, ys, zs, x0, dx, nx, y0, dy, ny, z0, dz, nz):
    npv = xs.shape[0]
    ldp = zeros((npv, data.shape[3], data.shape[4]))
    for ipv in prange(npv):
        ldp[ipv, :, :] = trilinear_interpolation_single(data, xs[ipv], ys[ipv], zs[ipv], x0, dx, nx, y0, dy, ny, z0, dz,
                                                        nz)
    return ldp


@njit
def integrate_profiles_single(mu, ldp):
    nmu = mu.size
    npb = ldp.shape[0]
    ldi = zeros(npb)

    z = sqrt(1.0 - mu ** 2)
    dz = z[1:] - z[:-1]
    for ipb in range(npb):
        for i in range(1, nmu):
            ldi[ipb] += dz[i - 1] * 0.5 * (z[i] * ldp[ipb, i] + z[i - 1] * ldp[ipb, i - 1])
    return 2.0 * pi * ldi


@njit(parallel=True)
def integrate_profiles_set(mu, ldp):
    nmu = mu.size
    npv = ldp.shape[0]
    npb = ldp.shape[1]
    ldi = zeros((npv, npb))

    z = sqrt(1.0 - mu ** 2)
    dz = z[1:] - z[:-1]
    for j in prange(npv * npb):
        ipv = j // npb
        ipb = j % npb
        for i in range(1, nmu):
            ldi[ipv, ipb] += dz[i - 1] * 0.5 * (z[i] * ldp[ipv, ipb, i] + z[i - 1] * ldp[ipv, ipb, i - 1])
    return 2.0 * pi * ldi
