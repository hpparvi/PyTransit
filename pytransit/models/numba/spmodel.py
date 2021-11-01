from numpy import sqrt, abs, zeros, ones
from numba import njit, prange

from .rrmodel import circle_circle_intersection_area
from .ldmodels import ld_quadratic, ldi_quadratic

@njit(parallel=True)
def quadratic_small_planet_zp(z, k, ldc):
    """Small planet approximation with quadratic limb darkening.

    Parallelised transit model using the small-planet approximation as a 
    function of normalized planet-star separation. The function is meant to
    be used to evaluate the model for npv radius ratios and limb darkening 
    coefficients in parallel, where npv should be relatively large to offset
    the cost from the threading initialization. 

    Parameters
    ----------
    z: array
        Planet-star separations as a 1d ndarray
    k: array
        Radius ratios as a 1d ndarray
    ldc: array
        Limb darkening coefficients as [npv, 2] ndarray

    Returns
    -------

    """
    npt = z.size
    npv = k.size

    bs = zeros(npt)
    mus = zeros(npt)
    for ipt in range(npt):
        bs[ipt] = abs(z[ipt])
        mus[ipt] = sqrt(1. - min(bs[ipt]**2, 1.0))

    flux = ones((npv, npt))
    for ipv in prange(npv):
        istar = ldi_quadratic(ldc[ipv])
        for ipt in range(npt):
            if bs[ipt] < 1.0 + k[ipv]:
                iplanet = ld_quadratic(mus[ipt], ldc[ipv])
                aplanet = circle_circle_intersection_area(1.0, k[ipv], bs[ipt])
                flux[ipv, ipt] = (istar - iplanet*aplanet) / istar
    return flux