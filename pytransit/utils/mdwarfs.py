#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2020  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Union

from numpy import array, diag, zeros, log10
from numpy.random import normal, multivariate_normal
from uncertainties import UFloat


def md_rs_from_rho(rho: Union[UFloat, float], nsamples: int = 100):
    """M dwarf stellar radius from stellar density (Hartman et al. 2015)

    Calculates the stellar radius from a stellar density estimate using the
    empirical M dwarf density-radius relation presented by Hartman et al. (2015).

    Parameters
    ----------
    rho: float or UFloat
        Stellar density in g/cm^3
    nsamples
        Number of samples to compute

    Returns
    -------
    ndarray
        Stellar radius in Solar radii

    """

    arm = array([0.054, -0.2989, 0.0815, -0.18280])
    are = array([0.048, 0.0083, 0.0020, 0.00075])
    A = array([0.131674, -0.605624, 0.739127, -0.263765,
               -0.494952, 0.614479, 0.437886, -0.430922,
               -0.765425, -0.291096, 0.099548, 0.565224,
               0.389626, 0.413398, 0.502032, 0.652117]).reshape([4, 4])

    ars = multivariate_normal(arm, diag(are ** 2), size=nsamples)

    if isinstance(rho, UFloat):
        rhos = normal(rho.n, rho.s, nsamples)
        rs = zeros(nsamples)
        for i in range(nsamples):
            lrho = log10(rhos[i])
            lra = array([1, lrho, lrho ** 2, lrho ** 3])
            rs[i] = 10 ** (ars[i] @ A @ lra)
        return rs
    elif isinstance(rho, float):
        lrho = log10(rho)
        lra = array([1, lrho, lrho ** 2, lrho ** 3])
        return 10 ** (ars @ A @ lra)
    else:
        raise NotImplementedError
