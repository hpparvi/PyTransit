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

from numpy.random.mtrand import normal, uniform
from pandas import DataFrame
from astropy.units import Unit, R_jup, R_sun, AU
from numpy import sqrt

from ..utils.phasecurves import equilibrium_temperature
from ..orbits import as_from_rhop, i_from_ba, d_from_pkaiews


def derive_qois(data: DataFrame, rstar: tuple = None, teff: tuple = None, distance_unit: Unit = R_jup):
    df = data.copy()
    ns = df.shape[0]

    df['period'] = period = df.p.values if 'p' in df else df.pr.values

    if 'k2_true' in df:
        df['k_true'] = sqrt(df.k2_true)
    if 'k2_app' in df:
        df['k_app'] = sqrt(df.k2_app)

    if 'k2_true' in df and 'k2_app' in df:
        df['cnt'] = 1. - df.k2_app / df.k2_true

    if 'g' in df:
        if 'k' in df:
            df['b'] = df.g * (1 + df.k)
        elif 'k_true' in df:
            df['b'] = df.g * (1 + df.k_true)

    df['a'] = as_from_rhop(df.rho.values, period)
    df['inc'] = i_from_ba(df.b.values, df.a.values)
    df['t14'] = d_from_pkaiews(period, df.k_true.values, df.a.values, df.inc.values, 0.0, 0.0, 1)
    df['t14_h'] = 24 * df.t14

    if rstar is not None:
        from astropy.units import R_sun
        rstar_d = (normal(*rstar, size=ns) * R_sun).to(distance_unit).value
        df['r_app'] = df.k_app.values * rstar_d
        df['r_true'] = df.k_true.values * rstar_d
        df['a_au'] = df.a * (rstar_d * distance_unit).to(AU)

    if teff is not None:
        df['teq_p'] = equilibrium_temperature(normal(*teff, size=ns), df.a, uniform(0.25, 0.50, ns), uniform(0, 0.4, ns))
    return df