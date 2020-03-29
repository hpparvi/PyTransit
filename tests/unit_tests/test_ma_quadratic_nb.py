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

import unittest
from numpy import array

from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s, eval_quad_z_v

k = 0.1
ldc1d = array([0.2, 0.1])
ldc2d = array([[0.2, 0.1]])


def test_eval_quad_z_v_edge_cases():
    eval_quad_z_v(array([0.0]), k, ldc2d)
    eval_quad_z_v(array([1.0]), k, ldc2d)
    eval_quad_z_v(array([  k]), k, ldc2d)
    eval_quad_z_v(array([1-k]), k, ldc2d)
    eval_quad_z_v(array([1+k]), k, ldc2d)


def test_eval_quad_z_s_edge_cases():
    eval_quad_z_s(0.0, k, ldc1d)
    eval_quad_z_s(1.0, k, ldc1d)
    eval_quad_z_s(  k, k, ldc1d)
    eval_quad_z_s(1-k, k, ldc1d)
    eval_quad_z_s(1+k, k, ldc1d)
