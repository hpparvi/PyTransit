#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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


"""PyTransit: fast and easy exoplanet transit modelling in Python

This package offers Python interfaces for a set of exoplanet transit light curve
models implemented in Python (with Numba acceleration) and OpenCL.

Author
  Hannu Parviainen  <hannu@iac.es>

Date
  24.04.2019

"""

# Numba models
# ------------
from .models.qpower2 import QPower2Model
from .models.ma_quadratic import QuadraticModel
from .models.ma_uniform import UniformModel
from .models.ma_chromosphere import ChromosphereModel

# OpenCL models
# -------------
from .models.qpower2_cl import QPower2ModelCL
from .models.ma_quadratic_cl import QuadraticModelCL
from .models.ma_uniform_cl import UniformModelCL

from .models.transitmodel import TransitModel

from .lpf.lpf import BaseLPF
from .lpf.cntlpf import PhysContLPF
from .lpf.baselines.legendrebaseline import LegendreBaseline
from .lpf.baselines.linearbaseline import LinearModelBaseline

from .param.parameter import UniformPrior, NormalPrior