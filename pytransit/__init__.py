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
models with Numba (JIT-compiled) and JAX backends.

Author
  Hannu Parviainen  <hannu@iac.es>

Date
  06.08.2024

"""

__version__ = '3.0.0'

# Generic
# -------
from .models.transitmodel import TransitModel

# Transit models
# --------------
from .models.rrmodel import RoadRunnerModel
from .models.udmodel import UniformDiskModel
from .models.tsmodel import TransmissionSpectroscopyModel

# Secondary-eclipse model
# -----------------------
from .models.semodel import SecondaryEclipseModel

# Limb darkening
# --------------
from .models.ldmodel import LDModel

# Model parameterization
# ----------------------
from .parameter import (Parameter, ParameterBlock, ParameterSet, normal_prior, uniform_prior,
                        prior_kind, prior_limits, prior_moments, describe_prior)

__all__ = ['TransitModel', 'RoadRunnerModel', 'UniformDiskModel',
           'TransmissionSpectroscopyModel', 'SecondaryEclipseModel', 'LDModel',
           'Parameter', 'ParameterBlock', 'ParameterSet', 'normal_prior', 'uniform_prior',
           'prior_kind', 'prior_limits', 'prior_moments', 'describe_prior']
