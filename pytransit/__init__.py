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
models implemented in Python (with Numba acceleration) and OpenCL.

Author
  Hannu Parviainen  <hannu@iac.es>

Date
  06.08.2024

"""

__version__ = '2.6.14'

# Generic
# -------
from .models.transitmodel import TransitModel
from .contamination.filter import DeltaFilter, BoxcarFilter, TabulatedFilter, sdss_g, sdss_r, sdss_i, sdss_z

# Numba models
# ------------
from .models.qpower2 import QPower2Model
from .models.ma_quadratic import QuadraticModel
from .models.ma_uniform import UniformModel
from .models.eclipse_model import EclipseModel
from .models.ma_chromosphere import ChromosphereModel
from .models.general import GeneralModel
from .models.osmodel import OblateStarModel
from .models.gdmodel import GravityDarkenedModel

from .models import RoadRunnerModel, OblatePlanetModel, TransmissionSpectroscopyModel

TSModel = TransmissionSpectroscopyModel
OPModel = OblatePlanetModel
RRModel = RoadRunnerModel

# OpenCL models
# -------------
from .models.qpower2_cl import QPower2ModelCL
from .models.ma_quadratic_cl import QuadraticModelCL
from .models.ma_uniform_cl import UniformModelCL

# LDTk limb darkening for the Swift model
# ---------------------------------------

try:
    from .models.ldtkldm import LDTkLDModel, LDTkLD
except ImportError:
    pass

# Log posterior functions
# -----------------------
from .lpf.lpf import BaseLPF
from .lpf.transitlpf import TransitLPF
from .lpf.cntlpf import PhysContLPF
from .lpf.baselines.legendrebaseline import LegendreBaseline
from .lpf.baselines.linearbaseline import LinearModelBaseline
from .lpf.transitanalysis import TransitAnalysis

# Utilities
# ---------
from .param.parameter import UniformPrior, NormalPrior
from .utils import md_rs_from_rho
from .utils.mocklc import create_mock_light_curve
