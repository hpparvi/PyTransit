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
  9.08.2020

"""

from .version import __version__

# Generic
# -------
from .models.transitmodel import TransitModel

# Numba models
# ------------
from .models.qpower2 import QPower2Model
from .models.ma_quadratic import QuadraticModel
from .models.ma_uniform import UniformModel
from .models.eclipse_model import EclipseModel
from .models.ma_chromosphere import ChromosphereModel
from .models.general import GeneralModel
from .models.rrmodel import RoadRunnerModel
from .models.swiftmodel import SwiftModel
from .models.osmodel import OblateStarModel

# OpenCL models
# -------------
class DummyModelCL:
    def __init__(self, *args, **kwargs):
        raise ImportError('Cannot use the OpenCL models because pyopencl is not installed. Please install pyopencl.')


try:
    from .models.qpower2_cl import QPower2ModelCL
    from .models.ma_quadratic_cl import QuadraticModelCL
    from .models.ma_uniform_cl import UniformModelCL
    from .models.swiftmodel_cl import SwiftModelCL, SwiftModelCL as SWIFTModelCL
except ModuleNotFoundError:
    QPower2ModelCL = DummyModelCL
    QuadraticModelCL = DummyModelCL
    UniformModelCL = DummyModelCL
    SwiftModelCL = SWIFTModelCL = DummyModelCL


# LDTk limb darkening for the Swift model
# ---------------------------------------
class DummyLDTkLDModel:
    def __init__(self, *args, **kwargs):
        raise ImportError('Cannot use the LDTk limb darkening model because ldtk is not installed. Please install ldtk.')


try:
    from .models.ldtkldm import LDTkLDModel, LDTkLD
except ModuleNotFoundError:
    LDTkLD = LDTkLDModel = DummyLDTkLDModel


# Log posterior functions
# -----------------------
from .lpf.lpf import BaseLPF
from .lpf.cntlpf import PhysContLPF
from .lpf.baselines.legendrebaseline import LegendreBaseline
from .lpf.baselines.linearbaseline import LinearModelBaseline


# Utilities
# ---------
from .param.parameter import UniformPrior, NormalPrior
from .utils import md_rs_from_rho
