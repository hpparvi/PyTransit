API reference
=============

The transit models have their APIs on their own pages under :doc:`../models/index`, and the
supporting modules on :doc:`../stars`, :doc:`../contamination`, and :doc:`../io`. This section
collects what is left: the base class the models share, the limb darkening functions, and the
deprecated pre-2.9 evaluation methods.

.. toctree::
    :maxdepth: 2

    transitmodel
    limb_darkening
    deprecated

Quick index
-----------

Transit models
**************

.. list-table::
    :header-rows: 1
    :widths: 34 32 34

    * - Class
      - Import
      - Page
    * - `RoadRunnerModel`
      - ``from pytransit import RoadRunnerModel``
      - :doc:`../models/roadrunner`
    * - `TransmissionSpectroscopyModel`
      - ``from pytransit import TSModel``
      - :doc:`../models/tsmodel`
    * - `OblatePlanetModel`
      - ``from pytransit import OPModel``
      - :doc:`../models/opmodel`
    * - `EclipseModel`
      - ``from pytransit import EclipseModel``
      - :doc:`../models/eclipse`
    * - `EclipseSpectroscopyModel`
      - ``from pytransit import ESModel``
      - :doc:`../models/eclipse`
    * - `QuadraticModel`
      - ``from pytransit import QuadraticModel``
      - :doc:`../models/quadratic`
    * - `QPower2Model`
      - ``from pytransit import QPower2Model``
      - :doc:`../models/qpower2`
    * - `GeneralModel`
      - ``from pytransit import GeneralModel``
      - :doc:`../models/general`
    * - `UniformModel`
      - ``from pytransit import UniformModel``
      - :doc:`../models/uniform`
    * - `ChromosphereModel`
      - ``from pytransit import ChromosphereModel``
      - :doc:`../models/chromosphere`
    * - `GravityDarkenedModel`
      - ``from pytransit import GravityDarkenedModel``
      - :doc:`../models/gdmodel`
    * - `UniformModelCL`
      - ``from pytransit import UniformModelCL``
      - :doc:`../models/opencl`
    * - `QuadraticModelCL`
      - ``from pytransit import QuadraticModelCL``
      - :doc:`../models/opencl`
    * - `QPower2ModelCL`
      - ``from pytransit import QPower2ModelCL``
      - :doc:`../models/opencl`
    * - `RoadRunnerModelCL`
      - ``from pytransit.models.roadrunner.rrmodel_cl import RoadRunnerModelCL``
      - :doc:`../models/opencl`

Supporting modules
******************

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Module
      - Contents
    * - :doc:`pytransit.stars <../stars>`
      - BT-Settl and Husser et al. (2013) stellar spectrum grids.
    * - :doc:`pytransit.contamination <../contamination>`
      - Passbands, instruments, and flux contamination models.
    * - :doc:`pytransit.utils.io <../io>`
      - Light curve and radial velocity data containers.
    * - :doc:`pytransit.models.limb_darkening <limb_darkening>`
      - Numba-compiled limb darkening profiles and their disk integrals.
