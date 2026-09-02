Transit models
==============

PyTransit implements a family of exoplanet transit light curve models, each with model-specific
optimisations that make it efficient for the cases it was designed for. They all share the
interface described in the :doc:`../guide/index`; this section describes what each model computes,
when to reach for it, and its full API.

Choosing a model
----------------

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Situation
      - Model
    * - Anything, unless something below applies
      - :doc:`RoadRunnerModel <roadrunner>`
    * - Spectroscopic transit time series
      - :doc:`TransmissionSpectroscopyModel <tsmodel>`
    * - Secondary eclipse
      - :doc:`EclipseModel <eclipse>`
    * - Spectroscopic eclipse time series
      - :doc:`EclipseSpectroscopyModel <eclipse>`
    * - Planet with a measurably oblate projection
      - :doc:`OblatePlanetModel <opmodel>`
    * - Fast-rotating, gravity-darkened host star
      - :doc:`GravityDarkenedModel <gdmodel>`
    * - Transit over a chromosphere in an emission line
      - :doc:`ChromosphereModel <chromosphere>`
    * - Reproducing a published Mandel & Agol analysis
      - :doc:`QuadraticModel <quadratic>`
    * - Transit search, or limb darkening genuinely irrelevant
      - :doc:`UniformModel <uniform>`
    * - GPU acceleration
      - :doc:`the OpenCL models <opencl>`

**Start with RoadRunner.** It supports any radially symmetric limb darkening law, is fast, and is
accurate to sub-ppm with its default settings. The specialised models above exist because they
model something RoadRunner does not (an oblate planet, a gravity-darkened star, an eclipse) or
because they exploit structure RoadRunner cannot (the shared geometry of a spectroscopic time
series).

The remaining classical models --
:class:`~pytransit.models.ma_quadratic.QuadraticModel`,
:class:`~pytransit.models.qpower2.QPower2Model`, and
:class:`~pytransit.models.general.GeneralModel` -- are analytic solutions for one specific limb
darkening law each. They are kept because they are well tested, widely used, and directly
comparable to the literature, but for new work RoadRunner does what they do with more flexibility.
In particular, `GeneralModel` is superseded by RoadRunner with the ``'general'`` limb darkening
law, which is both faster and more flexible.

.. note::

    This guidance reflects the general design of the package rather than a benchmark of your
    particular problem. If model evaluation is the bottleneck in your analysis, time the
    candidates on your own data and hardware.

The catalogue
-------------

.. list-table::
    :header-rows: 1
    :widths: 30 26 44

    * - Model
      - Limb darkening
      - Reference
    * - :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel`
      - Any
      - Parviainen (MNRAS 499, 1633, 2020)
    * - :class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel`
      - Any
      - Parviainen (MNRAS 499, 1633, 2020)
    * - :class:`~pytransit.models.roadrunner.opmodel.OblatePlanetModel`
      - Any
      - Seager & Hui (2002); Barnes & Fortney (2003)
    * - :class:`~pytransit.models.ma_quadratic.QuadraticModel`
      - Quadratic
      - Mandel & Agol (ApJ 580, L171, 2002)
    * - :class:`~pytransit.models.ma_uniform.UniformModel`
      - None
      - Mandel & Agol (ApJ 580, L171, 2002)
    * - :class:`~pytransit.models.qpower2.QPower2Model`
      - Power-2
      - Maxted & Gill (A&A 622, A33, 2019)
    * - :class:`~pytransit.models.general.GeneralModel`
      - General
      - Giménez (A&A 450, 1231, 2006)
    * - :class:`~pytransit.models.ma_chromosphere.ChromosphereModel`
      - None
      - Schlawin et al. (ApJL 722, L75, 2010)
    * - :class:`~pytransit.models.eclipse_model.EclipseModel`
      - None
      - Mandel & Agol (ApJ 580, L171, 2002)
    * - :class:`~pytransit.models.roadrunner.esmodel.EclipseSpectroscopyModel`
      - None
      - Parviainen (MNRAS 499, 1633, 2020)
    * - :class:`~pytransit.models.gdmodel.GravityDarkenedModel`
      - Any
      - Barnes (ApJ 705, 683, 2009)

.. toctree::
    :maxdepth: 2
    :caption: RoadRunner family

    roadrunner
    tsmodel
    opmodel

.. toctree::
    :maxdepth: 2
    :caption: Eclipse models

    eclipse

.. toctree::
    :maxdepth: 2
    :caption: Classical models

    quadratic
    qpower2
    general
    uniform
    chromosphere

.. toctree::
    :maxdepth: 2
    :caption: Specialised models

    gdmodel

.. toctree::
    :maxdepth: 2
    :caption: GPU models

    opencl
