The transit model interface
===========================

All PyTransit transit models derive from :class:`~pytransit.models.transitmodel.TransitModel` and
share a common interface, with small variations that account for model-specific parameters and
settings. Some models add evaluation methods aimed at particular science cases, such as
transmission spectroscopy of a spectroscopic time series.

The models are built to work with heterogeneous photometric time series: a single evaluation can
model observations taken in different passbands, with different exposure times, and with different
supersampling rates.

Model creation
--------------

Model creation takes the model's own configuration, never any data or physical parameters. For
most models the defaults are sensible and no arguments are needed

.. code-block:: python

    from pytransit import QuadraticModel

    tm = QuadraticModel()

while a flexible model such as RoadRunner takes the limb darkening law it should use, and the
parameters that control its numerical accuracy

.. code-block:: python

    from pytransit import RoadRunnerModel

    tm = RoadRunnerModel('power-2', nzin=20, nzlimb=20)

The arguments each model accepts are documented on its own page under :doc:`../models/index`.

The parameter vocabulary
------------------------

The physical parameters passed to `evaluate` are named consistently across the models.

.. list-table::
    :header-rows: 1
    :widths: 10 22 68

    * - Name
      - Quantity
      - Description
    * - ``k``
      - Radius ratio
      - Planet radius divided by the stellar radius, :math:`R_\mathrm{p}/R_\star`. May be
        passband-dependent.
    * - ``ldc``
      - Limb darkening coefficients
      - The coefficients of the model's limb darkening law, one set per passband. See
        :doc:`limb_darkening`.
    * - ``t0``
      - Zero epoch
      - Mid-transit time of a reference transit, in the same units and zero point as `time`.
    * - ``p``
      - Orbital period
      - In days.
    * - ``a``
      - Scaled semi-major axis
      - Orbital semi-major axis divided by the stellar radius, :math:`a/R_\star`.
    * - ``i``
      - Orbital inclination
      - In radians. A central transit has :math:`i = \pi/2`.
    * - ``e``
      - Eccentricity
      - Optional, defaults to a circular orbit.
    * - ``w``
      - Argument of periastron
      - In radians. Optional, and meaningful only for an eccentric orbit.

The transit depth is set by ``k``, its duration and shape by ``a``, ``i``, and ``p``, and the
curvature of its floor by ``ldc``.

Individual models add parameters of their own: the oblate planet model takes a flattening ``f``
and a projected obliquity ``alpha``, the eclipse model takes a flux ratio ``fr``, and the
gravity-darkened model takes stellar rotation and gravity darkening parameters. These are
documented with the model.

.. tip::

    Several useful transformations between these and other common parametrisations -- impact
    parameter to inclination, stellar density to scaled semi-major axis, transit duration -- live
    in ``pytransit.orbits``.

Evaluating the model
--------------------

Every model is evaluated through a single `evaluate` method that accepts scalars, 1D arrays, or 2D
arrays and broadcasts accordingly. One call handles a single parameter set, passband-dependent
parameters, and a whole population of parameter vectors; the shapes of the arguments decide which.
:doc:`evaluation` describes the rules in full.

.. code-block:: python

    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

Models with extra physics extend the argument list rather than adding methods: the oblate planet
model takes a flattening and a projected obliquity, the eclipse models take a flux ratio, and the
spectroscopy models return an extra wavelength axis.

.. note::

    PyTransit versions before 2.9 also had ``evaluate_ps`` and ``evaluate_pv`` methods. They are
    deprecated as of 2.9 and will be removed in 3.0; ``evaluate`` does everything they did. See
    :doc:`../api/deprecated` if you have code that still calls them.

The `copy` argument
-------------------

Most `evaluate` methods accept a ``copy`` keyword. It is meaningful only for the OpenCL models,
where ``copy=False`` leaves the computed model in GPU memory instead of transferring it back to
the host. This is worth doing when the likelihood is also evaluated on the GPU. For the Numba
models the argument is accepted and ignored, so that the same calling code works with either
backend.
