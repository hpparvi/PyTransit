RoadRunner model
================

The RoadRunner model (Parviainen, MNRAS 499, 1633, 2020) is PyTransit's recommended
general-purpose transit model, and the one to use unless a specialised model fits your problem
better.

What makes it different
-----------------------

The classical transit models are analytic solutions to the overlap integral of a limb-darkened
disk and an opaque circle, derived separately for each limb darkening law. A new law needs a new
derivation, and some laws have no closed-form solution at all.

RoadRunner separates the problem into two parts. The *geometry* -- how much of the stellar disk the
planet covers at each distance from the disk centre -- is solved numerically once and tabulated as
a set of weights. The *stellar intensity profile* enters only as a set of samples on a fixed grid
of normalised distances. The transit is then a weighted sum over that grid.

Two things follow:

- **Any radially symmetric intensity profile works.** A built-in law, a Python callable, or a
  numerically tabulated profile from a stellar atmosphere model are all equally valid inputs.
- **The cost barely depends on the law.** A four-coefficient non-linear profile costs about the
  same as a linear one, because the expensive part -- the geometry -- is shared.

Accuracy is controlled by the discretisation of the stellar disk: `nzin` nodes across the inner
disk, `nzlimb` nodes across the limb, split at `zcut`, and `ng` grazing-geometry nodes. The error
grows with the radius ratio: measured against the analytic Mandel & Agol solution, the defaults
give roughly 1 ppm at :math:`k = 0.02`, a few ppm around :math:`k = 0.1`, and tens of ppm above
:math:`k = 0.2`. See :doc:`../features/roadrunner` for the measured curve and for which settings
to raise.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import RoadRunnerModel

    time = linspace(-0.1, 0.1, 1000)

    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

Choosing a different limb darkening law is a change to one string:

.. code-block:: python

    tm = RoadRunnerModel('power-2')
    tm.set_data(time)
    flux = tm.evaluate(k=0.1, ldc=[0.6, 0.5], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

See :doc:`../guide/limb_darkening` for the available laws and for supplying your own.

Performance options
-------------------

**Threading.** ``nthreads`` above one enables the parallel model.

.. code-block:: python

    tm = RoadRunnerModel('quadratic', nthreads=4)

Numba's thread count is process-global, so the most recently created model sets it for every model
in the process, and it is capped at the ``NUMBA_NUM_THREADS`` value fixed when Numba was imported.

**Precomputed weights.** ``precompute_weights=True`` builds a 3D weight table covering the radius
ratio range `klims`, trading initialisation time for evaluation speed. Worth it when the model is
evaluated many times with the radius ratio confined to a known range.

**Small planet approximation.** For a single light curve with a radius ratio at or below
``small_planet_limit`` (0.01 by default), the model approximates the mean blocked intensity by the
intensity at the planet's centre. The error grows roughly quadratically with the radius ratio:
below 1 ppm at the default limit, about 100 ppm at :math:`k = 0.05`. Raise the limit only if speed
matters more than ppm accuracy, and pass ``None`` or ``0.0`` to disable it.

Examples
--------

.. toctree::
    :maxdepth: 1

    ../notebooks/models/roadrunner/roadrunner_model_example_1
    ../notebooks/models/roadrunner/roadrunner_model_example_2

API
---

.. autoclass:: pytransit.models.roadrunner.rrmodel.RoadRunnerModel
    :members: set_data, evaluate, init_integration
    :special-members: __init__
