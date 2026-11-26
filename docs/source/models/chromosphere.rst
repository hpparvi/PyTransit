Chromosphere model
==================

:class:`~pytransit.models.ma_chromosphere.ChromosphereModel` implements the optically thin shell
model of Schlawin et al. (ApJL 722, L75, 2010), for a transit observed in a chromospheric emission
line.

Limb brightening
----------------

A stellar photosphere is limb *darkened*: the disk is brightest at the centre. An optically thin,
spherically symmetric emitting shell -- a chromosphere seen in an emission line such as Hα or
Ca II K -- behaves in the opposite way. A line of sight near the limb passes through a longer path
of emitting material than one through the disk centre, so the projected disk is brightest at its
*edge*.

This inverts the transit shape. A model built for a limb-darkened photosphere cannot reproduce it,
which is why this model exists as a separate implementation rather than as a limb darkening law.

The effect is large, not subtle. For a planet with :math:`k = 0.1` on a central transit, the shell
model blocks half as much light at mid-transit as a uniform disk would, and 1.6 times as much just
inside the limb: the transit is shallower in the middle and deeper at the edges than the naive
:math:`k^2` expectation.

The model assumes an infinitesimally thin shell, whose surface brightness grows as :math:`1/\mu`
towards the limb, and evaluates the transit analytically. The profile follows from the geometry of
the shell, so the model takes no limb darkening coefficients. :doc:`../features/chromosphere` shows
why limb darkening laws, and RoadRunner's numerical integration, cannot reproduce it.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import ChromosphereModel

    time = linspace(-0.1, 0.1, 1000)

    tm = ChromosphereModel()
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi)

Threading
---------

The model evaluates a single light curve on one thread by default. ``nthreads`` above one
evaluates it in parallel.

.. code-block:: python

    tm = ChromosphereModel(nthreads=4)

The light curve is dealt out to the threads in small interleaved blocks, so that each thread gets
an even share of the in-transit points. A light curve of 100 000 points is about seven times
faster on eight threads than on one, and a light curve too short to gain from the threads is
evaluated serially. A parameter population is always evaluated in parallel over the parameter
sets.

Numba's thread count is process-global, so the most recently created model sets it for every model
in the process, and it is capped at the ``NUMBA_NUM_THREADS`` value fixed when Numba was imported.

.. note::

    Up to PyTransit 2.12 the model always used every Numba thread. Pass ``nthreads`` to get that
    behaviour back for long light curves.

API
---

.. autoclass:: pytransit.models.ma_chromosphere.ChromosphereModel
    :members: evaluate
    :special-members: __init__
