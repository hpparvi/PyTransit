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
model blocks about half as much light at mid-transit as a uniform disk would, and about three times
as much near the limb: the transit is shallower in the middle and deeper at the edges than the
naive :math:`k^2` expectation.

The profile follows from the geometry of the shell, so the model takes no limb darkening
coefficients.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import ChromosphereModel

    time = linspace(-0.1, 0.1, 1000)

    tm = ChromosphereModel()
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi)

API
---

.. autoclass:: pytransit.models.ma_chromosphere.ChromosphereModel
    :members: evaluate
