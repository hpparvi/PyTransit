Oblate planet model
===================

Rotation flattens a giant planet at its poles: Saturn's flattening is about 0.1 and Jupiter's about
0.06. Projected on the sky, such a planet is an ellipse rather than a circle, and its transit light
curve differs slightly from that of a spherical planet. Measuring the difference would constrain
the planet's rotation, which is otherwise very hard to access for exoplanets.

:class:`~pytransit.models.roadrunner.opmodel.OblatePlanetModel`, exported as ``OPModel``, models
the transit of such a planet. It is the :doc:`RoadRunner model <roadrunner>` with three parameters
for the planet's shape: the radius ratio ``k`` is the projected semi-major axis, ``f`` the
flattening, and ``alpha`` the projected obliquity, the angle from the planet's direction of motion
to its semi-major axis.

The signal is small, and much of it can be absorbed by a spherical planet with slightly different
parameters. Seeing the rest needs a model whose own numerical error is smaller still, so the model
offers accuracy settings that can be chosen to match the science.

.. plot::
    :context:
    :nofigs:

    from _figures.opmodel import plot_oblateness_signal, plot_accuracy_levels

What oblateness does to a transit
---------------------------------

A Saturn-like planet, :math:`k = 0.1` and :math:`f = 0.1`, crossing the star at an impact parameter
of 0.5, with three orientations of its axis:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import pi, linspace, arccos
    from pytransit import OPModel

    time = linspace(-0.15, 0.15, 1500)
    k, f, b = 0.1, 0.1, 0.5
    t0, p, a = 0.0, 10.0, 15.0

    tm = OPModel('quadratic', exact_ld=True, exact_areas=True)
    tm.set_data(time)

    fluxes = {alpha: tm.evaluate(k, f, alpha, [0.4, 0.25], t0, p, a, arccos(b / a))
              for alpha in (0.0, pi / 4, pi / 2)}

    plot_oblateness_signal(time, fluxes, k, f, b)

The upper row sketches the geometry, with the flattening exaggerated so that it is visible. The
lower row shows how the oblate planet's light curve differs from two spherical ones.

Compared with a spherical planet of the same area (dashed), the difference reaches 50 to 90 ppm,
all of it at ingress and egress. But a light curve is never compared with a sphere of a given size;
it is fitted. A spherical planet whose radius ratio, impact parameter, and orbit are fitted to the
oblate light curve (solid) absorbs almost all of the difference when the planet's axis is aligned
with its path or perpendicular to it: what remains is 5 to 7 ppm. An elongated planet crossing the
star looks much like a spherical one crossing it on a different chord.

A tilted axis is different. On a chord away from the stellar centre, the limb crosses the planet's
path at mirrored angles at ingress and egress. At :math:`\alpha = 45°` the planet's long axis meets
the limb at a different angle when it enters than when it leaves, so ingress and egress no longer
mirror each other. No spherical planet produces that asymmetry, and the full difference, 87 ppm,
survives the fit. This is the part of the signal that a spherical model cannot hide. (On a central
chord the two crossings look alike again, and a planet at :math:`\alpha = 45°` differs from a
sphere by only a few ppm.)

Accuracy to match the signal
----------------------------

A signal of a few ppm to a few tens of ppm, concentrated at ingress and egress, is only as
reliable as the model it is measured with. The figure compares the model settings with a direct
numerical integration of the limb-darkened stellar disk, for the tilted planet above and for a
grazing transit:

.. plot::
    :context: close-figs

    plot_accuracy_levels()

The **default** setting approximates the limb darkening behind the planet with that behind a
circle of the same area. Its error grows towards the limb and reaches about 30 ppm for
:math:`f = 0.1`: small enough when the oblateness is a nuisance, but comparable to the signal
itself when it is the measurement. ``exact_ld`` integrates the limb darkening over the planet's
exact elliptical footprint and brings the error below 1 ppm, except in grazing transits, where the
numerically computed intersection areas set a floor of up to about 7 ppm. Adding ``exact_areas``,
which computes those areas analytically, brings the grazing case below 1 ppm too.

The accuracy has a price. Timing a 1000-point light curve on one machine:

.. list-table::
    :header-rows: 1
    :widths: 50 25 25

    * - Setting
      - Time per evaluation
      - Relative cost
    * - Default
      - 0.12 ms
      - 1
    * - ``exact_areas``
      - 0.17 ms
      - 1.4
    * - ``exact_ld``
      - 2.9 ms
      - 24
    * - ``exact_ld`` and ``exact_areas``
      - 9.5 ms
      - 78

For comparison, ``RoadRunnerModel`` evaluates the same light curve for a spherical planet in
0.054 ms. ``exact_areas`` on its own changes little, since the limb darkening approximation, not
the areas, dominates the default error.

Fit fast, check exactly
-----------------------

The settings can be given to the initialiser, and both can be overridden in every call to
``evaluate``. A fit can therefore explore the parameter space with the fast default and check the
result with the exact settings, without building a second model:

.. code-block:: python

    tm = OPModel('quadratic')
    tm.set_data(time)

    flux_fast = tm.evaluate(k, f, alpha, ldc, t0, p, a, i)
    flux_exact = tm.evaluate(k, f, alpha, ldc, t0, p, a, i, exact_ld=True, exact_areas=True)

If the difference between the two is small compared with the residuals of the fit, the fast
setting was accurate enough. If it is not, the final sampling should use ``exact_ld``, adding
``exact_areas`` for grazing geometries.

.. seealso::

    :doc:`../models/opmodel` for the parameters, how the accuracy settings converge, and the API;
    :doc:`roadrunner` for the model it is built on.
