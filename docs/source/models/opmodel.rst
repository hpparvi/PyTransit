Oblate planet model
===================

:class:`~pytransit.models.roadrunner.opmodel.OblatePlanetModel` -- also exported as ``OPModel`` --
is the RoadRunner model for a planet whose sky projection is an ellipse rather than a circle.

Why oblateness matters
----------------------

A rapidly rotating planet is flattened at its poles. Saturn's oblateness is about 0.1, Jupiter's
about 0.06. A flattened planet projected on the sky is an ellipse, and an ellipse crossing a
limb-darkened disk produces a slightly different light curve than the circle of equal area: the
difference is largest at ingress and egress, where the projected shape changes what is covered
fastest.

The signal is small -- tens of parts per million (Seager & Hui, 2002; Barnes & Fortney, 2003) --
which is why the model offers several accuracy levels. At that amplitude the numerical error of the
model itself is a real consideration, not a formality.

The planet's projection is described by three parameters beyond the usual ones:

``k``
    The projected semi-major axis in stellar radii, i.e. the radius ratio.

``f``
    The flattening, :math:`f = (a - b)/a`, where :math:`a` and :math:`b` are the projected
    semi-major and semi-minor axes. Zero for a spherical planet.

``alpha``
    The projected obliquity in radians, measured from the sky-plane x-axis -- the direction of
    orbital motion at mid-transit -- to the projected semi-major axis.

Accuracy levels
---------------

.. list-table::
    :header-rows: 1
    :widths: 26 30 44

    * - Setting
      - Intersection areas
      - Limb darkening
    * - default
      - θ-sampled scanlines
      - Mean intensity over an area-equivalent circle
    * - ``exact_areas=True``
      - Analytic ellipse-circle
      - Mean intensity over an area-equivalent circle
    * - ``exact_ld=True``
      - θ-sampled scanlines
      - Integrated over the exact elliptical footprint
    * - both
      - Analytic ellipse-circle
      - Integrated over the exact elliptical footprint

The **default** is below the ppm level for a spherical planet but can reach tens of ppm for a
strongly oblate planet in a grazing geometry.

**``exact_areas``** replaces the scanline intersection areas with an analytic ellipse-circle
intersection algorithm accurate to machine precision, removing the geometric discretisation error
but keeping the mean-intensity limb darkening approximation.

**``exact_ld``** integrates the limb darkening over the planet's exact elliptical footprint, which
brings the model error below the ppm level for every tested geometry at roughly 25 times the cost
of the default. Combined with ``exact_areas``, the integration is also free of the scanline
resolution floor.

Both can be set in the initialiser and overridden per call in `evaluate`, which makes it easy to
run a fit with the fast settings and check the result with the accurate ones.

The scanline discretisation error falls as ``nlines**-2`` in ordinary geometries and ``nlines**-1.5``
in grazing ones; the annulus error of the exact-footprint integration falls as ``nannuli**-2``.

.. note::

    The accuracy levels are validated against direct numerical integration of the limb-darkened
    stellar disk in ``tests/test_opmodel.py``.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import OPModel

    time = linspace(-0.1, 0.1, 1000)

    tm = OPModel('quadratic')
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, f=0.1, alpha=0.3, ldc=[0.2, 0.1],
                       t0=0.0, p=1.0, a=3.0, i=0.5*pi)

    # Check the result with the accurate settings
    flux_exact = tm.evaluate(k=0.1, f=0.1, alpha=0.3, ldc=[0.2, 0.1],
                             t0=0.0, p=1.0, a=3.0, i=0.5*pi,
                             exact_areas=True, exact_ld=True)

API
---

.. autoclass:: pytransit.models.roadrunner.opmodel.OblatePlanetModel
    :members: evaluate
    :special-members: __init__
