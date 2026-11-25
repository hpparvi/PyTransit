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

How it works
------------

The model is the :doc:`RoadRunner model <roadrunner>` with the circle replaced by an ellipse. The
flux at a given time is

.. math::

    F = 1 - \frac{\bar{I} A}{F_\star},

where :math:`F_\star` is the flux of the stellar disk, :math:`A` the area of the disk the planet
covers, and :math:`\bar{I}` the mean stellar intensity over that area. For a circular planet both
have fast and accurate solutions: the area of two overlapping circles is a closed-form expression,
and the mean intensity depends only on the planet's distance from the disk centre, so RoadRunner
tabulates it once per evaluation.

An ellipse breaks both. The covered area has no simple closed form, and the mean intensity depends
on the planet's orientation as well as its position. The model offers a fast and an accurate
answer to each, and the two choices, ``exact_areas`` and ``exact_ld``, are independent.

.. plot::
    :context: close-figs

    from _figures.opmodel import plot_integration_schemes
    plot_integration_schemes()

The figure sketches the three constructions described below. The planet is drawn far larger and
flatter than a real one, with fewer scanlines and annuli than the model uses.

The covered area by scanlines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the planet is cut into ``nlines`` horizontal chords (left panel). The end points of the
chords depend only on the planet's shape and orientation, so they are computed once per
evaluation. For every exposure, each chord is moved to the planet's position and clipped against
the stellar disk at its height, and the covered area is the weighted sum of the clipped lengths
(blue).

The chords are not spaced evenly. Near the top and bottom of the ellipse the chord length grows
like the square root of the distance to the tip, which evenly spaced lines integrate poorly. The
lines are placed at :math:`y = h \sin\theta` with :math:`\theta` spaced evenly, which crowds them
towards the tips and turns the square root into a smooth function of :math:`\theta`. This is the
same device RoadRunner uses at the edges of the planet. The error then falls as ``nlines**-2``.
In a grazing transit the chords are also cut off where the stellar limb itself is horizontal, which
the spacing does not help with, and the error falls as ``nlines**-1.5``.

The exact covered area
^^^^^^^^^^^^^^^^^^^^^^

With ``exact_areas`` the area is computed from the boundary of the covered region (middle panel).
The region is bounded by arcs of the ellipse and arcs of the stellar limb, joined where the two
curves cross.

- **The crossing points.** Following the ellipse around with an angle :math:`t`, the squared
  distance from the disk centre minus one is

  .. math::

      g(t) = A \cos^2 t + B \cos t + C \sin t + D,

  which is negative inside the star and positive outside. Its roots are the crossing points, and
  there are at most four. The model scans :math:`g` on a fixed grid, subdivides every cell where
  a bound on the curvature of :math:`g` cannot rule out a root, and refines each root with Newton
  iteration. The bound matters when the ellipse barely touches the limb, where two roots lie close
  together and a plain sign check would miss them.
- **The area.** Green's theorem turns the area of a region into an integral along its boundary,
  :math:`A = \tfrac{1}{2} \oint (x\,\mathrm{d}y - y\,\mathrm{d}x)`, and this integral has a
  closed form for an arc of a circle and for an arc of an ellipse. The area is the sum of one such
  term per arc.

The result is accurate to machine precision, and costs a root search per exposure instead of a sum
over chords.

The limb darkening behind the planet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the mean intensity is that behind a circular planet of the same area at the same
position, with a radius ratio of :math:`k \sqrt{1 - f}` (dashed circle in the right panel). This
is RoadRunner's mean intensity table, built once per evaluation and read once per exposure, so the
limb darkening costs the same as for a spherical planet. The circle and the ellipse cover the same
area but not the same parts of the star. Where the intensity changes slowly this hardly matters,
but at the limb it changes fast, and the approximation is what limits the default accuracy.

With ``exact_ld`` the limb darkening is integrated over the area the ellipse actually covers
(right panel). The star is cut into ``nannuli`` rings centred on the disk, spanning the distances
the planet reaches. The intensity is constant along a ring, so the blocked flux is the sum over
the rings of the area the planet covers in each, times the ring's intensity:

.. math::

    \bar{I} A = \sum_j \left(A_{j} - A_{j+1}\right) I(\mu_j),

where :math:`A_j` is the area of the ellipse inside a disk of the ring's outer radius. The area
inside a disk is the same problem as the covered area, with a smaller disk, so it is solved with
the scanlines or, with ``exact_areas``, exactly. The rings are spaced evenly in :math:`\mu`, which
makes them narrow at the limb where the intensity changes fastest, and the error falls roughly as
``nannuli**-2``.

This is where the cost comes from. The default settings compute one area per exposure, and
``exact_ld`` computes ``nannuli + 1`` of them, 21 by default.

What is shared with RoadRunner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The limb darkening laws, the stellar intensity profile tabulated in :math:`\mu`, the orbit, the
supersampling, the passband-dependent radius ratios, and the evaluation of many parameter vectors
in one call all work as in :doc:`RoadRunnerModel <roadrunner>`, and ``nthreads`` parallelises both
the per-vector setup and the flux computation. With :math:`f = 0` the model gives RoadRunner's light
curve to well below 1 ppm.

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

The **default** is below the ppm level for a spherical planet, but its error grows roughly in
proportion to the flattening and is concentrated at ingress and egress: about 30 ppm for
:math:`f = 0.1` and up to about 80 ppm for :math:`f = 0.3`. This is enough when the oblateness is a
nuisance parameter, but not when it is the measurement; :doc:`../features/opmodel` compares the
settings with the oblateness signal itself.

**``exact_areas``** replaces the scanline intersection areas with an analytic ellipse-circle
intersection algorithm accurate to machine precision, removing the geometric discretisation error
but keeping the mean-intensity limb darkening approximation.

**``exact_ld``** integrates the limb darkening over the planet's exact elliptical footprint, which
brings the model error below the ppm level for every tested geometry at roughly 25 times the cost
of the default. Combined with ``exact_areas``, the integration is also free of the scanline
resolution floor.

Both can be set in the initialiser and overridden per call in `evaluate`, which makes it easy to
run a fit with the fast settings and check the result with the accurate ones.

The two resolutions, ``nlines`` (100 by default) and ``nannuli`` (20), are set in the initialiser.
`How it works`_ describes what they control and how fast the error falls with each.

.. note::

    The accuracy levels are validated against direct numerical integration of the limb-darkened
    stellar disk in ``tests/test_opmodel.py``.

API
---

.. autoclass:: pytransit.models.roadrunner.opmodel.OblatePlanetModel
    :members: evaluate
    :special-members: __init__
