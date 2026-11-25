Gravity-darkened model
======================

:class:`~pytransit.models.gdmodel.GravityDarkenedModel` implements the transit model of Barnes
(ApJ 705, 683, 2009) for a fast-rotating, oblate, gravity-darkened star.

The physics
-----------

A rapidly rotating star is not a sphere and not uniformly bright. Centrifugal force flattens it, so
its equatorial radius exceeds its polar radius. The flattening reduces the effective surface
gravity at the equator, and by von Zeipel's theorem the local effective temperature scales as a
power of the local gravity,

.. math::

    T_\mathrm{eff} \propto g^{\beta},

so the equator is cooler and dimmer than the poles. This is *gravity darkening*.

For a transit this changes everything. The stellar disk is no longer circular, its brightness is no
longer radially symmetric, and the light curve depends on where the planet's path crosses the star
relative to the stellar spin axis. A transit across the hot pole is deeper than one across the cool
equator, and the light curve becomes asymmetric unless the orbit is aligned with the stellar
equator. That asymmetry carries information about the spin-orbit angle.

Because the disk is not radially symmetric, none of the other PyTransit models apply: they all
assume a radially symmetric intensity profile. This model computes the brightness of the stellar
surface point by point and integrates it numerically instead. `How it works`_ describes the
integration, and `Accuracy and resolution`_ how its error depends on the three settings that
control it. :doc:`../features/gdmodel` shows what the model's signal looks like in practice.

.. plot::
    :context:
    :nofigs:

    from _figures.gdmodel import plot_surface_maps, plot_integration

Parameters
----------

Beyond the usual transit parameters, the model takes

.. list-table::
    :header-rows: 1
    :widths: 16 84

    * - Name
      - Description
    * - ``rho``
      - Stellar density in g/cm³.
    * - ``rperiod``
      - Stellar rotation period in days. This sets the oblateness.
    * - ``tpole``
      - Effective temperature at the stellar pole in K.
    * - ``phi``
      - Stellar obliquity to the plane of the sky in radians.
    * - ``beta``
      - Gravity darkening exponent. About 0.25 for radiative envelopes, 0.08 for convective ones.
    * - ``l``
      - Orbital azimuth angle in radians, i.e. the projected spin-orbit angle.

.. plot::
    :context: close-figs

    from _figures.gdmodel import plot_geometry
    plot_geometry()

The two angles place the star and the orbit relative to the observer. The projected spin axis of
the star always points up in the model's frame, so the orientation of the star on the sky is not a
parameter.

- ``phi`` tilts the spin axis out of the plane of the sky (right panel). At zero the star is seen
  equator-on, with both poles on the limb. A positive value tilts the upper pole towards the
  observer, which brings it onto the visible disk, and at :math:`\pi/2` the star is seen pole-on.
- ``l`` rotates the planet's path about the centre of the disk (left panel). At zero the planet
  moves from left to right, parallel to the stellar equator, and crosses below the centre of the
  disk at a distance of the impact parameter. A positive value rotates the path counterclockwise.

The figure shows the example star used on this page, with ``phi = 0.5`` and ``l`` of 60°.

Because the surface brightness varies with temperature, the model needs to convert temperature to
flux in the observed passband. The `filters` argument takes an effective wavelength, a
:class:`~pytransit.contamination.filter.Filter`, or a list of filters, and the `model` argument
selects the stellar spectra used: ``'blackbody'``, ``'husser2013'``, or ``'bt-settl'``. See
:doc:`../stars` and :doc:`../contamination`.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import GravityDarkenedModel
    from pytransit.contamination import BoxcarFilter

    time = linspace(-0.12, 0.12, 500)

    tm = GravityDarkenedModel(filters=BoxcarFilter('TESS', 600, 1000),
                              model='blackbody', tmin=6000, tmax=9000)
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, rho=0.5, rperiod=0.5, tpole=8000.0, phi=0.3, beta=0.25,
                       ldc=[0.2, 0.1], t0=0.0, p=2.0, a=5.0, i=0.5*pi, l=0.4)

Unlike the other models, `evaluate` takes a single parameter vector per call.

.. note::

    Before PyTransit 2.10.1 the model had no ``evaluate`` method, and ``evaluate_ps`` was its only
    entry point. ``evaluate_ps`` is deprecated and will be removed in PyTransit 3.0; ``evaluate``
    takes the same arguments, and additionally accepts a scalar ``k`` and a list ``ldc``.

How it works
------------

The model works in the plane of the sky, with the star's equatorial radius as the unit of length
and the projected spin axis along the :math:`y` axis. An evaluation has three stages. The stellar
parameters give the star's shape and the brightness of every point on its surface. The planet's
path across that surface gives a table of the mean brightness behind the planet. The light curve
then follows from the table and the exact area the planet covers.

The shape of the star
^^^^^^^^^^^^^^^^^^^^^

The star is an oblate spheroid whose flattening follows from its rotation rate
:math:`\omega = 2\pi / P_\mathrm{rot}` and mean density :math:`\rho`,

.. math::

    f = \frac{R_\mathrm{eq} - R_\mathrm{pole}}{R_\mathrm{eq}} = \frac{3 \omega^2}{8 \pi G \rho}.

This is why the model takes the stellar density and rotation period, ``rho`` and ``rperiod``,
instead of the flattening itself: the two parameters set both the shape and the gravity. The
equatorial radius comes from the ``rstar`` argument of the initialiser, and the stellar mass is the
density times the volume of the spheroid, :math:`M = \tfrac{4}{3} \pi R_\mathrm{eq}^3 (1 - f) \rho`.

Seen from the side, the star's outline is an ellipse with a flattening of :math:`f`. Tilting the
spin axis by the angle :math:`\phi` (``phi``) out of the plane of the sky makes the outline
rounder, and seen pole-on it is a circle. The outline is always an ellipse, with the flattening

.. math::

    f_\mathrm{eff} = 1 - \sqrt{(1 - f)^2 \cos^2\phi + \sin^2\phi}.

From gravity to brightness
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a point :math:`(x, y)` on the sky inside the outline, the model solves for the depth
:math:`z` at which the line of sight meets the spheroid, which is a quadratic equation. That gives
the point's position on the stellar surface, and the brightness follows in four steps.

1. **Effective gravity.** The local gravity is the sum of the star's gravitational attraction,
   towards its centre, and the centrifugal acceleration, away from the spin axis:

   .. math::

       \mathbf{g} = -\frac{G M}{r^2}\,\hat{\mathbf{r}} + \omega^2 r_\perp\,\hat{\mathbf{r}}_\perp,

   where :math:`r` is the distance from the centre and :math:`r_\perp` the distance from the spin
   axis. The two oppose each other at the equator and the centrifugal term vanishes at the poles.

2. **Temperature.** Von Zeipel's law gives the local effective temperature from the gravity
   relative to the pole, :math:`T = T_\mathrm{pole}\,(g / g_\mathrm{pole})^\beta`.

3. **Flux in the passband.** The initialiser tabulates the flux through each passband against
   temperature, from blackbody spectra or model atmospheres, for 200 temperatures between ``tmin``
   and ``tmax``. The evaluation reads the table by linear interpolation, so a blackbody and a
   model atmosphere cost the same.

4. **Limb darkening.** The flux is multiplied by a quadratic limb darkening law in
   :math:`\mu`, the cosine of the angle between the line of sight and the surface normal. The
   normal is that of the spheroid, not of a sphere, so the limb darkening follows the star's
   actual shape. Setting ``accurate_mu=False`` uses the direction to the star's centre instead,
   which is a cruder approximation.

The figure shows the three quantities for a star of 1.8 solar radii with a density of 0.48 g/cm³
that rotates once in 11 hours, with its spin axis tilted by 0.5 radians towards the observer. The
star is flattened by 9%, its equatorial gravity is two thirds of the polar one, and with
:math:`\beta = 0.22` its equator is 720 K cooler than its 8500 K pole.

.. plot::
    :context: close-figs

    plot_surface_maps()

The brightness map is not a copy of the temperature map. Limb darkening dims the edge of the disk
whatever its temperature, so the brightest point lies between the pole and the centre of the disk.
The contrast also depends on the passband: a given temperature difference changes the brightness
more at short wavelengths than at long ones.

.. note::

    A surface temperature outside the range from ``tmin`` to ``tmax`` has no entry in the flux
    table, and the model then returns NaN for the in-transit points. Choose the range to cover the
    star from its equator to its pole for every parameter set a fit may try. The default range,
    5000 to 7000 K, does not suit a hot star.

The light curve
^^^^^^^^^^^^^^^

The flux at a given time is

.. math::

    F(t) = 1 - \frac{\bar{I}(t)\, A(t)}{F_\star},

where :math:`F_\star` is the flux of the whole stellar disk, :math:`A` the area of the disk the
planet covers, and :math:`\bar{I}` the mean brightness of that area. The model computes the three
separately.

- **The stellar flux** is the mean brightness over a grid of ``sres`` by ``sres`` points covering
  the disk, times the area of the outline, :math:`\pi (1 - f_\mathrm{eff})`.
- **The mean brightness behind the planet** is the average over a grid of points covering the
  planet's disk, ``pres`` points across. Points that fall outside the star are left out of the
  average, so across the limb the mean is that of the covered part of the star.
- **The covered area** is the exact intersection area of the planet's circle and the star's
  elliptical outline. Because the area is exact, the sample points only have to estimate a mean
  brightness, which varies smoothly, and not the shape of the overlap, which a grid of points
  resolves poorly.

The mean brightness is not computed for every exposure. The brightness of the disk depends on two
coordinates, but during one evaluation the planet only ever follows one path across it. The model
therefore tabulates the mean brightness at ``tres`` times spaced evenly from the first contact to
the last, and reads the table by linear interpolation for every exposure, and for every subsample
of a supersampled one. This is the same division of work as in the
:doc:`RoadRunner model <roadrunner>`, except that the table runs along the planet's path instead
of along the distance from the disk centre, and so belongs to one orbit only.

.. plot::
    :context: close-figs

    plot_integration()

The left panel shows the table nodes for the default ``tres`` of 60 and a planet crossing the
example star with a projected spin-orbit angle of 60°. The middle panel shows the planet at the
circled node, with the sample points of the default ``pres`` of 20: the white ones are averaged and
the grey ones, which are off the star, are not. The right panel is the resulting table. The planet
moves from the cooler lower part of the disk towards the hot pole, so the brightness behind it
rises through most of the transit. This is the asymmetry of the light curve. Nodes where the
planet is entirely off the star take the value of the nearest node that is not.

The planet's position comes from a series expansion of the Keplerian orbit around the transit
centre, computed in a frame where the planet moves along the :math:`x` axis and then rotated by the
orbital azimuth angle ``l``. Exposures far from the transit are set to one without further work.

An evaluation with the default settings computes the surface brightness at about 20 000 points for
the table and 25 600 for the stellar flux, for each passband, and this fixed cost dominates. An
evaluation takes 2.9 ms for 100 exposures, 3.0 ms for 1000, and 4.0 ms for 10 000, measured on an
AMD Ryzen 7 5800X, so each exposure adds about 0.1 µs.

What the model leaves out
^^^^^^^^^^^^^^^^^^^^^^^^^

- **The limb darkening is quadratic**, with the same coefficients over the whole surface, although
  the temperature and gravity vary.
- **The gravity is that of a point mass plus rotation.** The star rotates as a solid body, and its
  shape is a spheroid instead of a surface of constant potential.
- **One orbit per call.** The table belongs to one path across the star, so ``evaluate`` takes a
  single parameter vector.
- **Radius ratios per passband share one table.** When ``k`` has one value per passband, the mean
  brightness table is computed with the first one, and the others only enter through the covered
  area.

Accuracy and resolution
-----------------------

The model integrates numerically over a discretised stellar surface and planet, so its accuracy is
set by three resolutions: ``sres`` for the star, ``pres`` for the planet, and ``tres`` for the
orbit. For a radius ratio of 0.1, measured against the analytic Mandel & Agol model for a
non-rotating star, and against the brute-force evaluation for a rapidly rotating star with
misaligned orbits (whose own error at 1500 pixels is 10 to 15 ppm):

.. list-table::
    :header-rows: 1
    :widths: 28 24 24 24

    * - ``sres``, ``pres``, ``tres``
      - Non-rotating star
      - Rotating star
      - Time per 1000 points
    * - 160, 20, 60 (default)
      - 30 ppm
      - 50 ppm
      - 3 ms
    * - 160, 40, 240
      - 10 ppm
      - 26 ppm
      - 24 ms

``sres`` sets the accuracy of the normalising stellar flux, and ``pres`` that of the mean stellar
brightness behind the planet, which matters most at ingress and egress. ``tres`` sets how well the
table follows the mean brightness, which changes fastest near the limb, where the linear
interpolation is least accurate. Before PyTransit 2.10.1 the
defaults were ``sres=80, pres=5``, which gave errors of about 165 ppm.

``evaluate_brute`` renders the stellar disk as an image and sums the pixels behind the planet. It
is far slower, but its only approximation is the image resolution, which makes it the reference for
checking the fast model in a new region of parameter space.

Visualisation
-------------

``visualize`` draws the gravity-darkened stellar disk with the planet's path across it, which is by
far the quickest way to check that a parameter set means what you think it means.

.. code-block:: python

    tm.visualize(k=0.1, p=2.0, rho=0.5, b=0.3, e=0.0, w=0.0, alpha=0.4,
                 rperiod=0.5, tpole=8000.0, istar=0.3, beta=0.25, ldc=[0.2, 0.1])

API
---

.. autoclass:: pytransit.models.gdmodel.GravityDarkenedModel
    :members: evaluate, evaluate_brute, visualize
    :special-members: __init__
