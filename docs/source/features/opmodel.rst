Oblate planet model
===================

.. note::

    This page is a stub. Worked examples and figures will be added in a future release. The full
    API and usage documentation is on the
    :doc:`oblate planet reference page <../models/opmodel>`.

A rapidly rotating planet is flattened at its poles. Saturn's oblateness is about 0.1, Jupiter's
about 0.06, and a flattened planet projected on the sky is an ellipse rather than a circle. An
ellipse crossing a limb-darkened disk produces a slightly different light curve than the circle of
equal area, with the difference concentrated at ingress and egress where the projected shape
changes what gets covered fastest.

Measuring planetary oblateness would give a direct handle on rotation rates, which are otherwise
almost inaccessible for exoplanets. The obstacle is amplitude: the signal is tens of parts per
million (Seager & Hui, 2002; Barnes & Fortney, 2003).

At that level the *numerical* error of the transit model is a real consideration rather than a
formality, and this is what makes
:class:`~pytransit.models.roadrunner.opmodel.OblatePlanetModel` unusual. It offers several
accuracy levels that can be dialled to match the precision the science needs:

- the default, which approximates the intersection areas with sampled scanlines and the limb
  darkening with the mean intensity over an area-equivalent circle;
- ``exact_areas``, which replaces the scanline areas with an analytic ellipse-circle intersection
  accurate to machine precision;
- ``exact_ld``, which integrates the limb darkening over the planet's exact elliptical footprint,
  reaching sub-ppm accuracy for every tested geometry.

Both flags can be overridden per call, so a fit can run with the fast settings and be checked
against the accurate ones without rebuilding the model.

.. seealso::

    :doc:`../models/opmodel` for the API and the accuracy scaling of each setting.
