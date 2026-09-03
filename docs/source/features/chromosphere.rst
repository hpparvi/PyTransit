Chromosphere model
==================

.. note::

    This page is a stub. Worked examples and figures will be added in a future release. The full
    API and usage documentation is on the
    :doc:`chromosphere reference page <../models/chromosphere>`.

A stellar photosphere is limb *darkened*: the disk is brightest at its centre, because looking
straight down we see deep, hot layers, while a line of sight near the limb reaches only shallower,
cooler ones.

An optically thin, spherically symmetric emitting shell -- a chromosphere seen in an emission line
such as H\ :math:`\alpha` or Ca II K -- does the opposite. A line of sight near the limb passes
through a longer path of emitting material than one through the centre, so the projected disk is
brightest at its *edge*.

This inverts the transit. Instead of the familiar rounded profile, the planet blocks the most light
near ingress and egress and the least at mid-transit. For a planet with
:math:`k = 0.1` on a central transit, the shell model blocks about half as much light at
mid-transit as a uniform disk would, and about three times as much near the limb.

No limb darkening law can reproduce that shape, however flexible, because the profile does not
merely differ in degree -- it runs the wrong way. That is why
:class:`~pytransit.models.ma_chromosphere.ChromosphereModel` exists as a separate implementation of
the Schlawin et al. (ApJL 722, L75, 2010) model rather than as another entry in the limb darkening
catalogue.

The profile follows from the geometry of the shell, so the model takes no limb darkening
coefficients at all.

.. seealso::

    :doc:`../models/chromosphere` for the API.
