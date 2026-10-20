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
assume a radially symmetric intensity profile. This model discretises the stellar surface and
integrates numerically instead.

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
intensity behind the planet, which matters most at ingress and egress. Before PyTransit 2.10.1 the
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
