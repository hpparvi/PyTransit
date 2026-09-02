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

    from numpy import pi, array
    from pytransit import GravityDarkenedModel
    from pytransit.contamination import BoxcarFilter

    tm = GravityDarkenedModel(filters=BoxcarFilter('TESS', 600, 1000),
                              model='blackbody', tmin=6000, tmax=9000)
    tm.set_data(time)

    flux = tm.evaluate_ps(k=[0.1], rho=0.5, rperiod=0.5, tpole=8000.0, phi=0.3, beta=0.25,
                          ldc=array([0.2, 0.1]), t0=0.0, p=2.0, a=5.0, i=0.5*pi, l=0.4)

.. warning::

    **This model has no supported evaluation method.** Unlike every other model it never gained an
    ``evaluate`` method, so its only entry point is ``evaluate_ps`` -- which is deprecated as of
    PyTransit 2.9 and scheduled for removal in 3.0. Until the model gains an ``evaluate`` method,
    calling it emits a deprecation warning that has no alternative to point to. The method itself
    is documented in :doc:`../api/deprecated`.

    ``evaluate_brute`` provides a slower brute-force evaluation useful for checking the fast one,
    and is not deprecated.

.. note::

    This model needs ``k`` as a 1D array and ``ldc`` as an ndarray. A scalar ``k`` raises a Numba
    typing error rather than being broadcast.

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
    :members: evaluate_brute, visualize
    :special-members: __init__
