Gravity-darkened star model
===========================

Every other model in PyTransit assumes that the star is a circular disk whose brightness depends
only on the distance from its centre. A rapidly rotating star is neither. Rotation flattens it, and
the flattening lowers the effective surface gravity at the equator. By von Zeipel's theorem the
local effective temperature follows the gravity,

.. math::

    T_\mathrm{eff} \propto g^{\beta},

so the equator is cooler and dimmer than the poles. This is *gravity darkening*.

:class:`~pytransit.models.gdmodel.GravityDarkenedModel` implements the model of Barnes (ApJ 705,
683, 2009) for a transit across such a star. Because the star's brightness depends on where its
spin axis points, so does the transit: a planet whose orbit is misaligned with the stellar equator
crosses regions of different temperature, and the light curve becomes asymmetric. The spin-orbit
angle is then measurable from photometry alone, without the spectroscopy a Rossiter-McLaughlin
measurement needs.

The model integrates numerically over the stellar surface, and converts the local temperature into
flux in the observed passband using either blackbody spectra or model atmospheres (see
:doc:`../stars` and :doc:`../contamination`).

.. plot::
    :context:
    :nofigs:

    from _figures.gdmodel import plot_gravity_darkening_signal

What gravity darkening does to a transit
----------------------------------------

An A-type star of 1.8 solar masses and radii, with a polar temperature of 8500 K and a rotation
period of 11 hours, is flattened by 9% and rotates at 194 km/s at its equator. A planet with a
radius ratio of 0.1 transits it at an impact parameter of 0.3, observed in a blue and a red
passband, for three projected spin-orbit angles :math:`\lambda`:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import pi, linspace, arccos
    from pytransit import GravityDarkenedModel
    from pytransit.contamination import BoxcarFilter

    time = linspace(-0.13, 0.13, 800)
    rho, rperiod, tpole, phi, beta = 0.48, 0.47, 8500.0, 0.5, 0.22
    ldc, t0, p, a, i = [0.3, 0.2], 0.0, 3.0, 6.0, arccos(0.3 / 6.0)

    models = {}
    for band, wl in (('blue', (400, 500)), ('red', (800, 900))):
        models[band] = GravityDarkenedModel(filters=BoxcarFilter(band, *wl), rstar=1.8,
                                            model='blackbody', tmin=5000, tmax=9500,
                                            sres=160, pres=40, tres=240)
        models[band].set_data(time)

    fluxes = {(band, l): tm.evaluate(0.1, rho, rperiod, tpole, phi, beta, ldc, t0, p, a, i, l)
              for band, tm in models.items() for l in (0.0, pi / 4, pi / 2)}

    plot_gravity_darkening_signal(time, fluxes, models, rho, rperiod, tpole, phi, beta,
                                  ldc, t0, p, a, i)

The upper row shows the star as the model sees it in the blue passband. Its spin axis is tilted
by :math:`\phi = 0.5` towards the observer, so the hot pole at the top is visible, and the planet
moves along the dashed line. The lower row shows what is left after fitting the light curve with
a standard transit model: a spherical, limb-darkened star, with the radius ratio, orbit, impact
parameter, transit centre, and limb darkening all free.

When the orbit is aligned with the stellar equator, the standard model absorbs the gravity
darkening almost entirely: less than 10 ppm remains. A misaligned orbit is different. The planet
first crosses the cooler, dimmer part of the disk and then passes near the hot pole, so the transit
is shallower than average early and deeper late. No spherical star produces that, and the residual
reaches about 580 ppm for :math:`\lambda = 45°` and 1500 ppm for :math:`\lambda = 90°` in the blue
passband.

The residual is about half as large in the red passband, because a given temperature difference
changes the surface brightness more at short wavelengths than at long ones. Observations in several
passbands therefore help to separate gravity darkening from other effects, and the model evaluates
each passband with its own temperature-to-flux conversion.

Accuracy and resolution
-----------------------

The model's accuracy is set by the resolution of the stellar surface, the planet, and the orbit.
With the default resolution, the model error is 30 to 50 ppm for a radius ratio of 0.1, which
is small compared with the signals above. The figure uses ``sres=160, pres=40, tres=240``, which
brings the error to 10 to 25 ppm, so that the small residuals of the aligned orbit show the physics
rather than the model:

.. list-table::
    :header-rows: 1
    :widths: 40 30 30

    * - ``sres``, ``pres``, ``tres``
      - Maximum error
      - Time per 1000 points
    * - 160, 20, 60 (default)
      - 30 to 50 ppm
      - 3 ms
    * - 160, 40, 240
      - 10 to 25 ppm
      - 24 ms

``evaluate_brute`` computes the same transit by rendering the stellar disk as an image and summing
the pixels behind the planet. It is much slower, but is a useful check when the model is used in a
new region of parameter space.

.. seealso::

    :doc:`../models/gdmodel` for the parameters, the accuracy settings, and the API;
    :doc:`../stars` and :doc:`../contamination` for the stellar spectra and passbands.
