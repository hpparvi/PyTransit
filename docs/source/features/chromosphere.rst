Chromosphere model
==================

A stellar photosphere is limb *darkened*. Looking at the centre of the disk we see deep, hot
layers, while a line of sight near the limb reaches only shallower, cooler ones. An optically thin
emitting shell, such as a chromosphere seen in an emission line like H\ :math:`\alpha` or Ca II K,
does the opposite: a line of sight near the limb passes through a longer path of emitting gas, so
the disk is brightest at its edge. For a thin shell the surface brightness grows as :math:`1/\mu`,
without limit at the limb.

:class:`~pytransit.models.ma_chromosphere.ChromosphereModel` computes a transit over such a shell
using the analytic solution of Schlawin et al. (ApJL 722, L75, 2010). The profile follows from the
geometry of the shell, so the model takes no limb darkening coefficients.

.. plot::
    :context:
    :nofigs:

    from _figures.chromosphere import plot_profiles_and_transits, plot_limb_darkening_fits

An inverted transit
-------------------

The same planet transiting a limb-brightened shell, a uniform disk, and a limb-darkened
photosphere:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import pi, linspace
    from pytransit import ChromosphereModel, RoadRunnerModel

    time = linspace(-0.1, 0.1, 1500)
    k, t0, p, a, i = 0.1, 0.0, 4.0, 13.0, 0.5 * pi

    tm = ChromosphereModel()
    tm.set_data(time)
    fluxes = {'shell': tm.evaluate(k, t0, p, a, i)}

    for law, ldc in (('uniform', []), ('quadratic', [0.4, 0.25])):
        tm = RoadRunnerModel(law)
        tm.set_data(time)
        fluxes[law] = tm.evaluate(k, ldc, t0, p, a, i)

    plot_profiles_and_transits(time, fluxes, [0.4, 0.25])

With the profiles normalised to the same total flux, the shell is half as bright as a uniform disk
at its centre and brighter than it beyond about 0.87 stellar radii. The transit follows. At
mid-transit the planet covers the faintest part of the shell and blocks half as much light as it
would from a uniform disk, 5000 ppm instead of :math:`k^2 = 10\,000` ppm. Just inside the limb it
covers the brightest part, and the transit reaches its greatest depth, 1.6 times :math:`k^2`. A
limb-darkened transit is deepest in the middle; a transit over a shell is deepest at its edges.

Why a separate model
--------------------

RoadRunner accepts any radially symmetric profile, so a separate model for one profile needs a
reason. The reason is the limb, where the shell's brightness diverges.

Limb darkening laws cannot follow it. Fitting the shell transit with the standard laws, with the
radius ratio, orbit, impact parameter, transit centre, and limb darkening all free, and the
coefficients allowed to make the disk brighten towards the limb:

.. plot::
    :context: close-figs

    plot_limb_darkening_fits(time, fluxes['shell'])

The best fits leave residuals of up to 1260 ppm for the quadratic law, 1860 ppm for the power-2
law, and 530 ppm for the four-coefficient non-linear law, concentrated where the planet crosses the
limb. Even RoadRunner given the exact profile and its disk integral does not help: its quadrature
assumes a smooth profile, and the singularity at the limb leaves errors of more than 1% even at a
high resolution.

The analytic solution has neither problem. It agrees with a direct numerical integration of the
shell's brightness over the planet to within :math:`10^{-8}`, and it is fast, evaluating a
1500-point light curve in 0.08 ms.

.. seealso::

    :doc:`../models/chromosphere` for the API.
