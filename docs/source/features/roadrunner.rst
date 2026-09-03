RoadRunner model
================

:class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` (Parviainen, MNRAS 499, 1633, 2020)
is PyTransit's recommended general-purpose transit model, and the one to reach for unless a
specialised model fits your problem better. What sets it apart is that the stellar intensity
profile is an *input* rather than something baked into the derivation.

.. plot::
    :context:
    :nofigs:

    from _figures.roadrunner import plot_laws, plot_profile_and_transit, plot_accuracy

Geometry and profile, separated
-------------------------------

Every classical transit model is an analytic solution to one specific integral: a limb-darkened
disk overlapped by an opaque circle, solved separately for the quadratic law, the power-2 law, and
so on. Add a new limb darkening law and you need a new derivation. Some laws have no closed form
at all.

RoadRunner breaks that coupling by splitting the problem in two.

The **geometry** -- how much of the stellar disk the planet covers at each distance from the disk
centre -- depends only on the radius ratio and the projected separation, not on how bright the star
is anywhere. It is solved numerically once, when the model is created, and stored as a table of
weights.

The **intensity profile** then enters only as samples on a fixed grid of normalised distances from
the disk centre. The transit is a weighted sum over that grid.

Everything expensive lives in the first half, and the first half does not care what the profile
looks like. That single fact produces the three properties the rest of this page demonstrates: any
radially symmetric profile works, the choice costs essentially nothing, and accuracy is a
discretisation setting rather than a property of the law.

The law is a one-line choice
----------------------------

The built-in laws are selected by name in the initialiser. Nothing else about the call changes.

.. plot::
    :context: close-figs
    :include-source:

    from numpy import pi, linspace
    from pytransit import RoadRunnerModel

    window = 3.5 / 24
    time = linspace(-0.5 * window, 0.5 * window, 1500)
    k, t0, p, a, i = 0.1, 0.0, 4.0, 13.0, 0.49 * pi

    laws = {'linear':    [0.4],
            'quadratic': [0.3, 0.1],
            'power-2':   [0.6, 0.5],
            'nonlinear': [0.4, 0.3, 0.2, 0.1]}

    fluxes = {}
    for name, ldc in laws.items():
        tm = RoadRunnerModel(name)
        tm.set_data(time)
        fluxes[name] = tm.evaluate(k, ldc, t0, p, a, i)

    plot_laws(laws, fluxes, time)

Eleven laws ship with the model, listed in :doc:`../guide/limb_darkening`. Two of them --
``'quadratic-tri'`` and ``'power-2-pm'`` -- are reparametrisations designed to behave well as free
parameters in a fit rather than new physics.

Any profile you can write down
------------------------------

The interesting part is what happens when you stop passing a name. RoadRunner accepts any callable
``f(mu, pv)`` returning the intensity at :math:`\mu = \cos\gamma`, so long as it is
Numba-compilable.

Nothing requires that function to be a sensible limb darkening law. Here is a profile that
oscillates across the disk:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import sqrt, sin, ceil
    from numba import njit

    @njit(fastmath=True)
    def ld_wavy(mu, pv):
        z = sqrt(1.0 - mu**2)
        return 1.0 - (1.0 - z) * sin(ceil(pv[0]) * 2 * pi * z)

    tm = RoadRunnerModel(ld_wavy, ng=200)
    tm.set_data(time)
    flux = tm.evaluate(k, [5], t0, p, a, i)

    plot_profile_and_transit(ld_wavy, [5], flux, time, 'A wavy profile')

The rings of alternating brightness show up as ripples in the light curve as the planet crosses
them. No analytic transit model could integrate this profile; RoadRunner does not notice that
anything unusual has happened.

A discontinuous profile is no harder:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import where

    @njit
    def ld_step(mu, pv):
        z = sqrt(1.0 - mu**2)
        return where(z < 0.5, 0.8, 0.4)

    tm = RoadRunnerModel(ld_step, ng=200)
    tm.set_data(time)
    flux = tm.evaluate(k, [5], t0, p, a, i)

    plot_profile_and_transit(ld_step, [5], flux, time, 'A step profile')

The light curve reads straight off the profile. While the planet is entirely over the dim outer
annulus the flux sits on a flat shelf, then drops sharply the moment it crosses onto the bright
inner disk at :math:`z = 0.5`, and the corners at those crossings are as sharp as the profile's
own discontinuity.

These two are deliberately unphysical, and that is the point: if the model handles them, the
realistic cases -- a numerically tabulated profile from a stellar atmosphere grid, a law nobody has
derived a transit solution for, a profile with a starspot-driven feature -- are unremarkable.

.. tip::

    Supplying only a profile makes the model integrate it over the disk numerically on every
    evaluation. If the disk integral :math:`2\pi \int_0^1 I(\mu)\, z\, dz` has a closed form, pass
    a ``(profile, integral)`` pair instead and the model uses it directly. All the built-in laws
    are defined this way.

The law is nearly free
----------------------

Because the expensive half of the calculation is the geometry, and the geometry is shared, the
choice of law barely registers. Timing a 1500-point light curve on one machine, from a
one-coefficient linear law to a six-coefficient general one:

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Law
      - Coefficients
      - Time per evaluation
    * - ``'linear'``
      - 1
      - 0.148 ms
    * - ``'quadratic'``
      - 2
      - 0.146 ms
    * - ``'power-2'``
      - 2
      - 0.147 ms
    * - ``'nonlinear'``
      - 4
      - 0.146 ms
    * - ``'general'``
      - 6
      - 0.147 ms

A 1.4% spread across a sixfold change in the number of coefficients. Choosing a more flexible law
is a modelling decision, not a performance one.

Accuracy and the discretisation
-------------------------------

Splitting the problem buys flexibility at the cost of integrating the profile numerically, and
that integration is where the accuracy is decided. For every radius ratio the model computes the
mean intensity under the planet as a function of the grazing parameter by Gauss quadrature matched
to the geometry: Gauss-Jacobi rules absorb the square-root zeros of the planet's angular extent
at its contacts, substitutions regularise the limb, and the integration variable is chosen per
regime so that each integrand is smooth. The result is tabulated in two segments split at the
limb contact
-- the one place the exact function has a kink -- and read with a cubic during the evaluation.

Two settings control it: `nq`, the number of quadrature nodes, and `ng`, the size of the table.
The profile itself is tabulated once per evaluation on a fixed grid, which is why a tabulated
stellar-atmosphere profile costs the same as an analytic law.

For the quadratic law there is an exact analytic solution to compare against --
:class:`~pytransit.models.ma_quadratic.QuadraticModel` -- so the difference is RoadRunner's
integration error alone.

.. plot::
    :context: close-figs

    plot_accuracy()

With the default settings the error is about 0.3 ppm at :math:`k = 0.02`, 1.5 ppm at
:math:`k = 0.1`, and 5.4 ppm at :math:`k = 0.3`, taking the worst case over impact parameters up to
0.9 -- three to five times better than the annulus discretisation this replaced, at the same
cost. ``nq=12, ng=200`` roughly halves that again, and ``nq=16, ng=400`` is below 1 ppm at every
radius ratio.

For most work the defaults are appropriate: a few ppm is far below the noise of any real transit
observation. Raise the resolution when the *model* error would compete with the signal you are
after, as it does for planetary oblateness or for very large planets.

Profiles from stellar atmospheres
---------------------------------

The most physically motivated use of the flexibility is to skip analytic laws entirely.
:class:`~pytransit.models.ldmodel.LDModel` is an interface for profiles that come from a stellar
atmosphere model, and :class:`~pytransit.models.ldtkldm.LDTkLDModel` implements it on top of
`LDTk <https://github.com/hpparvi/ldtk>`_, computing profiles from the Husser et al. (2013) PHOENIX
specific intensity spectra.

.. code-block:: python

    from pytransit import RoadRunnerModel
    from pytransit.models.ldtkldm import LDTkLDModel
    from pytransit.contamination import sdss_g, sdss_r, sdss_i, sdss_z

    ldm = LDTkLDModel(pbs=(sdss_g, sdss_r, sdss_i, sdss_z),
                      teff=(5500, 100), logg=(4.5, 0.1), metal=(0.0, 0.1))

    tm = RoadRunnerModel(ldm)

The free parameters are then the stellar parameters rather than abstract coefficients, and the
profile follows the model atmospheres directly.

.. seealso::

    :doc:`../models/roadrunner` for the API and the initialiser arguments,
    :doc:`../guide/limb_darkening` for the built-in laws and how to supply your own, and
    :doc:`../highlights` for what the shared data setup adds on top.
