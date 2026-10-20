RoadRunner model
================

:class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` (Parviainen, MNRAS 499, 1633, 2020)
is PyTransit's recommended general-purpose transit model. Its distinguishing feature is that the
stellar intensity profile is an *input* to the model rather than something built into its
derivation. Any radially symmetric profile works, and the choice costs essentially nothing.

The flexibility does not cost speed either. For most realistic analyses RoadRunner is faster than
the analytic quadratic model of Mandel & Agol, and for long light curves four times faster, even
when evaluated to an accuracy of 1 ppm.

.. plot::
    :context:
    :nofigs:

    from _figures.roadrunner import plot_laws, plot_profile_and_transit

Geometry and profile, separated
-------------------------------

The classical transit models are analytic solutions to one specific integral: a limb-darkened disk
overlapped by an opaque circle, solved separately for each limb darkening law. A new law needs a
new derivation, and some laws have no closed form at all.

RoadRunner splits the problem in two. The **geometry** -- which parts of the stellar disk the
planet covers -- depends only on the radius ratio and the planet's position, not on how bright the
star is. The model turns it into a handful of points at which to read the stellar intensity, and a
weight for each reading. The **intensity profile** enters only through those readings, from a table
computed once per evaluation, so the model never sees the function that produced it.

Nothing in this depends on the form of the profile, and the rest of this page shows what follows
from that. :doc:`../models/roadrunner` describes how the readings and their weights are chosen.

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

Eleven laws ship with the model, listed in :doc:`../guide/limb_darkening`. Two of them,
``'quadratic-tri'`` and ``'power-2-pm'``, are not new physics but reparametrisations that behave
better as free parameters in a fit.

Any profile you can write down
------------------------------

Instead of a name, RoadRunner also accepts any Numba-compilable function ``f(mu, pv)`` that returns
the intensity at :math:`\mu = \cos\gamma`. Nothing requires that function to be a sensible limb
darkening law. Here is a profile that oscillates across the disk:

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
them. No analytic transit model could integrate this profile, but for RoadRunner it is an ordinary
input.

A step in brightness works the same way:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import tanh

    @njit
    def ld_step(mu, pv):
        z = sqrt(1.0 - mu**2)
        return 0.6 + 0.2 * tanh((0.5 - z) / pv[0])

    tm = RoadRunnerModel(ld_step, ng=200)
    tm.set_data(time)
    flux = tm.evaluate(k, [0.05], t0, p, a, i)

    plot_profile_and_transit(ld_step, [0.05], flux, time, 'A step profile')

The light curve follows the profile directly. While the planet is entirely over the dim outer
annulus the flux sits on a flat shelf, and it drops to a deeper minimum as the planet crosses onto
the bright inner disk at :math:`z = 0.5`. The step is smoothed over :math:`0.05\,R_\star` on
purpose: the model integrates the profile numerically, and a true discontinuity would leave small
steps in the light curve.

Both profiles are deliberately unphysical. If the model handles them, the realistic cases are
straightforward: a profile tabulated from a stellar atmosphere grid, a law nobody has derived a
transit solution for, or a profile with a feature from starspots. :doc:`../guide/limb_darkening`
describes how to supply your own profile, and how to make it faster by also giving its disk
integral.

The law is nearly free
----------------------

Because the profile enters only through a table of fixed size, the choice of law barely affects
the cost. Timing a 1500-point light curve on one machine, from the one-coefficient linear law to
the six-coefficient general one:

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Law
      - Coefficients
      - Time per evaluation
    * - ``'linear'``
      - 1
      - 0.067 ms
    * - ``'quadratic'``
      - 2
      - 0.067 ms
    * - ``'power-2'``
      - 2
      - 0.070 ms
    * - ``'nonlinear'``
      - 4
      - 0.072 ms
    * - ``'general'``
      - 6
      - 0.072 ms

The differences are a few microseconds, about as large as the variation between timing runs.
Choosing a more flexible law is a modelling decision, not a performance one.

Faster than the analytic model
------------------------------

An evaluation has two parts: a fixed cost, building the table of the mean stellar intensity behind
the planet, and one lookup in that table for every exposure. The analytic quadratic model of
Mandel & Agol, :class:`~pytransit.models.ma_quadratic.QuadraticModel`, has almost no fixed cost,
but evaluates elliptic integrals for every exposure. Once a light curve is long enough to spread
the fixed cost, RoadRunner is faster:

.. plot::
    :context: close-figs

    from _figures.rr_benchmark import plot_benchmark
    plot_benchmark()

With the default settings RoadRunner overtakes the exact quadratic model at about 900 exposures,
and for long light curves it needs 13.6 ns per exposure against 55 ns, four times less. This holds
at every accuracy level from 500 ppm to 1 ppm, since the accuracy changes only the fixed cost.
RoadRunner is also 1.7 times faster than the quadratic model's interpolated mode, which is accurate
to only about 11 ppm.

Most analyses are well past the crossover. A single TESS sector at two-minute cadence has about
20 000 exposures. Long-cadence data are usually supersampled, which multiplies the number of points
evaluated: 300 long-cadence exposures with ten samples each already run 1.5 times faster with
RoadRunner. A joint fit of several transits evaluates all of them in one call. The quadratic model
is faster only for a single short light curve, such as one ground-based transit of a few hundred
exposures, where both models take a few tens of microseconds per evaluation.

The timings are for a single thread on one machine; :doc:`../models/roadrunner` describes the
benchmark and how to run it on your own.

Accuracy
--------

The price of the flexibility is that the profile is integrated numerically rather than
analytically. With the default settings, RoadRunner agrees with the analytic Mandel & Agol
solution for the quadratic law to 1.5 ppm at :math:`k = 0.1`, and the accuracy can be raised further
when the science needs it. :doc:`../models/roadrunner` shows how the error depends on the radius
ratio and on the integration settings.

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

    :doc:`../models/roadrunner` for how the model works, its accuracy, and its API;
    :doc:`../guide/limb_darkening` for the built-in laws and how to supply your own; and
    :doc:`../highlights` for what the shared data setup adds on top.
