RoadRunner model
================

:class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` (Parviainen, MNRAS 499, 1633, 2020)
is PyTransit's recommended general-purpose transit model, and the one to reach for unless a
specialised model fits your problem better. What sets it apart is that the stellar intensity
profile is an *input* rather than something baked into the derivation.

.. plot::
    :context:
    :nofigs:

    from _figures.roadrunner import (plot_laws, plot_profile_and_transit, plot_accuracy,
                                     plot_quadrature_nodes, plot_mean_intensity_table)

Geometry and profile, separated
-------------------------------

Every classical transit model is an analytic solution to one specific integral: a limb-darkened
disk overlapped by an opaque circle, solved separately for the quadratic law, the power-2 law, and
so on. Add a new limb darkening law and you need a new derivation. Some laws have no closed form
at all.

RoadRunner breaks that coupling by splitting the problem in two.

The **geometry** -- which parts of the stellar disk the planet covers -- depends only on the radius
ratio and the projected separation, not on how bright the star is anywhere. RoadRunner turns it
into quadrature nodes and weights: for each planet position, a handful of distances from the disk
centre at which to read the intensity, and how much each reading counts towards the mean
intensity under the planet.

The **intensity profile** enters only through those readings. It is tabulated on a fixed grid once
per evaluation, and the nodes read it from that table, so the model never sees the function that
produced it. The mean intensity under the planet is then tabulated against the planet's position,
and every exposure reads it from there.

Nothing in this pipeline depends on the form of the profile. That single fact produces the three
properties the rest of this page demonstrates: any radially symmetric profile works, the choice
costs essentially nothing, and accuracy is a quadrature setting rather than a property of the law.

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

A sharp step in brightness is no harder:

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

The light curve reads straight off the profile. While the planet is entirely over the dim outer
annulus the flux sits on a flat shelf, then drops to a deeper minimum as it crosses onto the bright
inner disk at :math:`z = 0.5`.

The step is smooth on a scale of :math:`0.05\,R_\star`, and that is deliberate. The model
integrates the profile by quadrature, which assumes the intensity varies smoothly between its
nodes. A true discontinuity breaks that assumption: it leaves a staircase in the light curve that
a larger ``nq`` smooths out, but no setting makes it exact.

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

Because the profile enters only through a fixed-size table, and the quadrature and the light curve
never see anything else, the choice of law barely registers. Timing a 1500-point light curve on one machine, from a
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

How the mean intensity is computed
----------------------------------

For each planet position the model needs one number: the mean stellar intensity behind the planet.
The intensity depends only on the distance :math:`z` from the disk centre, so the region behind the
planet can be cut into thin rings centred on the star. Every point on a ring is equally bright, and
only an arc of each ring lies behind the planet. The mean intensity is the intensity of each ring
weighted by the length of its arc, summed over the rings the planet covers.

That sum is an integral over :math:`z`, and the difficulty is at the planet's edges. Where a ring
only just touches the planet its arc is vanishingly short, and moving inwards the arc grows like the
square root of the distance to the edge: steeply at first, then levelling off. Evenly spaced
samples handle an edge like this poorly, and would need many rings crowded against it.

The model deals with the two edges in two ways.

- **One edge is stretched away.** Writing the distance from that edge as a square,
  :math:`z = z_\mathrm{edge} - s^2`, turns the steep square-root rise into a smooth curve in
  :math:`s`.
- **The other edge is built into the integration rule.** Gauss quadrature estimates an integral
  from a weighted sum of a few samples, with the sample positions and weights chosen so that the
  sum is exact for any polynomial up to a high degree. A Gauss-Jacobi rule does the same for a
  polynomial *multiplied by* a square root that falls to zero at one end. Because the rule already
  accounts for the square root, its samples only have to describe the smooth remainder: the
  intensity and the rest of the geometry. On an integrand of this shape, eight Gauss-Jacobi nodes
  reach machine precision, while eight Gauss-Legendre nodes, the standard rule for smooth
  integrands, are still wrong in the third decimal place.

The result is a short list: a few distances from the disk centre at which to read the intensity,
and a weight for each reading. The mean intensity is the weighted average of those readings. Each
node is a ring, so the figure draws the arc of each ring that lies behind the planet, with the line
width proportional to the node's weight, for the default eight nodes per segment.

.. plot::
    :context: close-figs

    plot_quadrature_nodes()

Where the planet is decides how the integral is set up:

- **On the disk**, one Gauss-Jacobi segment covers the planet. The stretched edge is the outer
  one, and the nodes crowd towards it.
- **Over the disk centre**, the rings inside the planet are complete circles with no edge to
  handle, so they take a plain Gauss-Legendre segment. The partial rings further out take a
  Gauss-Jacobi segment.
- **Across the limb**, the integral runs in :math:`\mu` from the limb inwards, which also removes
  the square root the intensity profile itself has at the limb. The planet's inner edge goes to
  Gauss-Jacobi.

Accuracy and the discretisation
-------------------------------

Splitting the problem buys flexibility at the cost of integrating the profile numerically, and
that integration is where the accuracy is decided.

The mean intensity is tabulated against the grazing parameter :math:`g = b/(1+k)` in two segments
split at the limb contact, where the exact function has a kink, and read with a cubic during the
evaluation. The lower panel shows what the split buys, as flux error from the table interpolation
alone.

.. plot::
    :context: close-figs

    plot_mean_intensity_table()

The split halves the worst error around the limb contact, and cuts it threefold for
:math:`k = 0.3`. The weaker kink where the planet's edge crosses the disk centre is left unsplit;
at the default table size it costs well under a ppm (0.1 ppm here, 0.4 ppm at :math:`k = 0.3`).

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
