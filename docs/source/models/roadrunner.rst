RoadRunner model
================

The RoadRunner model (Parviainen, MNRAS 499, 1633, 2020) is PyTransit's recommended
general-purpose transit model, and the one to use unless a specialised model fits your problem
better.

What makes it different
-----------------------

The classical transit models are analytic solutions to the overlap integral of a limb-darkened
disk and an opaque circle, derived separately for each limb darkening law. A new law needs a new
derivation, and some laws have no closed-form solution at all.

RoadRunner separates the problem into two parts. The *geometry* -- which parts of the stellar disk
the planet covers -- becomes a set of Gauss quadrature nodes and weights matched to the planet's
position. The *stellar intensity profile* enters only as a table on a fixed grid, which the nodes
read. Together they give the mean intensity under the planet, tabulated against the planet's
position once per evaluation and read for every exposure.

Two things follow:

- **Any radially symmetric intensity profile works.** A built-in law, a Python callable, or a
  numerically tabulated profile from a stellar atmosphere model are all equally valid inputs.
- **The cost barely depends on the law.** A four-coefficient non-linear profile costs about the
  same as a linear one, because the profile is evaluated only on the fixed table grid.

The price is that the profile is integrated numerically. `How it works`_ describes the
integration, and `Accuracy`_ how its error depends on the radius ratio and on the two settings that
control it. :doc:`../features/roadrunner` shows the flexibility in practice.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import RoadRunnerModel

    time = linspace(-0.1, 0.1, 1000)

    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

Choosing a different limb darkening law is a change to one string:

.. code-block:: python

    tm = RoadRunnerModel('power-2')
    tm.set_data(time)
    flux = tm.evaluate(k=0.1, ldc=[0.6, 0.5], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

See :doc:`../guide/limb_darkening` for the available laws and for supplying your own.

Performance options
-------------------

**Threading.** ``nthreads`` above one enables the parallel model.

.. code-block:: python

    tm = RoadRunnerModel('quadratic', nthreads=4)

Numba's thread count is process-global, so the most recently created model sets it for every model
in the process, and it is capped at the ``NUMBA_NUM_THREADS`` value fixed when Numba was imported.

**Small planet approximation.** For a single light curve with a radius ratio at or below
``small_planet_limit`` (0.01 by default), the model approximates the mean blocked intensity by the
intensity at the planet's centre. The error grows roughly quadratically with the radius ratio:
below 1 ppm at the default limit, about 100 ppm at :math:`k = 0.05`. Raise the limit only if speed
matters more than ppm accuracy, and pass ``None`` or ``0.0`` to disable it.

How it works
------------

For each planet position the model needs one number: the mean stellar intensity behind the planet.
It computes it in two stages. Quadrature nodes and weights describe the geometry, and a table of
the mean intensity against the planet's position lets every exposure read the result instead of
integrating it again.

The intensity profile itself is tabulated once per evaluation on a fixed grid in :math:`\mu`, and
the quadrature nodes read it from there. This is why a tabulated stellar atmosphere profile costs
the same as an analytic law.

The mean intensity under the planet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    from _figures.roadrunner import plot_quadrature_nodes, plot_mean_intensity_table, plot_accuracy
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

Because the profile is read only at the nodes, the integration assumes that it varies smoothly
between them. A profile with a true discontinuity leaves small steps in the light curve, which a
larger ``nq`` reduces but no setting removes.

The mean intensity table
^^^^^^^^^^^^^^^^^^^^^^^^

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

Accuracy
--------

Two settings control the accuracy: `nq`, the number of quadrature nodes per segment, and `ng`, the
size of the mean intensity table. Both can be given to the initialiser, or changed later with
:meth:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel.init_integration`.

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

For most work the defaults are appropriate: a few ppm is far below the noise of a real transit
observation. Raise the resolution when the *model* error would compete with the signal you are
after, as it does for planetary oblateness or for very large planets.

The parameters `nz`, `nzin`, `nzlimb`, `zcut`, `precompute_weights`, `klims` and `nk` of earlier
versions are accepted and ignored with a `FutureWarning`: the disk is no longer discretised into
annuli, and there is no weight table to precompute.

Speed against the quadratic model
---------------------------------

An evaluation has two parts. Building the mean intensity table is a fixed cost that grows with `nq`
and `ng`, and each exposure then costs one table lookup, whatever the accuracy setting. The analytic
quadratic model of Mandel & Agol has almost no fixed cost, but evaluates elliptic integrals for
every exposure. RoadRunner is therefore slower for short light curves and faster for long ones.

The benchmark compares four RoadRunner accuracy levels, and the defaults, with
:class:`~pytransit.models.ma_quadratic.QuadraticModel`, both exact and with its interpolated mode.
Each level is the cheapest ``nq, ng`` whose worst-case error against the exact quadratic model stays
below the target for radius ratios up to 0.2 and impact parameters up to 0.9:

.. list-table::
    :header-rows: 1
    :widths: 22 14 14 25 25

    * - Model
      - ``nq``
      - ``ng``
      - Worst-case error
      - Faster than the exact quadratic model above
    * - RoadRunner, 1 ppm
      - 5
      - 300
      - 0.9 ppm
      - 1350 exposures
    * - RoadRunner, default
      - 8
      - 100
      - 5.9 ppm
      - 910 exposures
    * - RoadRunner, 10 ppm
      - 4
      - 75
      - 7.2 ppm
      - 690 exposures
    * - RoadRunner, 100 ppm
      - 3
      - 20
      - 48 ppm
      - 560 exposures
    * - RoadRunner, 500 ppm
      - 2
      - 10
      - 316 ppm
      - 530 exposures
    * - Quadratic, interpolated
      -
      -
      - 11 ppm
      -

Both models run on a single thread, for a light curve where about half of the exposures are in
transit:

.. plot::
    :context: close-figs

    from _figures.rr_benchmark import plot_benchmark
    plot_benchmark()

For long light curves every RoadRunner level costs the same, 13.6 ns per exposure, against 55 ns
for the exact quadratic model and 24 ns for the interpolated one. RoadRunner is then four times
faster than the exact model and 1.7 times faster than the interpolated one, even at an accuracy of
1 ppm, where the interpolated model is accurate to 11 ppm. The accuracy setting only decides how
many exposures it takes to reach that point: RoadRunner overtakes the interpolated model above
2300 exposures at 500 ppm, and above 5900 exposures at 1 ppm.

The accuracy is therefore almost free for the light curves where speed matters most. A fit to a
short light curve is fast with either model. For a long one, such as a TESS or Kepler light curve
or a spectroscopic time series, the 1 ppm setting costs 19% more than the 500 ppm one at 10 000
exposures, and 2% more at 100 000.

.. note::

    The timings were measured on an AMD Ryzen 7 5800X with Numba 0.62, and depend on the machine.
    To measure them on your own, run ``python docs/source/_figures/rr_benchmark.py`` in a source
    checkout, which rewrites the data this figure is drawn from.

Examples
--------

.. toctree::
    :maxdepth: 1

    ../notebooks/models/roadrunner/roadrunner_model_example_1
    ../notebooks/models/roadrunner/roadrunner_model_example_2

API
---

.. autoclass:: pytransit.models.roadrunner.rrmodel.RoadRunnerModel
    :members: set_data, evaluate, init_integration
    :special-members: __init__
