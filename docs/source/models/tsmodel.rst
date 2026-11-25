Transmission spectroscopy model
===============================

:class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel` -- also exported as
``TSModel`` -- is the RoadRunner model specialised for transmission spectroscopy.

The problem it solves
---------------------

A transmission spectroscopy dataset is a set of spectroscopic light curves extracted from a single
spectroscopic time series. Every wavelength bin is observed at the *same* times, through the *same*
transit, of the *same* planet. What differs between bins is the radius ratio -- the transmission
spectrum we are after -- and the limb darkening.

Evaluating a general model bin by bin would recompute the identical transit geometry once per bin.
With hundreds or thousands of bins, as JWST routinely produces, that is the dominant cost and it is
entirely wasted.

This model computes the geometry once and reuses it across all the wavelength bins, leaving each
bin only a few microseconds of work of its own. :doc:`../features/tsmodel` compares its cost with
evaluating :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` bin by bin.

Differences from the other models
---------------------------------

.. important::

    **The number of passbands comes from ``ldc``, not from ``set_data``.** The model reads ``npb``
    from the shape of the limb darkening array on every call. A dataset can therefore be re-binned
    in wavelength without touching the data setup.

    **``evaluate`` always returns a 3D array** of shape ``(npv, npb, npt)``, even for a single
    parameter vector, where it is ``(1, npb, npt)``.

Because all the bins share one time array, `set_data` takes the times only. The `lcids` and `pbids`
arrays play no role here -- there is one light curve per wavelength bin by construction.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace, full, tile
    from pytransit import TSModel

    time = linspace(-0.1, 0.1, 1000)
    npb = 100                                    # wavelength bins

    tm = TSModel()
    tm.set_data(time)

    k = full(npb, 0.1)                           # one radius ratio per bin
    ldc = tile([0.2, 0.1], (npb, 1))             # one coefficient pair per bin

    flux = tm.evaluate(k=k, ldc=ldc, t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    # flux.shape == (1, 100, 1000)

The number of radius ratios must match the number of limb darkening rows; a mismatch raises a
`ValueError`.

Since the model inherits from :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel`, it
takes the same initialiser arguments, including the choice of limb darkening law and the
discretisation parameters.

Re-binning
^^^^^^^^^^

As the number of bins is read from the limb darkening array on every call, one model evaluates a
spectrum at any resolution, including a single white light bin:

.. code-block:: python

    flux_white = tm.evaluate(k=0.1, ldc=[[0.2, 0.1]], t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    # flux_white.shape == (1, 1, 1000)

    flux_coarse = tm.evaluate(k=full(20, 0.1), ldc=tile([0.2, 0.1], (20, 1)),
                              t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    # flux_coarse.shape == (1, 20, 1000)

This makes it straightforward to explore the binning of a spectrum, or to fit the white light
curve first and the spectroscopic light curves after, without rebuilding the model or the data.

Populations of parameter vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a population of parameter vectors, as used by population-based optimisers and ensemble
samplers, the radius ratios are given as a 2D array with one row per parameter vector, the
orbital parameters as 1D arrays, and the limb darkening coefficients as an explicit 3D array of
shape ``(npv, npb, nldc)``; anything else raises a `ValueError`.

.. code-block:: python

    from numpy import zeros, ones

    npv = 50
    flux = tm.evaluate(k=full((npv, npb), 0.1),
                       ldc=tile([0.2, 0.1], (npv, npb, 1)),
                       t0=zeros(npv), p=ones(npv), a=full(npv, 3.0), i=full(npv, 0.5*pi))
    # flux.shape == (50, 100, 1000)

Each parameter vector gets its own mean radius ratio and its own shared geometry.

Threads
^^^^^^^

With ``nthreads`` greater than one, the bins are divided between the threads. Each bin's table and
light curve are independent of the others', so the threads never wait for each other, but the gain
depends on how much work there is to divide: on eight threads, a thousand-bin spectrum of a
thousand exposures drops from 2.0 ms to 0.76 ms, while a hundred bins stay at 0.24 ms.

The threads divide the bins *within* each parameter vector, and the parameter vectors are
evaluated one after another. Threads therefore speed up large spectra, not large populations of
small ones.

How it works
------------

The model is the :doc:`RoadRunner model <roadrunner>` with its work divided into the part that
depends on the wavelength bin and the part that does not. As in RoadRunner, the flux of a bin at a
given time is

.. math::

    F_\lambda = 1 - \frac{\bar{I}_\lambda A_\lambda}{F_{\star,\lambda}},

where :math:`A_\lambda` is the area of the stellar disk the planet covers,
:math:`\bar{I}_\lambda` the mean stellar intensity over that area, and :math:`F_{\star,\lambda}`
the flux of the whole disk. The bins see the same orbit, so the planet is at the same position in
every bin. Only its size and the star's limb darkening change from bin to bin, and for a
transmission spectrum the size changes by a percent or two.

The model therefore computes the geometry for one planet, with the mean radius ratio of the bins,
:math:`\bar{k}`, and gives each bin only the two things that are its own: a mean intensity table
for its limb darkening, and a correction of the covered area to its radius ratio.

.. plot::
    :context: close-figs

    from _figures.tsmodel import plot_shared_work
    plot_shared_work()

One map for all the limb darkening profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RoadRunner tabulates the mean intensity under the planet against the planet's position, and builds
the table by reading the star's intensity profile at a set of quadrature nodes (see
:doc:`roadrunner`). Where the nodes are, and how much each one weighs, depends only on the size of
the planet. The profile enters only through the values read at the nodes, so the table is a linear
function of the profile:

.. math::

    \mathbf{T}_\lambda = \mathsf{W}\,\mathbf{p}_\lambda,

where :math:`\mathbf{p}_\lambda` is the bin's intensity profile on the model's fixed grid of 200
points in :math:`\mu` and :math:`\mathbf{T}_\lambda` its mean intensity table. The matrix
:math:`\mathsf{W}` folds together the quadrature nodes, their weights, and the interpolation of
the profile between its grid points. It is built once per parameter vector, for a planet of radius
:math:`\bar{k}`.

The left panel shows which profile points each table node reads. A table node is a planet
position, and the planet covers a narrow range of distances from the disk centre, so each row of
:math:`\mathsf{W}` is mostly zeros: on average a row reads 28 of the 200 profile points, and the
model sums only over the band that contains them. A bin's table is one short sum per row, and the
middle panel shows the result for seven bins of a toy spectrum whose limb darkening weakens towards
the red.

One geometry for all the bins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For every exposure, or every subsample of a supersampled one, the model computes once

- the planet's distance from the disk centre, and from it the interval of the mean intensity table
  the planet falls in and the position within the interval;
- the area a planet of radius :math:`\bar{k}` covers, and how fast that area changes with the
  radius.

Samples where the planet does not touch the star are dropped at this stage, so the bins only loop
over the in-transit samples.

Each bin then scales the area to its own radius ratio.

- **While the planet is fully on the disk**, the covered area is the planet's own area, and the
  scaling is exact: :math:`A_\lambda = \bar{A}\,(k_\lambda / \bar{k})^2`. This is used whenever
  the largest planet of the spectrum is inside the limb.
- **While the planet crosses the limb**, the area is corrected to first order. A slightly larger
  planet gains a thin crescent along the part of its edge that lies inside the star (right panel,
  orange along the blue arc). If that arc spans the angle :math:`2\kappa`, its length is
  :math:`2\kappa\bar{k}`, and

  .. math::

      A_\lambda \approx \bar{A} + 2\kappa\bar{k}\,(k_\lambda - \bar{k}).

  The angle comes for free from the calculation of :math:`\bar{A}`.

What a bin does
^^^^^^^^^^^^^^^

A bin's own work is two short steps. It applies the shared map to its limb darkening profile to
get its mean intensity table, and turns the table into cubic coefficients. Then, for every
in-transit sample, it evaluates one cubic at the shared position and multiplies it by the scaled
area, which is about a dozen multiplications and additions with no square roots or trigonometric
functions.

Together the two steps take about 1.9 µs per bin for a thousand exposures, roughly a third of it
for the table and two thirds for the light curve. The cost of the shared geometry does not depend
on the number of bins.

What is approximated
^^^^^^^^^^^^^^^^^^^^

Two quantities are taken from the mean planet instead of from each bin's own: the mean intensity
under the planet, and the covered area while the planet crosses the limb. When every bin has the
same radius ratio both are exact, and the model reproduces
:class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` to rounding error. How far they drift
when the radius ratios differ is the subject of the next section.

The bins also share one exposure time and one supersampling rate, those of the first light curve
given to `set_data`.

Accuracy
--------

The error of sharing the geometry grows with how far a bin's radius ratio is from the mean. The
reference below is ``RoadRunnerModel`` evaluated bin by bin with the same discretisation, so the
difference is the error of sharing the geometry alone, not the quadrature error both models have in
common (see :doc:`roadrunner` for that).

.. plot::
    :context: close-figs

    from _figures.tsmodel import plot_accuracy
    plot_accuracy()

The left panel shows where the error lives for two bins one percent either side of the mean. In
the middle of the transit it is a fraction of a ppm. It concentrates at ingress and egress, and
peaks around second and third contact, where the planet's edge meets the limb: there the limb
darkening is steepest and the mean intensity under the planet is most sensitive to its size. The
two bins' errors have opposite signs, because they deviate from the mean in opposite directions.

The right panel shows how the worst-case error scales. For deviations up to a few percent it is
close to linear: about 1 ppm per percent of deviation for :math:`\bar{k} = 0.05`, 6 ppm per percent
for :math:`\bar{k} = 0.1`, and 40 ppm per percent for :math:`\bar{k} = 0.2`. It also grows steeply
with the size of the planet.

What matters is the spread of the radius ratios *within one spectrum*, not their absolute value.
The toy spectrum of :doc:`../features/tsmodel`, whose radius ratios stay within 2% of their mean,
is reproduced to within 6 ppm in every bin and every exposure, while the light curves of its bins
differ from each other by over 1000 ppm. The atmospheric signal of most planets changes the radius
ratio by a percent or two, and there the error is typically well below the precision of a fitted
spectroscopic transit depth.

.. tip::

    For a large planet with strong spectral features, check a fit by evaluating the best-fitting
    parameters with ``RoadRunnerModel`` bin by bin. If the difference is small compared with the
    residuals, the shared geometry has cost nothing.

API
---

.. autoclass:: pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel
    :members: evaluate
