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
thousand exposures drops from 2.6 ms to 0.92 ms, while a hundred bins go only from 0.32 ms to
0.28 ms.

The threads divide the bins *within* each parameter vector, and the parameter vectors are
evaluated one after another. Threads therefore speed up large spectra, not large populations of
small ones.

How it works
------------

The model splits the work into the part that depends on the bin and the part that does not.

For each parameter vector, it computes **once**:

- the solution of the orbit and the time window where the planet can overlap the star;
- the mean radius ratio over the bins, :math:`\bar{k}`, and the quadrature nodes for a planet of
  that size;
- the map from a limb darkening profile to a table of the mean stellar intensity under the planet
  against the planet's position;

and for each exposure, **once**:

- the projected star-planet distance, and where it falls in the mean intensity table;
- the area of the stellar disk covered by a planet of radius :math:`\bar{k}`, and how fast that
  area changes with the radius.

The map is what makes the limb darkening cheap. The mean intensity table is linear in the profile,
and the quadrature nodes, weights, and interpolation stencils that produce it depend only on the
planet, so they fold into a single linear map. Each table entry reads the profile only across the
planet, which keeps the map sparse.

What remains **per bin** is two short steps. The bin applies the shared map to its own limb
darkening profile to get its mean intensity table. Then, for each exposure, it evaluates one cubic
from that table at the shared position, and scales the shared overlap area to its own radius ratio:
exactly by :math:`(k/\bar{k})^2` while the planet is fully on the disk, and to first order in
:math:`k - \bar{k}` while it crosses the limb. The two steps take roughly equal time, about
2.6 µs per bin together for a thousand exposures.

Two quantities are therefore taken from the mean planet rather than from each bin's own: the mean
intensity under the planet, and the overlap area across the limb. When every bin has the same
radius ratio both are exact, and the model reproduces
:class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` to rounding error. How far they drift
when the radius ratios differ is the subject of the next section.

Accuracy
--------

The error of sharing the geometry grows with how far a bin's radius ratio is from the mean. The
reference below is ``RoadRunnerModel`` evaluated bin by bin with the same discretisation, so the
difference is the error of sharing the geometry alone, not the quadrature error both models have in
common (see :doc:`roadrunner` for that).

.. plot::

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
