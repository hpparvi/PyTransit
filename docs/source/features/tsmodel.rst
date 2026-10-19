Transmission spectroscopy model
===============================

A spectroscopic transit observation gives hundreds or thousands of light curves that differ in
only two things. Every wavelength bin sees the same exposures, the same orbit and the same planet;
only the radius ratio and the limb darkening change with wavelength.

:class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel`, exported as
``TSModel``, is the :doc:`RoadRunner model <roadrunner>` built around this. It computes the
geometry once and evaluates every wavelength bin against it. For a spectrum of a thousand bins,
this is about twenty times faster than evaluating RoadRunner bin by bin.

.. plot::
    :context:
    :nofigs:

    from _figures.tsmodel import toy_spectrum, plot_spectroscopic_transit

One call, a whole spectrum
--------------------------

One radius ratio and one set of limb darkening coefficients per bin go in, and one light curve per
bin comes out. Here for a toy near-infrared spectrum of 200 bins between 0.6 and 5 micron:

.. plot::
    :context: close-figs
    :include-source:

    from numpy import pi, linspace
    from pytransit import TSModel

    window = 3.5 / 24
    time = linspace(-0.5 * window, 0.5 * window, 1500)
    t0, p, a, i = 0.0, 4.0, 13.0, 0.49 * pi

    wl, k, ldc = toy_spectrum(200)      # k.shape == (200,), ldc.shape == (200, 2)

    tm = TSModel('quadratic')
    tm.set_data(time)
    flux = tm.evaluate(k, ldc, t0, p, a, i)
    # flux.shape == (1, 200, 1500)

    plot_spectroscopic_transit(wl, k, flux[0], time)

The left panel is the input spectrum, and the right panel shows how each bin's light curve differs
from the mean over all bins. The absorption bands show up as horizontal stripes where the transit
is deeper. The largest differences, however, come from the limb darkening, which is stronger in the
blue: those bins are deeper at mid-transit and shallower near the limb. A transmission spectrum is
as much a question of the limb darkening in each bin as of the radius ratios, and the model takes
both per bin.

``set_data`` gets only the times. There are no light curve or passband indices to set up, and the
number of bins is read from the arrays given to ``evaluate``, so the same model evaluates a white
light curve, a coarse spectrum or a fine one without being set up again.

Cost per wavelength bin
-----------------------

Time per evaluation for a 1000-exposure light curve with the transit covering about half of it, on
one thread. ``RoadRunnerModel`` is timed both called once per bin and given all the bins at once as
the passbands of a single dataset:

.. list-table::
    :header-rows: 1
    :widths: 16 24 30 30

    * - Bins
      - ``TSModel``
      - ``RoadRunnerModel``, one call per bin
      - ``RoadRunnerModel``, all bins in one call
    * - 1
      - 0.064 ms
      - 0.058 ms
      - 0.054 ms
    * - 10
      - 0.090 ms
      - 0.59 ms
      - 0.50 ms
    * - 100
      - 0.33 ms
      - 5.7 ms
      - 4.8 ms
    * - 1000
      - 2.6 ms
      - 57 ms
      - 54 ms

With one bin there is nothing to share. Beyond that the difference grows with the number of bins:
about six times at ten bins, fifteen at a hundred, and twenty at a thousand. Each bin adds about
2.6 µs to an evaluation, where a separate RoadRunner evaluation costs 55 µs. Large spectra can also
be divided between threads with ``nthreads``: on eight threads, the thousand bins take 0.92 ms.

Accuracy
--------

The geometry is shared through the mean radius ratio of the bins, so the result is exact when all
the bins have the same radius ratio and approximate when they differ. For the toy spectrum above,
whose radius ratios stay within 2% of their mean, the model agrees with RoadRunner evaluated bin
by bin to within 6 ppm. :doc:`../models/tsmodel` shows how the error scales with the spread of the
spectrum and the size of the planet.

.. seealso::

    :doc:`../models/tsmodel` for how the model works, its accuracy, and its API;
    :doc:`roadrunner` for the model it is built on.
