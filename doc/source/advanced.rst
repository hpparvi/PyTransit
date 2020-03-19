Advanced topics
===============

Supersampling
-------------

The transit models offer built-in *supersampling* for accurate modelling of long-cadence observations. The number of
samples and the exposure time can be given when setting up the model

    tm.set_data(times, nsamples=10, exptimes=0.02)

Heterogeneous time series
-------------------------

PyTransit allows for heterogeneous time series, that is, a single time series can contain several individual light curves
(with, e.g., different time cadences and required supersampling rates) observed (possibly) in different passbands.

If a time series contains several light curves, it also needs the light curve indices for each exposure. These are given
through `lcids` argument, which should be an array of integers. If the time series contains light curves observed in
different passbands, the passband indices need to be given through `pbids` argument as an integer array, one per light
curve. Supersampling can also be defined on per-light curve basis by giving the `nsamples`and `exptimes` as arrays with
one value per light curve.

For example, a set of three light curves, two observed in one passband and the third in another passband

.. code-block:: python

    times_1 (lc = 0, pb = 0, sc) = [1, 2, 3, 4]
    times_2 (lc = 1, pb = 0, lc) = [3, 4]
    times_3 (lc = 2, pb = 1, sc) = [1, 5, 6]

Would be set up as

.. code-block:: python

    tm.set_data(time  = [1, 2, 3, 4, 3, 4, 1, 5, 6],
                lcids = [0, 0, 0, 0, 1, 1, 2, 2, 2],
                pbids = [0, 0, 1],
                nsamples = [  1,  10,   1],
                exptimes = [0.1, 1.0, 0.1])

Further, each passband requires two limb darkening coefficients, so the limb darkening coefficient array for a single parameter set should now be

    ldc = [u1, v1, u2, v2]

where u and v are the passband-specific quadratic limb darkening model coefficients.