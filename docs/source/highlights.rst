Highlights
==========

PyTransit is built around the idea that a real dataset is messy: transits observed on different
nights, by different instruments, in different passbands, at different cadences, with mid-transit
times that drift. Rather than making you stitch together one model call per light curve, PyTransit
lets you describe that structure once and evaluate the whole thing in a single call.

This page shows what that buys you. Every figure below is drawn by the code shown above it.

.. plot::
    :context:
    :nofigs:

    from _figures.highlights import plot_transit, plot_light_curves

A transit model in three lines
------------------------------

Create a model, give it the observation times, evaluate it. That is the whole interface.

.. plot::
    :context: close-figs
    :include-source:

    from numpy import pi, linspace
    from pytransit import QuadraticModel

    window = 4 / 24                                            # a four-hour window, in days
    times_sc = linspace(-0.5 * window, 0.5 * window, 240)      # one-minute cadence

    k, t0, p, a, i = 0.1, 0.0, 4.0, 13.0, 0.49 * pi
    ldc = [0.3, 0.1]

    tm = QuadraticModel()
    tm.set_data(times_sc)
    flux_sc = tm.evaluate(k, ldc, t0, p, a, i)

    plot_transit(times_sc, flux_sc, 'One-minute cadence')

Long exposures, handled for you
-------------------------------

A 30-minute exposure smears the transit: what you measure is the average of the flux over the
exposure, not its value at the mid-exposure time. Ignoring that rounds off the ingress and egress
and biases everything you infer from them.

Tell `set_data` the exposure time and how many samples to integrate over, and the model does the
averaging itself. Nothing else about the call changes.

.. plot::
    :context: close-figs
    :include-source:

    times_lc = linspace(-0.5 * window, 0.5 * window, 9)        # 30-minute cadence

    tm.set_data(times_lc, exptimes=0.02, nsamples=10)
    flux_lc = tm.evaluate(k, ldc, t0, p, a, i)

    plot_transit(times_lc, flux_lc, 'Thirty-minute cadence, supersampled',
                 step=True, reference=(times_sc, flux_sc))

The grey curve behind is the unbinned model from the previous section. The supersampled model
follows its average across each exposure rather than cutting the corners.

Many light curves, many passbands
---------------------------------

Now the part that makes heterogeneous datasets easy. Concatenate every observation into one time
array and describe the structure with two integer arrays:

``lcids``
    which light curve each *exposure* belongs to,

``pbids``
    which passband each *light curve* belongs to.

Here three light curves are stacked: one at short cadence and two at long cadence, with the first
and last sharing a passband. The exposure times and supersampling rates are given per light curve,
so the cadences can be mixed freely in a single model.

.. plot::
    :context: close-figs
    :include-source:

    from numpy import array, concatenate, full

    times = concatenate([times_sc, times_lc, times_lc])
    lcids = concatenate([full(times_sc.size, 0),
                         full(times_lc.size, 1),
                         full(times_lc.size, 2)])

    tm = QuadraticModel()
    tm.set_data(times, lcids=lcids, pbids=[0, 1, 0],
                exptimes=[0.0, 0.02, 0.02], nsamples=[1, 10, 10])

    flux = tm.evaluate(k, array([0.3, 0.1, 0.0, 0.1]), t0, p, a, i)

    plot_light_curves(tm, times, flux)

One `evaluate` call produced all three. The middle panel is drawn as a step function because the
model is supersampling it; the panels are drawn from the model's own `lcids`, `pbids` and
`nsamples`, so they show exactly what `set_data` recorded.

A radius ratio per passband
---------------------------

Because the passbands are already described, making the radius ratio wavelength-dependent costs
one extra number. Pass one `k` per passband instead of a scalar, and the light curves in each
passband get their own transit depth while sharing a single orbit.

.. plot::
    :context: close-figs
    :include-source:

    flux = tm.evaluate([k, 0.8 * k], array([0.3, 0.1, 0.0, 0.1]), t0, p, a, i)

    plot_light_curves(tm, times, flux)

The middle light curve is now shallower than the other two, which still share passband 0. This is
the whole of transmission spectroscopy in one argument.

Transit timing variations
-------------------------

A third index array, ``epids``, maps each light curve to an *epoch*. Give it one, and each light
curve can have its own mid-transit time while everything else stays shared, which is exactly what
fitting transit timing variations needs.

.. plot::
    :context: close-figs
    :include-source:

    tm = QuadraticModel()
    tm.set_data(times, lcids=lcids, pbids=[0, 1, 0], epids=[0, 1, 2],
                exptimes=[0.0, 0.02, 0.02], nsamples=[1, 10, 10])

    flux = tm.evaluate([k, 0.8 * k], array([0.3, 0.1, 0.0, 0.1]),
                       [-0.02, -0.01, 0.02], p, a, i)

    plot_light_curves(tm, times, flux, show_epochs=True)

The dashed line marks the unperturbed zero epoch. The first transit arrives early, the last one
late, and no part of the model call had to change beyond passing three transit centres instead of
one.

A whole parameter population at once
-------------------------------------

Sampling and optimisation need the model evaluated for many parameter vectors. Give `evaluate`
arrays instead of scalars and it computes the entire population in one call, in parallel, rather
than looping in Python.

.. plot::
    :context: close-figs
    :include-source:

    from numpy.random import seed, normal, uniform

    seed(0)
    npv = 15                                     # fifteen parameter vectors

    flux = tm.evaluate(normal([k, 0.8 * k], 0.002, (npv, 2)),   # k, per passband
                       uniform(0.10, 0.50, (npv, 4)),           # ldc, per passband
                       normal(0.0, 0.004, (npv, 3)),            # t0, per epoch
                       normal(4.0, 0.010, npv),                 # period
                       normal(13.0, 0.10, npv),                 # scaled semi-major axis
                       uniform(0.48 * pi, 0.5 * pi, npv))       # inclination

    plot_light_curves(tm, times, flux, show_epochs=True)

Fifteen parameter vectors, three light curves, two passbands, three epochs, two cadences: one
call, one array of fluxes with shape ``(15, 258)``.

Where to go next
----------------

- :doc:`guide/data_setup` explains the index arrays in detail, with diagrams of how they line up.
- :doc:`guide/evaluation` covers the broadcasting rules the last example relies on.
- :doc:`models/index` lists the transit models. A gallery of them will appear on this page in a
  future release.
