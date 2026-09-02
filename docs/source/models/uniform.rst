Uniform model
=============

:class:`~pytransit.models.ma_uniform.UniformModel` reproduces a transit over a disk of constant
surface brightness. With no limb darkening, the overlap integral reduces to the analytic
intersection area of two circles, which makes this the fastest model in PyTransit.

It takes no limb darkening coefficients, so its `evaluate` signature is shorter than the other
models':

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import UniformModel

    time = linspace(-0.1, 0.1, 1000)

    tm = UniformModel()
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi)

When to use it
--------------

**Secondary eclipses.** The occulted body is the planet, whose dayside really is close to uniform,
so the model is physically correct here rather than merely convenient. Create it with
``eclipse=True`` to shift the modelled event to the secondary eclipse.

.. code-block:: python

    tm = UniformModel(eclipse=True)

:doc:`EclipseModel <eclipse>` does the same thing while also handling the planet-star flux ratio,
and is usually the better choice.

**Speed-critical work where transit shape does not matter.** Transit searches, injection-recovery
tests, and rough duration or depth estimates.

.. warning::

    Do not use this model to measure a radius ratio from real photometry. Ignoring limb darkening
    biases the depth, and therefore the inferred planet radius, at a level that matters for any
    modern dataset.

API
---

.. autoclass:: pytransit.models.ma_uniform.UniformModel
    :members: evaluate
    :special-members: __init__
