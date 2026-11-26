Quadratic model
===============

:class:`~pytransit.models.ma_quadratic.QuadraticModel` implements the analytic transit model of
Mandel & Agol (ApJ 580, L171, 2002) for a star with quadratic limb darkening,

.. math::

    I(\mu) = 1 - u(1-\mu) - v(1-\mu)^2.

This is the classical transit model and the most widely used one in the field. Its value today is
mostly that: it is thoroughly tested, and results computed with it are directly comparable to two
decades of literature.

For new work, :doc:`RoadRunner <roadrunner>` with the ``'quadratic'`` law computes the same model
while leaving the door open to any other limb darkening law.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import QuadraticModel

    time = linspace(-0.1, 0.1, 1000)

    tm = QuadraticModel()
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

Interpolation
-------------

The analytic solution involves elliptic integrals, which are expensive. ``interpolate=True``
switches to the interpolation scheme of Parviainen (MNRAS 450, 3233, 2015): the model is
precomputed on a grid of radius ratios and projected distances and interpolated at evaluation time.

.. code-block:: python

    tm = QuadraticModel(interpolate=True, klims=(0.05, 0.15), nk=512, nz=512)

.. warning::

    The interpolated model is only defined inside `klims`. Evaluating it with a radius ratio
    outside the range the table was built for gives a wrong answer rather than an error, so set
    `klims` from the priors of your fit, with margin.

`nk` and `nz` set the radius ratio and distance grid sizes. Larger grids are more accurate and cost
more memory and initialisation time.

Threading
---------

``nthreads`` above one evaluates a single light curve in parallel.

.. code-block:: python

    tm = QuadraticModel(nthreads=4)

The light curve is dealt out to the threads in small interleaved blocks, so that each thread gets
an even share of the in-transit points. A light curve of 100 000 points is about seven times
faster on eight threads than on one. Threads do not help a short light curve, for which starting
them costs more than the model itself; the model notices this and evaluates such a light curve
serially, so setting ``nthreads`` never slows it down by more than a fraction of a microsecond.

A parameter population (1D or 2D parameter arrays) is always evaluated in parallel over the
parameter sets, whatever the value of ``nthreads``.

Numba's thread count is process-global, so the most recently created model sets it for every model
in the process, and it is capped at the ``NUMBA_NUM_THREADS`` value fixed when Numba was imported.

Converting to OpenCL
--------------------

``to_opencl()`` returns a :class:`~pytransit.models.ma_quadratic_cl.QuadraticModelCL` with the same
`klims` and the current data setup already applied. See :doc:`../guide/opencl` for the caveats,
particularly the single-precision one.

Examples
--------

.. toctree::
    :maxdepth: 1

    ../notebooks/models/quadratic/example_quadratic_model

API
---

.. autoclass:: pytransit.models.ma_quadratic.QuadraticModel
    :members: evaluate, to_opencl
    :special-members: __init__
