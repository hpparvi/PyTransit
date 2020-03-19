PyTransit
=========

Welcome to PyTransit documentation! PyTransit is a package for exoplanet transit light curve modelling
that offers optimised CPU and GPU implementations of exoplanet transit models with a unified interface. Transit model
evaluation is trivial for simple use-cases, such as homogeneous light curves observed in a single passband, but also
straightforward for more complex use-cases, such as when dealing with *heterogeneous light curves containing transits
observed in different passbands and different instruments*, or with *transmission spectroscopy*.

The development of PyTransit began in 2009 to fill the need for a fast and reliable
exoplanet transit modelling toolkit for Python. Since then, PyTransit has gone through several
iterations, always thriving to be *the fastest and most versatile* exoplanet transit modelling tool for Python.

PyTransit v1.0 was described in Parviainen (2015), which also details the model-specific optimisations and model
performance. This version relied heavily on Fortran code, which made the installation complicated in non-Linux
systems. PyTransit v2 replaces all the Fortran routines with numba-accelerated python routines, and
aims to implement all the major functionality also in OpenCL.

While PyTransit is aimed to work as a library offering tools for customised transit analysis codes, it
can also be used directly for transit modelling and parameter estimation.

Example
-------

The transit model initialization is straightforward. At its simplest, the model takes an array the mid-exposure times,

.. code-block:: python

    from pytransit import QuadraticModel

    tm = QuadraticModel()
    tm.set_data(times)

after which it is ready to be evaluated

.. code-block:: python

    tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

.. image:: basic_example_1.svg

To complicate the situation a bit, we can consider a case where we want to model several transits observed in different
passbands. The stellar limb darkening varies from passband to passband, so we need to give a set of limb darkening
coefficients for each passband, and we may also want to allow the radius ratio to vary from passband to passband.
Now, we will only need to initialise the model with per-exposure light curve indices (`lcids`) and per-light-curve
passband indices (`pbids`) (don't worry, these are simple integer arrays), after which we are ready to evaluate the
model with passband-dependent radius ratio and limb darkening

.. code-block:: python

    tm.set_data(times, lcids=lcids, pbids=pbids)
    tm.evaluate(k=[0.10, 0.12], ldc=[[0.2, 0.1, 0.5, 0.1]], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

.. image:: basic_example_2.svg

We made both the radius ratio and limb darkening passband-dependent in the example above, but we could just as well
evaluate the model with a single scalar radius ratio (as in the first example), in which case only the limb darkening
would be passband-dependent.

We may often want to evaluate the model for a large set of parameters at the same time (such as when doing MCMC
sampling with *emcee*, or using some other population-based sampling or minimization method). Give `evaluate` an array
of parameters

.. code-block:: python

    tm.evaluate(k=[[0.10, 0.12], [0.11, 0.13]],
                ldc=[[0.2, 0.1, 0.5, 0.1],[0.4, 0.2, 0.75, 0.1]],
                t0=[0.0, 0.01], p=[1, 1], a=[3.0, 2.9], i=[.5*pi, .5*pi])

.. image:: basic_example_3.svg

and PyTransit will calculate the models for the whole parameter set in parallel.

All the models come in CPU and GPU (OpenCL) versions. OpenCL versions can be 10-20 times faster to evaluate
than the CPU versions (depending on the GPU), and switching to use the GPU is as simple as just importing an `CL` version
of the model, which will work identically to the CPU version

.. code-block:: python

    from pytransit import QuadraticModelCL

The examples use a transit model with quadratic limb darkening by Mandel & Agol, but all the models follow the same API,
so when you learn to use one, you can use all of them. Finally, these examples show just the use of the main `evaluate`
method, but the models have also more optimized evaluation methods for specific use-cases, and the package comes with
utilities for specialized analyses.

Contents
--------

.. toctree::
    :maxdepth: 2

    installation
    quickstart
    models
    implemented_models
    lpfs
    advanced
    api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
