PyTransit
=========

Fast and easy-to-use tools for exoplanet transit light curve modelling with Python.

Welcome to the PyTransit documentation! PyTransit is a package for exoplanet transit light curve modelling
that offers optimised CPU and GPU implementations of exoplanet transit models with a unified interface.

The first iteration of PyTransit was developed in 2009 to fill the need for a fast and reliable
exoplanet transit modelling toolkit for Python. Since then, PyTransit has gone through several
iterations, always thriving to be *the fastest and most versatile* exoplanet transit modelling tool for Python.

PyTransit v1.0 was described in Parviainen (2015), which also details the model-specific optimisations and model
performance. This version relied heavily on Fortran code, which made the installation complicated in non-Linux
systems. The current version replaces all the Fortran routines with numba-accelerated python routines, and
aims to implement all the major functionality also in OpenCL.

.. code-block::

    from pytransit import QuadraticModel

    tm = QuadraticModel()
    tm.set_data(time)

    flux = tm.evaluate_ps(k, ldc, t0, p, a, i, e, w)

or using either a parameter array

.. code-block::

    flux = tm.evaluate_pv(pv, ldc)


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    models
    lpfs
    advanced
    opencl

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
