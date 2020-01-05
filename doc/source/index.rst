PyTransit
=========

Welcome to the PyTransit documentation! PyTransit is a package for exoplanet transit light curve modelling
that offers optimised CPU and GPU implementations of exoplanet transit models with a unified interface.

The development of PyTransit began in 2009 to fill the need for a fast and reliable
exoplanet transit modelling toolkit for Python. Since then, PyTransit has gone through several
iterations, always thriving to be *the fastest and most versatile* exoplanet transit modelling tool for Python.

PyTransit v1.0 was described in Parviainen (2015), which also details the model-specific optimisations and model
performance. This version relied heavily on Fortran code, which made the installation complicated in non-Linux
systems. PyTransit v2 replaces all the Fortran routines with numba-accelerated python routines, and
aims to implement all the major functionality also in OpenCL.

While PyTransit is aimed to work as a library offering tools for customised transit analysis codes, it
can also be used almost directly for transit modelling and parameter estimation.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    quickstart
    models
    implemented_models
    lpfs
    advanced

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
