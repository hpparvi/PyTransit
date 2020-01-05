Transit Models
==============

PyTransit implements five of the most important exoplanet transit light curve models, each with
model-specific optimisations to make their evaluation efficient. The models come in two flavours

- Numba-accelerated implementations for CPU computing. These implementations are multi-threaded,
  and can be the best choice when modelling large amounts of short-cadence observations where the
  data transfer between the GPU and main memory would create a bottleneck.

- OpenCL implementations for GPU computing. These can be orders of magnitude faster than the CPU
  implementations if ran in a powerful GPU, especially when modelling long cadence data where
  the amount of computation per observation dominates over the time for data transfer.

The CPU and GPU implementations aim to offer equal functionality, but, at the moment of writing,
they have some variation in the available features.

Transit model interface
-----------------------

The transit models share a unified interface with small variations to account for model-specific parameters
and settings. Some of the models have also special evaluation methods aimed for specific science
cases, such as transmission spectroscopy where the light curves have been created from a spectroscopic
time series.

The models are made to work with heterogeneous photometric time series. That is, a single model
evaluation can model observations in different passbands (with different limb darkening),
different exposure times, and different supersampling rates.

Model initialisation
--------------------

Model initialisation is straightforward. The Mandel-Agol model with quadratic limb darkening can be
imported and initialised by

.. code-block:: python

    from pytransit import QuadraticModel

    tm = QuadraticModel()

After the initialisation, the model still needs to be set up by giving it the observation centre-times,
and optionally other information, such as passbands, exposure times, supersampling rates, etc.

Data setup
----------

At its simplest, the data setup requires the mid-observation times. If no other other information is
given, the model assumes that all the data have been observed in a single passband and that the
exposure time is short enough so that supersampling is not needed.

.. code-block:: python

    tm.set_data(time)

If the exposure time is long (Kepler and TESS long cadence mode, for example), supersampling can
be set up by giving the exposure time and supersampling rage

.. code-block:: python

    tm.set_data(time, exptime=xxx, nsamples=yyy)

where `exptime`and `nsamples` are floats.

Heterogeneous light curves
**************************

If the time array consists of many light curves heteregeneos

.. code-block:: python

    tm.set_data(time=[0,1,2,3,4], lcids=[0,0,0,1,1])


Multiple passbands
******************

.. code-block:: python

    tm.set_data(time=[0,1,2,3,4], lcids=[0,0,0,1,1], pbids=[0,1])

Common evaluation methods
-------------------------

.. code-block:: python

    tm.evaluate_ps()

    tm.evaluate_pv()

OpenCL
------

The OpenCL versions of the models work identically to the Python version, except
that the OpenCL context and queue can be given as arguments in the initialiser, and the model evaluation method can be
told to not to copy the model from the GPU memory. If the context and queue are not given, the model creates a default
context using `cl.create_some_context()`.

.. code-block:: python

    import pyopencl as cl
    from src import QuadraticModelCL

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    tm = QuadraticModelCL(cl_ctx=ctx, cl_queue=queue)
