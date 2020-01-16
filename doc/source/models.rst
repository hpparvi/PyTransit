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

Basics
******

At its simplest, the data setup requires the mid-observation times. If no other other information is
given, the model assumes that all the data have been observed in a single passband and that the
exposure time is short enough so that supersampling is not needed.

.. code-block:: python

    tm.set_data(time)

Heterogeneous light curves
**************************

PyTransit can be used to model heterogeneous time series. That is, the time array can consist of many transit light curves
observed in different passbands and with different exposure times (requiring different supersampling rates). For this to
work, the model first needs to assign each individual exposure to a single *light curve*. This is done by passing the
model an integer array of light curve indices (`lcids`), where each element maps an exposure to a light curve.

.. code-block:: python

    tm.set_data(time=[0,1,2,3,4], lcids=[0,0,0,1,1])

Just setting the light curve indices doesn't do anything by itself, but it is necessary to use the more advanced features
described below.

The model doesn't need to be told explicitly how many light curves the dataset contains, since the number
of light curves is obtained from the unique `lcids` elements.

Multiple passbands
******************

PyTransit can model transits observed in multiple passbands, where each passband has a different stellar limb darkening
profile. For this, the model needs to be given an integer array of passband indices (`pbids`), where *each element maps*
*a light curve to a single passband*. Expanding the previous example, we can tell the model that the two light curves
belong to different passbands as

.. code-block:: python

    tm.set_data(time=[0,1,2,3,4], lcids=[0,0,0,1,1], pbids=[0,1])

After this, the model expects to get a two-dimensional array of limb darkening coefficients when evaluated, as explained
later in more detail.

Supersampling
*************

If the exposure time is long (Kepler and TESS long cadence mode, for example), supersampling can
be set up by giving the exposure time (`exptime`) and supersampling rate (`nsamples`), where `exptime` and `nsamples`
are either floats or arrays.

A single float can be given when modelling a homogeneous time series

.. code-block:: python

    tm.set_data(time, exptime=0.02, nsamples=10)

in which case the whole time series will have a constant supersampling rate. An array of per-light-curve values can be
given when modelling heterogeneous time series

.. code-block:: python

    tm.set_data(time=[0,1,2,3,4], lcids=[0,0,0,1,1], exptime=[0.0007, 0.02], nsamples=[1, 10])

in which case each light curve will have a separate supersampling rate.

Advanced example
****************

For a slightly more advanced example, a set of three light curves, two observed in one passband and the third in another
passband, with times

.. code-block:: python

    times_1 (lc = 0, pb = 0, sc) = [1, 2, 3, 4]
    times_2 (lc = 1, pb = 0, lc) = [3, 4]
    times_3 (lc = 2, pb = 1, sc) = [1, 5, 6]

would be set up as

.. code-block:: python

    tm.set_data(time  = [1, 2, 3, 4, 3, 4, 1, 5, 6],
                lcids = [0, 0, 0, 0, 1, 1, 2, 2, 2],
                pbids = [0, 0, 1],
                nsamples = [  1,  10,   1],
                exptimes = [0.1, 1.0, 0.1])


Model evaluation
----------------

.. code-block:: python

    tm.evaluate_ps()

.. code-block:: python

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
