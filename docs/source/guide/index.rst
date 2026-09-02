User guide
==========

This guide describes the parts of PyTransit that are common to every transit model: how a model is
created, how it is told about the observations, how it is evaluated, and how stellar limb darkening
enters the calculation. The individual models are described in :doc:`../models/index`.

The three-step workflow
-----------------------

Every PyTransit transit model is used in the same three steps.

.. code-block:: python

    from pytransit import RoadRunnerModel

    tm = RoadRunnerModel('quadratic')             # 1. create the model
    tm.set_data(time)                             # 2. set the observations
    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1],     # 3. evaluate for a set of parameters
                       t0=0.0, p=1.0, a=3.0, i=0.5*pi)

The separation between steps 2 and 3 is the reason PyTransit is fast. Observation times change
rarely; parameters change constantly. Everything that depends only on the times -- index arrays,
supersampling offsets, interpolation tables -- is computed once in `set_data` and reused by every
later `evaluate` call. Inside an optimiser or a sampler, only step 3 runs, millions of times.

Create the model once, outside your log posterior function, and call `evaluate` inside it.

Conventions
-----------

PyTransit uses a consistent set of units and conventions throughout.

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Quantity
      - Convention
    * - Time
      - Days, normally BJD. Only differences matter, so any consistent zero point works.
    * - Flux
      - Normalised so that the out-of-transit level is 1.0.
    * - Angles
      - Radians.
    * - Wavelength
      - Nanometres, in the contamination and stellar spectrum modules.
    * - Arrays
      - NumPy ``float64``. Lists are accepted and converted, except by
        :class:`~pytransit.models.qpower2.QPower2Model`, which needs ``ldc`` as an ndarray.
    * - Distances
      - Stellar radii, unless stated otherwise.

Backends
--------

The models come in two flavours.

**Numba (default).** Just-in-time compiled CPU implementations, found in ``pytransit/models/`` and
``pytransit/models/numba/``. Several models are multi-threaded. This is often the better choice
when modelling large amounts of short-cadence data, where the cost of moving data between the GPU
and main memory would dominate.

**OpenCL.** GPU implementations, in files suffixed ``_cl.py``. On a powerful GPU these can be
orders of magnitude faster than the CPU versions, especially for long-cadence data where the
amount of computation per exposure is large. See :doc:`opencl`.

Both flavours aim to offer the same functionality, but the feature sets are not identical; the CPU
implementations are further ahead. Only some models have an OpenCL version.

.. note::

    The first call to a Numba-accelerated model triggers JIT compilation and is therefore much
    slower than the calls that follow. Time your code from the second call onwards, and expect a
    one-off delay of a few seconds when a model is evaluated for the first time in a session.

.. toctree::
    :maxdepth: 2

    interface
    data_setup
    evaluation
    limb_darkening
    opencl
