The OpenCL backend
==================

Several models have OpenCL implementations that run on a GPU. On capable hardware they can be
orders of magnitude faster than the CPU versions, especially for long-cadence data where each
exposure requires many supersampled evaluations.

.. list-table::
    :header-rows: 1
    :widths: 42 42 16

    * - CPU model
      - OpenCL model
      - Import
    * - :class:`~pytransit.models.ma_uniform.UniformModel`
      - :class:`~pytransit.models.ma_uniform_cl.UniformModelCL`
      - top level
    * - :class:`~pytransit.models.ma_quadratic.QuadraticModel`
      - :class:`~pytransit.models.ma_quadratic_cl.QuadraticModelCL`
      - top level
    * - :class:`~pytransit.models.qpower2.QPower2Model`
      - :class:`~pytransit.models.qpower2_cl.QPower2ModelCL`
      - top level
    * - :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel`
      - :class:`~pytransit.models.roadrunner.rrmodel_cl.RoadRunnerModelCL`
      - by module

The first three are importable straight from ``pytransit``; `RoadRunnerModelCL` is not exported at
the top level and must be imported from its module:

.. code-block:: python

    from pytransit.models.roadrunner.rrmodel_cl import RoadRunnerModelCL

Requirements
------------

The OpenCL models need `PyOpenCL <https://documen.tician.de/pyopencl/>`_ and a working OpenCL
runtime for your device. PyTransit does not require PyOpenCL, so the OpenCL models are unavailable
until it is installed.

.. code-block:: bash

    pip install pyopencl

Usage
-----

The OpenCL models work like their CPU counterparts. The only difference in setup is that the
OpenCL context and command queue can be given in the initialiser.

.. code-block:: python

    import pyopencl as cl
    from pytransit import QuadraticModelCL

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    tm = QuadraticModelCL(cl_ctx=ctx, cl_queue=queue)
    tm.set_data(time)
    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

If the context and queue are omitted, the model creates a default context with
``cl.create_some_context()``. That is convenient for interactive work, but note that it may prompt
for a device choice, and that every model created this way gets its own context. Create one context
and share it between models in a script.

An existing `QuadraticModel` can be converted directly, carrying its data setup across:

.. code-block:: python

    tm_cl = tm.to_opencl()

Keeping the model on the device
-------------------------------

``evaluate`` accepts a ``copy`` argument. With ``copy=True`` (the default) the computed fluxes are
transferred from device memory back to a NumPy array. With ``copy=False`` the transfer is skipped
and the model stays on the device.

.. code-block:: python

    tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi, copy=False)

This is worth doing only when the next step of the computation also runs on the GPU. If the
likelihood is evaluated on the host, the transfer has to happen anyway.

.. note::

    The device buffer holding the fluxes is a private attribute, and its name is not consistent
    across the models: it is ``_b_f`` in `QuadraticModelCL`, `QPower2ModelCL`, and
    `RoadRunnerModelCL`, but ``_b_flux`` in `UniformModelCL`.

Differences from the CPU models
-------------------------------

.. warning::

    **The OpenCL models compute in single precision.** This is the most important practical
    difference. It can affect extremely shallow transits, and it will certainly corrupt the model
    if the times are given as raw Julian dates: a JD near 2 460 000 has no single-precision
    precision left for the fraction of a day where the transit lives.

    Always subtract a constant epoch from the times before handing them to an OpenCL model, for
    example ``time - floor(time.mean())``.

Other differences to be aware of:

- The OpenCL `set_data` does not accept `epids`, so the epoch indexing used for TTV modelling is
  unavailable on the GPU.
- The CPU and OpenCL implementations aim for the same functionality, but the CPU side is further
  ahead. Not every feature of a CPU model is available in its OpenCL counterpart.

When the GPU is not worth it
----------------------------

The GPU has to be fed. Every evaluation copies parameters to the device and, unless ``copy=False``,
copies fluxes back. For short-cadence data with few supersamples, the amount of computation per
exposure is small and this transfer dominates, so a multi-threaded Numba model on the CPU is often
faster. The GPU wins when the computation per exposure is large: long-cadence data with high
supersampling rates, or large parameter populations.

Measure before committing to either backend.
