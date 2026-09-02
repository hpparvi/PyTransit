Model evaluation
================

`evaluate` computes the model flux for a set of physical parameters. It accepts scalars, 1D
arrays, and 2D arrays, and the shapes of the arguments determine both what is computed and the
shape of the result. This page describes those rules.

Three ways to evaluate
----------------------

**A single parameter set.** All the orbital parameters are scalars. This is the ordinary case, and
the fastest per model.

.. code-block:: python

    tm.set_data(time)
    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    # flux.shape == (npt,)

**Passband-dependent parameters.** With several passbands defined, the radius ratio and the limb
darkening coefficients may vary between them while the orbit stays shared.

.. code-block:: python

    tm.set_data(time, lcids=lcids, pbids=pbids)                      # two passbands
    flux = tm.evaluate(k=[0.10, 0.12], ldc=[[0.2, 0.1], [0.5, 0.1]],
                       t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    # flux.shape == (npt,)

The result is still a single light curve: each exposure is modelled with the parameters belonging
to its own passband.

**A parameter population.** Population-based samplers and optimisers -- ``emcee``, differential
evolution -- need the model evaluated for many parameter vectors at once. Give the orbital
parameters as 1D arrays of length ``npv`` and the radius ratio as a 2D array.

.. code-block:: python

    flux = tm.evaluate(k=[[0.10, 0.12], [0.11, 0.13]],
                       ldc=[[0.2, 0.1, 0.5, 0.1], [0.4, 0.2, 0.75, 0.1]],
                       t0=[0.0, 0.01], p=[1.0, 1.0], a=[3.0, 2.9], i=[0.5*pi, 0.5*pi])
    # flux.shape == (npv, npt)

PyTransit computes the whole population in one call, in parallel where the model supports it. This
is far faster than looping in Python, and it is the reason the models are worth using inside a
sampler at all.

The shape of ``k``
------------------

The radius ratio follows one rule that is easy to get wrong:

.. important::

    **The trailing axis of ``k`` is always the passband axis.** A 1D ``k`` gives one radius ratio
    per passband, never one per parameter vector. To vary the radius ratio across a population,
    ``k`` must be 2D with shape ``(npv, npb)`` -- including ``(npv, 1)`` when there is only one
    passband.

.. list-table::
    :header-rows: 1
    :widths: 24 76

    * - ``k``
      - Meaning
    * - ``0.1``
      - One radius ratio, shared by every passband and every parameter vector.
    * - ``[0.10, 0.12]``
      - Two passbands, one parameter vector.
    * - ``[[0.10], [0.11], [0.12]]``
      - One passband, three parameter vectors.
    * - ``[[0.10, 0.12], [0.11, 0.13]]``
      - Two passbands, two parameter vectors.

.. warning::

    A model with a single light curve selects its population branch from the shape of ``k``. If
    ``k`` is a scalar while the orbital parameters are vectors, the model returns a single light
    curve computed from the **first** parameter vector and silently ignores the rest:

    .. code-block:: python

        tm.set_data(time)
        flux = tm.evaluate(k=0.1, ldc=ldc, t0=t0s, p=ps, a=as_, i=is_)   # ps.size == 3
        # flux.shape == (npt,), not (3, npt)

    When evaluating a population, always pass ``k`` as a 2D ``(npv, npb)`` array.

The shape of ``ldc``
--------------------

The limb darkening coefficients are indexed by parameter vector, passband, and coefficient, in
that order. The model accepts the array in any of three forms and normalises it internally to
``(npv, npb, nldc)``.

.. list-table::
    :header-rows: 1
    :widths: 24 20 56

    * - Given shape
      - Interpreted as
      - Use
    * - ``(nldc,)``
      - ``(1, 1, nldc)``
      - One parameter vector, one passband.
    * - ``(npb, nldc)``
      - ``(1, npb, nldc)``
      - One parameter vector, several passbands. Requires ``npv == 1``.
    * - ``(npv, npb*nldc)``
      - ``(npv, npb, nldc)``
      - A population, with each row holding all the passbands' coefficients concatenated.
    * - ``(npv, npb, nldc)``
      - unchanged
      - The explicit form, and the least ambiguous.

The number of coefficients ``nldc`` is set by the limb darkening law: two for quadratic, power-2,
square-root, logarithmic, and exponential; one for linear; four for the non-linear law; and any
number for the general law. See :doc:`limb_darkening`.

.. note::

    A 2D ``ldc`` is ambiguous on its own -- it can be ``(npb, nldc)`` or ``(npv, npb*nldc)`` -- and
    is resolved using the number of parameter vectors, which the model reads from the length of
    ``p``. Passing the explicit 3D form removes the ambiguity entirely, and is worth doing whenever
    both ``npv`` and ``npb`` exceed one.

Return shapes
-------------

.. list-table::
    :header-rows: 1
    :widths: 40 26 34

    * - Model
      - Single parameter set
      - Population
    * - Ordinary transit models
      - ``(npt,)``
      - ``(npv, npt)``
    * - :class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel`
      - ``(1, npb, npt)``
      - ``(npv, npb, npt)``
    * - :class:`~pytransit.models.roadrunner.esmodel.EclipseSpectroscopyModel`
      - ``(1, npb, npt)``
      - ``(npv, npb, npt)``

The ordinary models squeeze out length-one axes, so a single parameter set gives a plain 1D flux
array. The spectroscopy models always keep all three axes, because a wavelength axis of length one
is still a wavelength axis.

Eccentric orbits
----------------

The eccentricity `e` and the argument of periastron `w` default to a circular orbit and can be left
out entirely. When they are given, both must be:

.. code-block:: python

    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi,
                       e=0.1, w=0.5*pi)

Eccentric orbits are slower to evaluate than circular ones because the projected distance no longer
has a closed form and must be solved iteratively. If the eccentricity is fixed at zero in a fit,
omit the arguments rather than passing zeros.

Performance notes
-----------------

- **Create the model once.** Model creation may build interpolation tables and always triggers JIT
  compilation on first evaluation. Do it outside the log posterior function.
- **Evaluate populations in one call.** One call with ``npv = 100`` is much faster than 100 calls
  with ``npv = 1``.
- **Reuse the time array object.** `set_data` returns immediately when handed the same array object
  with no other arguments, so calling it repeatedly in a loop is cheap -- see the warning in
  :doc:`data_setup` for the flip side of this.
- **Pass ``float64`` NumPy arrays.** Lists are converted on every call. Inside a hot loop this
  conversion can be a measurable fraction of the total cost.
- **Consider the number of threads.** :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel`
  takes an ``nthreads`` argument. Numba's thread count is process-global, so the most recently
  created model sets it for every model in the process, and it cannot exceed the
  ``NUMBA_NUM_THREADS`` value fixed when Numba was first imported.
