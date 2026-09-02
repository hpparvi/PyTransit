Limb darkening functions
========================

``pytransit.models.limb_darkening`` holds the Numba-compiled limb darkening profiles that
:class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` uses. See
:doc:`../guide/limb_darkening` for the narrative description and for how to supply your own.

Each law is defined by up to three functions:

``ld_<model>(mu, pv)``
    The intensity profile, evaluated at the cosine of the viewing angle ``mu`` for the coefficients
    ``pv``.

``ldi_<model>(pv)``
    The disk-integrated stellar intensity :math:`2\pi \int_0^1 I(\mu)\, z\, dz`, where an analytic
    solution exists. RoadRunner uses it to normalise the model; without one, it integrates the
    profile numerically on every evaluation.

``ldd_<model>(mu, pv)``
    The derivatives of the profile, where implemented.

A model is passed to RoadRunner either by name or as the ``(ld_<model>, ldi_<model>)`` pair.

.. code-block:: python

    from pytransit import RoadRunnerModel
    from pytransit.models.limb_darkening import ld_power_2, ldi_power_2

    tm = RoadRunnerModel('power-2')                     # by name
    tm = RoadRunnerModel((ld_power_2, ldi_power_2))     # equivalently, by function pair

.. note::

    These are Numba ``CPUDispatcher`` objects rather than plain Python functions. They can be
    called from Python and from other Numba-compiled functions alike, and a profile you write
    yourself must be ``@njit``-decorated to be usable by the model.

Uniform
-------

.. autofunction:: pytransit.models.limb_darkening.ld_uniform
.. autofunction:: pytransit.models.limb_darkening.ldi_uniform

Linear
------

.. autofunction:: pytransit.models.limb_darkening.ld_linear
.. autofunction:: pytransit.models.limb_darkening.ldi_linear
.. autofunction:: pytransit.models.limb_darkening.ldd_linear

Quadratic
---------

.. autofunction:: pytransit.models.limb_darkening.ld_quadratic
.. autofunction:: pytransit.models.limb_darkening.ldi_quadratic
.. autofunction:: pytransit.models.limb_darkening.ldd_quadratic

Quadratic, triangular parametrisation
-------------------------------------

.. autofunction:: pytransit.models.limb_darkening.ld_quadratic_tri
.. autofunction:: pytransit.models.limb_darkening.ldi_quadratic_tri

Power-2
-------

.. autofunction:: pytransit.models.limb_darkening.ld_power_2
.. autofunction:: pytransit.models.limb_darkening.ldi_power_2
.. autofunction:: pytransit.models.limb_darkening.ldd_power_2

Power-2, (h1, h2) parametrisation
---------------------------------

.. autofunction:: pytransit.models.limb_darkening.ld_power_2_pm
.. autofunction:: pytransit.models.limb_darkening.ldi_power_2_pm

Square root
-----------

.. autofunction:: pytransit.models.limb_darkening.ld_square_root
.. autofunction:: pytransit.models.limb_darkening.ldi_square_root

Logarithmic
-----------

.. autofunction:: pytransit.models.limb_darkening.ld_logarithmic
.. autofunction:: pytransit.models.limb_darkening.ldi_logarithmic

Exponential
-----------

.. autofunction:: pytransit.models.limb_darkening.ld_exponential
.. autofunction:: pytransit.models.limb_darkening.ldi_exponential

Non-linear
----------

.. autofunction:: pytransit.models.limb_darkening.ld_nonlinear
.. autofunction:: pytransit.models.limb_darkening.ldi_nonlinear

General
-------

.. autofunction:: pytransit.models.limb_darkening.ld_general
.. autofunction:: pytransit.models.limb_darkening.ldi_general

Evaluation helpers
------------------

These evaluate a limb darkening model for 1D, 2D, or 3D coefficient arrays, normalising the result
to the ``(npv, npb, nmu)`` layout the models expect.

.. autofunction:: pytransit.models.limb_darkening.evaluate_ld
.. autofunction:: pytransit.models.limb_darkening.evaluate_ldi
