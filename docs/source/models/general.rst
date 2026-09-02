General model
=============

:class:`~pytransit.models.general.GeneralModel` implements the transit model of Giménez
(A&A 450, 1231, 2006), with the optimisations of Parviainen (MNRAS 450, 3233, 2015), for the
*general* limb darkening law

.. math::

    I(\mu) = I(1)\left(1 - \sum_{n=1}^{N} u_n (1 - \mu^n)\right).

The law can be made arbitrarily flexible by adding coefficients, which is why it is called general:
the linear, quadratic, and higher-order polynomial laws are all special cases.

The model is evaluated as a series of Jacobi polynomials. The number of polynomials `npol` sets the
*accuracy* of the transit model and costs computation time directly; the number of limb darkening
coefficients `nldc` sets the *flexibility* of the limb darkening and costs almost nothing.

.. note::

    The four-coefficient non-linear law of Mandel & Agol (2002) is not implemented separately in
    PyTransit, because the Giménez law offers the same functionality with arbitrary flexibility.

.. tip::

    :doc:`RoadRunner <roadrunner>` supports the general law through its ``'general'`` limb darkening
    model and is both faster and more flexible. For new work, prefer it.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import GeneralModel

    time = linspace(-0.1, 0.1, 1000)

    tm = GeneralModel(npol=50, nldc=2)
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

Transmission spectroscopy mode
------------------------------

``mode=1`` enables a special evaluation mode that accelerates the model considerably when many
passbands share a single transit geometry, as in transmission spectroscopy.

.. code-block:: python

    tm = GeneralModel(npol=50, nldc=2, mode=1)

For new transmission spectroscopy work, :doc:`TransmissionSpectroscopyModel <tsmodel>` is the
better choice.

API
---

.. autoclass:: pytransit.models.general.GeneralModel
    :members: evaluate, to_opencl
    :special-members: __init__
