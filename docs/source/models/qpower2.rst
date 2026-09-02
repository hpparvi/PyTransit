Power-2 model
=============

:class:`~pytransit.models.qpower2.QPower2Model` implements the transit model of Maxted & Gill
(A&A 622, A33, 2019) for a star with a power-2 limb darkening profile,

.. math::

    I(\mu) = 1 - c\,(1 - \mu^{\alpha}).

Why the power-2 law
-------------------

The quadratic law is a poor description of the limb darkening of *cool stars*. It has the wrong
shape near the limb, and the mismatch grows towards the infrared. The power-2 law uses the same
number of coefficients but follows the profiles predicted by model atmospheres for M dwarfs and
other late-type stars much more closely.

The model is also fast: the qpower2 approximation is an analytic expression that avoids the
elliptic integrals of the Mandel & Agol solution.

Since the approximation is built into the model, it is at its best for small planets. Consult
Maxted & Gill (2019) for the accuracy as a function of radius ratio before using it for a
large planet.

Usage
-----

.. code-block:: python

    from numpy import pi, array, linspace
    from pytransit import QPower2Model

    time = linspace(-0.1, 0.1, 1000)

    tm = QPower2Model()
    tm.set_data(time)

    flux = tm.evaluate(k=0.1, ldc=array([0.6, 0.5]), t0=0.0, p=1.0, a=3.0, i=0.5*pi)

.. note::

    This is the one model that needs ``ldc`` as an **ndarray** rather than a list. ``evaluate``
    passes it straight to a Numba kernel, which cannot handle a Python list, so ``ldc=[0.6, 0.5]``
    raises a ``TypingError``.

The coefficients are :math:`(c, \alpha)`. When *fitting* them, consider RoadRunner with the
``'power-2-pm'`` law instead: it implements the same profile in the :math:`(h_1, h_2)`
parametrisation of Maxted (2018), whose parameters are far less correlated and therefore much
easier to sample. See :doc:`../guide/limb_darkening`.

API
---

.. autoclass:: pytransit.models.qpower2.QPower2Model
    :members: evaluate
