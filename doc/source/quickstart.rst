Quickstart
==========

Installation
------------

Pytransit can be installed by `pip`, or by cloning the repository from GitHub.

.. code-block:: bash

    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit
    python setup.py install

Basic transit model evaluation
------------------------------

First, the transit model needs to be imported from PyTransit. After this, it can be initialised, and
set up by giving it a set of mid-observation times.

.. code-block:: python

    from pytransit import QuadraticModel

    tm = QuadraticModel()
    tm.set_data(time)

After the initialisation and setup, the transit model can be evaluated for a set of parameters as

.. code-block:: python

    flux = tm.evaluate_ps(k, ldc, t0, p, a, i, e, w)

where `k` is the planet-star radius ratio, `t0` is the zero epoch, `p` is the orbital period, `a` is the scaled
semi-major axis, `i` is the inclination, `e` is the eccentricity, `w` is the argument of periastron, and
`ldc` is an `ndarray` containing the limb darkening coefficients.

The model can also be evaluated using a (1D or 2D) parameter array as

.. code-block:: python

    flux = tm.evaluate_pv(pv, ldc)

where pv is an `ndarray` containing `[k, t0, p, a, i, e, w]` and `ldc` is an `ndarray` containing the
limb darkening coefficients.
