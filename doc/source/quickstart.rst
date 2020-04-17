Quickstart
==========

Basic transit model evaluation
------------------------------

PyTransit comes with a set of transit models that share a common interface (with small
model-specific variations). So, while we use a Mandel & Agol quadratic limb darkening model
(:class:`pytransit.QuadraticModel`) as an example here, the evaluation works the same for all the models.

First, the transit model needs to be imported from PyTransit. After this, it can be initialised, and
set up by giving it (at least) a set of mid-exposure times.

.. code-block:: python

    from pytransit import QuadraticModel

    tm = QuadraticModel()
    tm.set_data(time)

where `time` is a NumPy array (or a list or a tuple) of mid-exposure times for which the model will be evaluated.

After the initialisation and setup, the transit model can be evaluated as

.. code-block:: python

    flux = tm.evaluate(k, ldc, t0, p, a, i, e, w)

where `k` is the planet-star radius ratio, `t0` is the zero epoch, `p` is the orbital period, `a` is the scaled
semi-major axis, `i` is the inclination, `e` is the eccentricity, `w` is the argument of periastron, and
`ldc` is an `ndarray` containing the model-specific limb darkening coefficients.

The calling simplifies further if we assume a circular orbit, when we can leave `e` and `w` out

.. code-block:: python

    flux = tm.evaluate(k, ldc, t0, p, a, i)

The radius ratio can either be a scalar, a 1D vector, or a 2D array, the limb darkening coefficients are given as a
1D vector or a 2D array, and the orbital parameters (`t0`, `p`, `a`, `i`, `e`, and `w`) can be either scalars or vectors.

In the most simple case the limb darkening coefficients are given as a single vector and the rest of the parameters are
scalars, in which case the `flux` array will also be one dimensional. However, if we want to evaluate the model for multiple parameter values (such as when using *emcee* for MCMC
sampling), giving a 2D array of limb darkening coefficients and the rest of the parameters as vectors allows PyTransit
to evaluate the models in parallel, which can lead to significant performance improvements (especially with the OpenCL
versions of the transit models). Evaluating the model for `n` sets of parameters will result in a `flux` array with a
shape  `(n, time.size)`.

A third case, giving a 2D array of radius ratios (or a 1D vector of radius ratios when the orbital parameters are
scalars), is slightly more advanced, and is used when modelling multicolor photometry (or transmission spectroscopy).
In this case the model assumes the radius ratio varies from passband to passband, and the setup requires also passband
indices (see later).