Quickstart
==========

Basic transit model evaluation
------------------------------

PyTransit comes with a set of transit models that share a common interface (with small
model-specific variations). So, while we use a Mandel & Agol quadratic limb darkening model
(`pytransit.QuadraticModel`) as an example here, the evaluation works the same for all the models.

First, the transit model needs to be imported from PyTransit. After this, it can be initialised, and
set up by giving it a set of mid-observation times.

.. code-block:: python

    from pytransit import QuadraticModel

    tm = QuadraticModel()
    tm.set_data(time)

where `time` is a NumPy array (or a list or a tuple) of transit mid-times for which the model will be evaluated.

After the initialisation and setup, the transit model can be evaluated for a set of scalar parameters as

.. code-block:: python

    flux = tm.evaluate_ps(k, ldc, t0, p, a, i, e, w)

where `k` is the planet-star radius ratio, `t0` is the zero epoch, `p` is the orbital period, `a` is the scaled
semi-major axis, `i` is the inclination, `e` is the eccentricity, `w` is the argument of periastron, and
`ldc` is an `ndarray` containing the model-specific limb darkening coefficients.

The model can also be evaluated using a 1D parameter vector or a 2D parameter array (one or two dimensional NumPy ndarrays)

.. code-block:: python

    flux = tm.evaluate_pv(pv, ldc)

If `pv` is one dimensional, it is assumed to contain the model parameters `[k, t0, p, a, i, e, w]`. If `pv` is
two-dimensional, it is assumed to contain `n` parameter vectors, one per row (that is, the parameter array should have
a shape `(n, 7)`).

In the 2D parameter array case, the model is evaluated for all the `n` parameter vectors in parallel and the `flux` array
will have a shape `(n, time.size)`. The parallelisation can yield significant gains in evaluation speed (especially
with the OpenCL versions of the transit models), and `TransitModel.evaluate_pv` with a 2D parameter array should be used
always when evaluating the model for a set of parameters (such as when doing MCMC with *emcee*).

.. note::

    Both the `TransitModel.evaluate_ps` and the 1D version of 'TransitModel.evaluate_pv'
    use the 2D version of 'TransitModel.evaluate_pv' under the hood.