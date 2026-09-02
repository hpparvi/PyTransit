Transmission spectroscopy model
===============================

:class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel` -- also exported as
``TSModel`` -- is the RoadRunner model specialised for transmission spectroscopy.

The problem it solves
---------------------

A transmission spectroscopy dataset is a set of spectroscopic light curves extracted from a single
spectroscopic time series. Every wavelength bin is observed at the *same* times, through the *same*
transit, of the *same* planet. What differs between bins is the radius ratio -- the transmission
spectrum we are after -- and the limb darkening.

Evaluating a general model bin by bin would recompute the identical transit geometry once per bin.
With hundreds or thousands of bins, as JWST routinely produces, that is the dominant cost and it is
entirely wasted.

This model computes the projected star-planet distances and the overlap weights once, then reuses
them across all the wavelength bins. The saving grows with the number of bins.

Differences from the other models
---------------------------------

.. important::

    **The number of passbands comes from ``ldc``, not from ``set_data``.** The model reads ``npb``
    from the shape of the limb darkening array on every call. A dataset can therefore be re-binned
    in wavelength without touching the data setup.

    **``evaluate`` always returns a 3D array** of shape ``(npv, npb, npt)``, even for a single
    parameter vector, where it is ``(1, npb, npt)``.

Because all the bins share one time array, `set_data` takes the times only. The `lcids` and `pbids`
arrays play no role here -- there is one light curve per wavelength bin by construction.

Usage
-----

.. code-block:: python

    from numpy import pi, linspace, full, tile
    from pytransit import TSModel

    time = linspace(-0.1, 0.1, 1000)
    npb = 100                                    # wavelength bins

    tm = TSModel()
    tm.set_data(time)

    k = full(npb, 0.1)                           # one radius ratio per bin
    ldc = tile([0.2, 0.1], (npb, 1))             # one coefficient pair per bin

    flux = tm.evaluate(k=k, ldc=ldc, t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    # flux.shape == (1, 100, 1000)

For a population of parameter vectors, the limb darkening coefficients must be given as an explicit
3D array of shape ``(npv, npb, nldc)``; anything else raises a `ValueError`.

.. code-block:: python

    npv = 50
    flux = tm.evaluate(k=full((npv, npb), 0.1),
                       ldc=tile([0.2, 0.1], (npv, npb, 1)),
                       t0=zeros(npv), p=ones(npv), a=full(npv, 3.0), i=full(npv, 0.5*pi))
    # flux.shape == (50, 100, 1000)

Since the model inherits from :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel`, it
takes the same initialiser arguments, including the choice of limb darkening law and the
discretisation parameters.

API
---

.. autoclass:: pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel
    :members: evaluate
