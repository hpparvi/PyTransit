Transmission spectroscopy model
===============================

.. note::

    This page is a stub. Worked examples and figures will be added in a future release. The full
    API and usage documentation is on the
    :doc:`transmission spectroscopy reference page <../models/tsmodel>`.

A transmission spectroscopy dataset is a set of spectroscopic light curves carved out of a single
spectroscopic time series. Every wavelength bin sees the *same* times, the *same* orbit, the *same*
planet. What differs between bins is the radius ratio -- the transmission spectrum itself -- and
the limb darkening.

Evaluating a general transit model bin by bin recomputes that identical geometry once per bin.
With the hundreds or thousands of bins JWST routinely produces, that recomputation is the dominant
cost, and all of it is wasted.

:class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel` computes the projected
star-planet distances and the overlap weights once and reuses them across every wavelength bin. The
saving grows with the number of bins, which is exactly the direction modern datasets are heading.

The model also drops an assumption the other models make: the number of passbands comes from the
shape of the limb darkening array on each call rather than from `set_data`, so a dataset can be
re-binned in wavelength without touching the data setup.

.. seealso::

    :doc:`../models/tsmodel` for the API, and :doc:`../guide/evaluation` for the array shapes it
    returns.
