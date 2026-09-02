Eclipse models
==============

A *secondary eclipse*, or occultation, happens when the planet passes behind its star and its
thermal emission and reflected light are removed from the total flux. The geometry is the transit
geometry with the roles reversed, and the depth measures the planet-star flux ratio rather than the
radius ratio.

PyTransit has two eclipse models: :class:`~pytransit.models.eclipse_model.EclipseModel` for
broadband photometry, and :class:`~pytransit.models.roadrunner.esmodel.EclipseSpectroscopyModel`
(``ESModel``) for spectroscopic time series.

Neither takes limb darkening coefficients. The occulted body is the planet, whose dayside is close
enough to uniform for the purpose, and the star's limb darkening does not enter at all.

.. note::

    :class:`~pytransit.models.ma_uniform.UniformModel` also models an eclipse when created with
    ``eclipse=True``. It gives the bare occultation profile; `EclipseModel` adds the flux ratio,
    which is normally what you want.

EclipseModel
------------

The depth is set by the planet-star flux ratio `fr`. With `fr` given, the model returns a light
curve normalised to unity out of eclipse and dipping to ``1 - fr`` at the bottom, so it multiplies
straight into a transit model. Without `fr`, it returns the raw uniform-disk occultation profile,
which is useful when the flux ratio is applied later or fitted per passband.

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import EclipseModel

    time = linspace(0.4, 0.6, 1000)

    em = EclipseModel()
    em.set_data(time)

    flux = em.evaluate(k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi, fr=1e-3)

The ``multiplicative=True`` option returns the fraction of the planet disk that is visible instead
of a flux, which is the natural quantity to multiply an arbitrary planetary emission model by.

.. autoclass:: pytransit.models.eclipse_model.EclipseModel
    :members: evaluate
    :special-members: __init__

EclipseSpectroscopyModel
------------------------

The eclipse counterpart of
:class:`~pytransit.models.roadrunner.tsmodel.TransmissionSpectroscopyModel`. It models many
wavelength bins that share a single eclipse, computing the geometry once and reusing it across all
the bins.

The wavelength-dependent planet-star flux ratio is the quantity of interest in eclipse
spectroscopy, so it is the *first* argument of `evaluate` rather than a trailing keyword.

.. code-block:: python

    from numpy import pi, linspace, full
    from pytransit import ESModel

    time = linspace(0.4, 0.6, 1000)
    npb = 100

    em = ESModel()
    em.set_data(time)

    flux = em.evaluate(f=full(npb, 1e-3), k=0.1, t0=0.0, p=1.0, a=3.0, i=0.5*pi, rstar=1.2)

.. important::

    The model applies the **light travel time correction** between the transit and the secondary
    eclipse. The eclipse is observed late by roughly :math:`2 a R_\star / c`, about 40 s for a hot
    Jupiter -- comparable to the timing precision of a good eclipse measurement, so it is not
    negligible. The correction needs a physical stellar radius, given through `rstar` in solar
    radii. It defaults to 1.0, so set it for anything other than a solar-radius host.

.. autoclass:: pytransit.models.roadrunner.esmodel.EclipseSpectroscopyModel
    :members: evaluate
    :special-members: __init__
