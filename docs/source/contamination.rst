Contamination
=============

``pytransit.contamination`` models *flux contamination*, also called third light or blending: light
from a star other than the transit host falling into the same photometric aperture.

Why it matters
--------------

A transit dilutes the total flux by a fraction set by the radius ratio. If a fraction :math:`c` of
the light in the aperture comes from an unrelated star, the observed transit is shallower than the
true one, and the radius ratio inferred from it is too small:

.. math::

    k_\mathrm{apparent} = k_\mathrm{true}\sqrt{1 - c}.

The two helper functions :func:`~pytransit.contamination.true_radius_ratio` and
:func:`~pytransit.contamination.apparent_radius_ratio` convert between the two.

The complication that makes this a modelling problem rather than a single correction is that
**contamination is wavelength-dependent**. The host and the contaminant generally have different
effective temperatures, so their flux ratio varies across the spectrum. A cool contaminant blended
with a hot host contributes little in the blue and much in the red, making the transit look
shallower in the red than in the blue -- exactly the signature a real wavelength-dependent radius
ratio would produce.

This is the classic false positive in multicolour transit photometry, and it is also why
contamination has to be modelled jointly across passbands rather than fitted independently in each.

Passbands and instruments
-------------------------

A contamination model needs to know which passbands the observations were made in.

:class:`~pytransit.contamination.filter.Filter` and its subclasses describe a passband's
transmission as a function of wavelength in nanometres.

.. list-table::
    :header-rows: 1
    :widths: 26 74

    * - Class
      - Description
    * - :class:`~pytransit.contamination.filter.BoxcarFilter`
      - Unit transmission between two wavelengths, zero outside. A good approximation for many
        broadband filters.
    * - :class:`~pytransit.contamination.filter.TabulatedFilter`
      - A measured transmission curve, interpolated with a cubic spline.
    * - :class:`~pytransit.contamination.filter.DeltaFilter`
      - Monochromatic, for narrow spectroscopic bins.

The module ships with the SDSS *g'*, *r'*, *i'*, and *z'* passbands as boxcar approximations, and
the Kepler passband as a tabulated curve.

:class:`~pytransit.contamination.instrument.Instrument` bundles the passbands, optionally with
detector quantum efficiency curves. **The order of the filters fixes the passband order used
everywhere downstream**, including the passband indices (`pbids`) given to a transit model.

.. code-block:: python

    from pytransit.contamination import Instrument, sdss_g, sdss_r, sdss_i, sdss_z

    instrument = Instrument('MuSCAT2', (sdss_g, sdss_r, sdss_i, sdss_z))
    instrument.pb_names
    # ["g'", "r'", "i'", "z'"]

Contamination models
--------------------

Both models are parametrised the same way: by the contamination **in a chosen reference passband**
and the effective temperatures of the host and the contaminant. Everything else follows. This is a
deliberate choice -- it reduces a per-passband nuisance parameter set to three physically meaningful
numbers, and it enforces the physical relation between the passbands that makes contamination
distinguishable from a real transmission spectrum.

:class:`~pytransit.contamination.contamination.BBContamination` approximates both stars as black
bodies. It is fast, has no data dependencies, and is adequate when the temperatures are not too
low.

:class:`~pytransit.contamination.contamination.SMContamination` uses real stellar spectrum models,
either the BT-Settl or the Husser et al. (2013) PHOENIX grid shipped with PyTransit (see
:doc:`stars`). It is the better choice for cool stars, where the black body approximation breaks
down badly: molecular absorption removes large fractions of the flux from specific bands, and no
black body reproduces that.

.. code-block:: python

    from numpy import array, full
    from pytransit.contamination import Instrument, SMContamination, sdss_g, sdss_r, sdss_i, sdss_z

    instrument = Instrument('MuSCAT2', (sdss_g, sdss_r, sdss_i, sdss_z))
    cm = SMContamination(instrument, ref_pb="i'")

    # 30% contamination in i', a 5500 K host, and a 3500 K contaminant
    cm.contamination(cref=0.3, teff1=5500.0, teff2=3500.0)
    # array([0.0939, 0.1201, 0.3   , 0.3944])

The contamination is 30% in the reference passband by construction, only 9% in *g'*, and 39% in
*z'*: the cool contaminant contributes far more red light than blue. That wavelength dependence is
the signal a contamination analysis exploits.

.. note::

    ``teff1`` and ``teff2`` must have the same shape. Pass both as scalars, or both as arrays of
    equal length -- use ``full(teff2.size, teff1)`` to scan a range of contaminant temperatures
    against a fixed host.

``c_as_pandas`` and ``c_as_xarray`` return the same numbers as labelled arrays, which is much
easier to work with when scanning temperatures.

.. code-block:: python

    teff2 = array([3000.0, 3500.0, 4000.0])
    cm.c_as_pandas(0.3, full(teff2.size, 5500.0), teff2)

    # passband      g'      r'   i'      z'
    # teff
    # 3000      0.0468  0.0540  0.3  0.4802
    # 3500      0.0939  0.1201  0.3  0.3944
    # 4000      0.1461  0.2118  0.3  0.3386

Applying contamination to a light curve
---------------------------------------

:func:`~pytransit.contamination.contamination.contaminate_light_curve` blends a modelled transit
with a per-passband contamination array,

.. math::

    F_\mathrm{obs} = c + (1 - c)\, F_\mathrm{model},

using the passband index of each exposure.

.. code-block:: python

    from pytransit.contamination import contaminate_light_curve

    flux = tm.evaluate(k=0.1, ldc=ldc, t0=0.0, p=1.0, a=3.0, i=0.5*pi)
    contaminated = contaminate_light_curve(flux, cm.contamination(0.3, 5500.0, 3500.0),
                                           pbids=pbids_per_exposure)

.. note::

    The `pbids` argument here maps *each exposure* to a passband, unlike the `pbids` given to
    `set_data`, which maps each *light curve* to a passband. Use ``model.pbids[model.lcids]`` to
    convert.

Planning observations
---------------------

``exposure_times`` gives the exposure times that yield equal flux in each passband, given a
reference exposure time in the reference passband. This is useful when planning multicolour
photometry with an instrument that can set exposure times per channel.

API
---

.. automodule:: pytransit.contamination
    :no-index:

Radius ratio conversions
************************

.. autofunction:: pytransit.contamination.true_radius_ratio

.. autofunction:: pytransit.contamination.apparent_radius_ratio

.. autofunction:: pytransit.contamination.contamination.contaminate_light_curve

Passbands
*********

.. autoclass:: pytransit.contamination.filter.Filter
    :members:

.. autoclass:: pytransit.contamination.filter.BoxcarFilter
    :members:
    :special-members: __init__

.. autoclass:: pytransit.contamination.filter.TabulatedFilter
    :members:
    :special-members: __init__

.. autoclass:: pytransit.contamination.filter.DeltaFilter
    :members:
    :special-members: __init__

.. autoclass:: pytransit.contamination.instrument.Instrument
    :members:

Contamination models
********************

.. autoclass:: pytransit.contamination.contamination.BBContamination
    :members:
    :inherited-members:
    :special-members: __init__

.. autoclass:: pytransit.contamination.contamination.SMContamination
    :members:
    :inherited-members:
    :special-members: __init__
