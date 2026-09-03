Gravity-darkened star model
===========================

.. note::

    This page is a stub. Worked examples and figures will be added in a future release. The full
    API and usage documentation is on the
    :doc:`gravity-darkened model reference page <../models/gdmodel>`.

Every other model in PyTransit assumes the stellar disk is circular and its brightness radially
symmetric. For a rapidly rotating star, neither holds.

Centrifugal force flattens the star, so its equatorial radius exceeds its polar radius. The
flattening lowers the effective surface gravity at the equator, and by von Zeipel's theorem the
local effective temperature follows the local gravity,

.. math::

    T_\mathrm{eff} \propto g^{\beta},

which leaves the equator cooler and dimmer than the poles. This is *gravity darkening*.

For a transit it changes everything at once. The disk is not circular, its brightness is not
radially symmetric, and the light curve depends on where the planet's path crosses the star
relative to the stellar spin axis. A transit across the hot pole is deeper than one across the cool
equator, and unless the orbit is aligned with the stellar equator the light curve becomes
*asymmetric*.

That asymmetry is the point: it carries information about the spin-orbit angle, measurable from
photometry alone, without the radial velocities a Rossiter-McLaughlin measurement needs.

:class:`~pytransit.models.gdmodel.GravityDarkenedModel` implements the model of Barnes
(ApJ 705, 683, 2009), discretising the stellar surface and integrating numerically because no
radially symmetric model can represent it. Converting local temperature into observed flux needs
real stellar spectra, so the model works with the passbands and spectrum grids described in
:doc:`../contamination` and :doc:`../stars`.

.. seealso::

    :doc:`../models/gdmodel` for the API and the parameters the model adds.
