Implemented transit models
==========================

PyTransit implements a set of transit models that all share a common API.

Uniform model
-------------

The uniform model (`pytransit.UniformModel`) reproduces an exoplanet transit over a uniform disc.
This model is useful when modelling secondary eclipses, or when the effects from the stellar limb
darkening can be ignored. Since the model assumes zero limb darkening, the model evaluation doesn't
allow limb darkening coefficients to be given.

Quadratic model
---------------

The quadratic transit model (`pytransit.QuadraticModel`) reproduces an exoplanet transit over a
stellar disk with the limb darkening modelled by a quadratic limb darkening model, as presented
in Mandel & Agol (2001).

Power-2 model
-------------

Power-2 transit model (`pytransit.QPower2`) by Maxted & Gill (A&A 622, A33 2019).

Giménez model
-------------

.. warning::

    The Giménez model is currently being rewritten from the original Fortran implementation described
    in Parviainen (2015) to Numba, and not functional.

The Giménez model (`pytransit.GimenezModel`) implements the flexible transit model presented in
Giménez (200x). The stellar limb darkening follows a "generic" limb darkening model, and the accuracy
of limb darkening can be increased as needed.

.. note::

    The four-coefficient model presented in Mandel & Agol is not implemented in PyTransit since
    the Giménez model offers the same functionality with higher flexibility.

Chromosphere model
------------------

Optically thin shell model by Schlawin et al. (ApJL 722, 75--79, 2010) to model a transit over a chromosphere.