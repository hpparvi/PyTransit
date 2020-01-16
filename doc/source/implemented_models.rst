Implemented transit models
==========================

PyTransit implements a set of transit models that all share a common interface that is described in more detail in
:doc:`models`.

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

Power-2 model (`pytransit.QPower2Model` and `pytransit.QPower2ModelCL`) implements the transit model with a power-2 law
limb darkening profile presented by Maxted & Gill
`(A&A 622, A33 2019) <https://www.aanda.org/articles/aa/abs/2019/02/aa34563-18/aa34563-18.html>`_.
The model is fast to evaluate and aims to model the limb darkening accurately for *cool stars*.

**Notes:**

- CPU and GPU implementations available in PyTransit.
- Accurate limb darkening model for cool stars.
- Fast to evaluate.

Giménez model
-------------

.. warning::

    The Giménez model is currently being rewritten from the original Fortran implementation described in Parviainen (2015) and not functional.

The Giménez model (`pytransit.GimenezModel`) implements the flexible transit model presented in
Giménez `(A&A 450, 2006) <https://www.aanda.org/articles/aa/abs/2006/18/aa4445-05/aa4445-05.html>`_. The stellar limb darkening follows a "generic" limb darkening model, and the accuracy
of limb darkening can be increased as needed.

**Notes:**

- A flexible model that can calculate the transit for any number of limb darkening coefficients.
- Somewhat slower to evaluate than the specialized 0-, 1-, and 2-coefficient limb darkening models.
- The four-coefficient model presented in Mandel & Agol is not implemented in PyTransit since the Giménez model offers the same functionality with higher flexibility.

Chromosphere model
------------------

Optically thin shell model by Schlawin et al. `(ApJL 722, 75-79, 2010) <https://iopscience.iop.org/article/10.1088/2041-8205/722/1/L75>`_
to model a transit over a chromosphere.