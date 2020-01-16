Implemented transit models
==========================

PyTransit implements a set of transit models that all share a common interface that is described in more detail in
:doc:`models`.

Uniform model
-------------

The uniform model (`pytransit.UniformModel` and `pytransit.UniformModelCL`) reproduces an exoplanet transit over a uniform disc.
This model is useful when modelling secondary eclipses, or when the effects from the stellar limb
darkening can be ignored.

- `Uniform model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_uniform_model.ipynb>`_

Quadratic model
---------------

The quadratic transit model (`pytransit.QuadraticModel` and `pytransit.QuadraticModelCL`) reproduces an exoplanet transit over a
stellar disk with the limb darkening modelled by a quadratic limb darkening model, as presented
in `Mandel & Agol (ApJ 580, 2001) <https://iopscience.iop.org/article/10.1086/345520/fulltext/>`_.

- `Quadratic model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_quadratic_model.ipynb>`_

**Notes:**

- The current implementation requires the minimum and maximum radius ratios to be given. **The model evaluates to unity
  for radius ratios outside these limits.**

Power-2 model
-------------

Power-2 model (`pytransit.QPower2Model` and `pytransit.QPower2ModelCL`) implements the transit model with a power-2 law
limb darkening profile presented by
`Maxted & Gill (A&A 622, A33 2019) <https://www.aanda.org/articles/aa/abs/2019/02/aa34563-18/aa34563-18.html>`_.
The model is fast to evaluate and aims to model the limb darkening accurately for *cool stars*.

- `Power-2 model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_qpower2_model.ipynb>`_

**Notes:**

- Accurate limb darkening model for cool stars.
- Fast to evaluate.

Giménez model
-------------

.. warning::

    The Giménez model is currently being rewritten from the original Fortran implementation described in Parviainen (2015) and not functional.

The Giménez model (`pytransit.GimenezModel`) implements the flexible transit model presented in
`Giménez (A&A 450, 2006) <https://www.aanda.org/articles/aa/abs/2006/18/aa4445-05/aa4445-05.html>`_. The stellar limb
darkening follows a "generic" limb darkening model, and the accuracy of limb darkening can be increased as needed.

**Notes:**

- A flexible model that can model limb darkening accurately.
- Somewhat slower to evaluate than the specialized models.
- The four-coefficient model presented in `Mandel & Agol (ApJ 580, 2001)`_ is not implemented in PyTransit since the
  Giménez model offers the same functionality with higher flexibility.

Chromosphere model
------------------

Optically thin shell model (`pytransit.ChromosphereModel` and `pytransit.ChromosphereModelCL`) by
`Schlawin et al. (ApJL 722, 2010) <https://iopscience.iop.org/article/10.1088/2041-8205/722/1/L75>`_ to model a transit
over a chromosphere.

- `Chromosphere model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_chromosphere_model.ipynb>`_