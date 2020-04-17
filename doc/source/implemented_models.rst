Implemented transit models
==========================

PyTransit implements a set of transit models that all share a common interface that is described in more detail in
:doc:`models`.

Uniform model
-------------

The uniform model (:class:`pytransit.UniformModel` and :class:`pytransit.UniformModelCL`) reproduces an exoplanet transit over a uniform disc.
This model is useful when modelling secondary eclipses, or when the effects from the stellar limb
darkening can be ignored.

- `Uniform model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_uniform_model.ipynb>`_

Quadratic model
---------------

The quadratic transit model (:class:`pytransit.QuadraticModel` and :class:`pytransit.QuadraticModelCL`) reproduces an exoplanet transit over a
stellar disk with the limb darkening modelled by a quadratic limb darkening model, as presented
in `Mandel & Agol (ApJ 580, 2001) <https://iopscience.iop.org/article/10.1086/345520/fulltext/>`_.

- `Quadratic model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_quadratic_model.ipynb>`_

**Notes:**

- The current implementation requires the minimum and maximum radius ratios to be given. **The model evaluates to unity
  for radius ratios outside these limits.**

Power-2 model
-------------

Power-2 model (:class:`pytransit.QPower2Model` and :class:`pytransit.QPower2ModelCL`) implements the transit model with a power-2 law
limb darkening profile presented by
`Maxted & Gill (A&A 622, A33 2019) <https://www.aanda.org/articles/aa/abs/2019/02/aa34563-18/aa34563-18.html>`_.
The model is fast to evaluate and aims to model the limb darkening accurately for *cool stars*.

- `Power-2 model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_qpower2_model.ipynb>`_

**Notes:**

- Accurate limb darkening model for cool stars.
- Fast to evaluate.

General model
-------------

The general model (:class:`pytransit.GeneralModel`) implements the flexible transit model presented by
`Giménez (A&A 450, 2006) <https://www.aanda.org/articles/aa/abs/2006/18/aa4445-05/aa4445-05.html>`_. The stellar limb
darkening follows a "general" limb darkening model, and the accuracy of limb darkening can be increased as needed.

The model is calculated using a polynomial series and both the number of polynomials `npoly` and the number of limb
darkening coefficients `nldc` can be set in the initialisation. Higher `npoly` leads to a more accurate transit model,
but also increases computation time. Increasing the number of limb darkening coefficients doesn't significantly increase
computation time, but

**Notes:**

- A flexible model that can model limb darkening accurately.
- Somewhat slower to evaluate than the specialized models.
- PyTransit implements a special "transmission spectroscopy mode" for the general model that accelerates the transit model
  evaluation significantly for transmission spectroscopy where the light curves are computed from a spectroscopic time
  series.
- The four-coefficient model presented in `Mandel & Agol (ApJ 580, 2001)`_ is not implemented in PyTransit since the
  Giménez model offers the same functionality with higher flexibility.

Chromosphere model
------------------

Optically thin shell model (:class:`pytransit.ChromosphereModel` and :class:`pytransit.ChromosphereModelCL`) by
`Schlawin et al. (ApJL 722, 2010) <https://iopscience.iop.org/article/10.1088/2041-8205/722/1/L75>`_ to model a transit
over a chromosphere.

- `Chromosphere model example <https://github.com/hpparvi/PyTransit/blob/master/notebooks/example_chromosphere_model.ipynb>`_