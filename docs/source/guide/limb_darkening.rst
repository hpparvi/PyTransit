Limb darkening
==============

A star is not a uniformly bright disk. Looking at its centre we see deep, hot layers; looking near
the limb our line of sight passes obliquely through the atmosphere and reaches only shallower,
cooler layers. The disk is therefore brightest at the centre and dimmest at the edge, an effect
called *limb darkening*.

Limb darkening sets the shape of a transit. A planet crossing a uniform disk would produce a
flat-bottomed box; a planet crossing a limb-darkened disk produces the familiar rounded profile,
because it blocks more light near mid-transit than near ingress and egress. Getting limb darkening
wrong biases the radius ratio, so it matters for anything measuring a transit depth.

The intensity profile is written as a function of

.. math::

    \mu = \cos\gamma = \sqrt{1 - z^2},

where :math:`\gamma` is the angle between the surface normal and the line of sight and :math:`z`
is the distance from the disk centre in stellar radii. The disk centre is :math:`\mu = 1` and the
limb is :math:`\mu = 0`.

Which models support which laws
-------------------------------

The classical models are analytic solutions derived for one specific law and cannot use any other:

.. list-table::
    :header-rows: 1
    :widths: 44 56

    * - Model
      - Limb darkening
    * - :class:`~pytransit.models.ma_uniform.UniformModel`
      - None (uniform disk)
    * - :class:`~pytransit.models.ma_quadratic.QuadraticModel`
      - Quadratic only
    * - :class:`~pytransit.models.qpower2.QPower2Model`
      - Power-2 only
    * - :class:`~pytransit.models.general.GeneralModel`
      - General law, any number of coefficients
    * - :class:`~pytransit.models.ma_chromosphere.ChromosphereModel`
      - None (optically thin shell)
    * - :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel`
      - **Any radially symmetric profile**

RoadRunner is the exception: it tabulates the intensity profile numerically rather than solving
the integral analytically, so the law is a free choice with essentially no cost in speed or
accuracy. This is the main reason to prefer it.

Built-in laws
-------------

RoadRunner and its subclasses take the law by name in the initialiser

.. code-block:: python

    from pytransit import RoadRunnerModel

    tm = RoadRunnerModel('power-2')

The available names are listed in ``RoadRunnerModel.ldmodels``.

.. list-table::
    :header-rows: 1
    :widths: 18 8 74

    * - Name
      - ``nldc``
      - Profile
    * - ``'uniform'``
      - 0
      - :math:`I(\mu) = 1`
    * - ``'linear'``
      - 1
      - :math:`I(\mu) = 1 - u(1-\mu)`
    * - ``'quadratic'``
      - 2
      - :math:`I(\mu) = 1 - u(1-\mu) - v(1-\mu)^2`
    * - ``'quadratic-tri'``
      - 2
      - Quadratic in the triangular sampling parametrisation :math:`(q_1, q_2)`
    * - ``'square_root'``
      - 2
      - :math:`I(\mu) = 1 - u(1-\mu) - v(1-\sqrt{\mu})`
    * - ``'logarithmic'``
      - 2
      - :math:`I(\mu) = 1 - u(1-\mu) - v\,\mu\ln\mu`
    * - ``'exponential'``
      - 2
      - :math:`I(\mu) = 1 - u(1-\mu) - v/(1-e^{\mu})`
    * - ``'power-2'``
      - 2
      - :math:`I(\mu) = 1 - c(1-\mu^\alpha)`
    * - ``'power-2-pm'``
      - 2
      - Power-2 in the :math:`(h_1, h_2)` parametrisation
    * - ``'nonlinear'``
      - 4
      - :math:`I(\mu) = 1 - \sum_{n=1}^{4} c_n (1 - \mu^{n/2})`
    * - ``'general'``
      - any
      - :math:`I(\mu) = 1 - \sum_{n=1}^{N} u_n (1 - \mu^n)`

Choosing a law
--------------

**Quadratic** is the default and the right starting point for optical photometry of solar-type
stars. It is what almost everyone uses, which also makes results directly comparable to the
literature.

**Power-2** describes cool stars, and M dwarfs in particular, considerably better than the
quadratic law with the same two coefficients, and the difference grows towards the infrared. Prefer
it for late-type hosts and for near-infrared data.

**Non-linear** is the standard form for tabulated theoretical coefficients. Its four coefficients
are strongly correlated, so it is a good law to *fix* to model predictions and a poor one to *fit*.

**Uniform** is not a limb darkening law but its absence. Use it for secondary eclipses and for
speed-critical work where transit shape does not matter.

Parametrisations for fitting
----------------------------

Two of the laws are reparametrisations of others, designed to behave better as free parameters.

``'quadratic-tri'`` implements the triangular sampling of Kipping (MNRAS 435, 2152, 2013):

.. math::

    u = 2\sqrt{q_1}\, q_2, \qquad v = \sqrt{q_1}\,(1 - 2 q_2).

A uniform prior on :math:`(q_1, q_2) \in [0,1]^2` maps to a uniform prior over exactly the region
of :math:`(u,v)` space that gives a physically valid profile -- everywhere positive and
monotonically decreasing outwards. Sampling :math:`u` and :math:`v` directly with uniform priors
instead wastes the sampler's time in unphysical regions and imposes a prior nobody intended.

``'power-2-pm'`` implements the parametrisation of Maxted (A&A 622, A33, 2018), using the intensity
at two fixed points on the disk,

.. math::

    h_1 = I(\mu = 1/2), \qquad h_2 = h_1 - I(\mu = 1/\sqrt{2}),

which are far less correlated than :math:`c` and :math:`\alpha` and therefore much easier to
sample.

Use these when fitting, and the plain parametrisations when reporting or when reproducing another
analysis.

Coefficient layout
------------------

The coefficients are passed to `evaluate` through `ldc`, ordered by passband and then by
coefficient. For two passbands with a quadratic law

.. code-block:: python

    ldc = [[u1, v1],
           [u2, v2]]        # or, flattened, [u1, v1, u2, v2]

The accepted shapes are described in :doc:`evaluation`.

Custom limb darkening
---------------------

RoadRunner accepts three things besides a name.

**A callable.** Any function ``f(mu, pv) -> ndarray`` that returns the intensity profile. It must
be Numba-compilable, so decorate it with ``@njit``.

.. code-block:: python

    from numba import njit
    from pytransit import RoadRunnerModel

    @njit
    def ld_custom(mu, pv):
        return 1.0 - pv[0]*(1.0 - mu**pv[1]) - pv[2]*(1.0 - mu)**2

    tm = RoadRunnerModel(ld_custom)

Given only a profile, the model integrates it numerically over the stellar disk to get the mean
stellar intensity. This works but costs time on every evaluation.

**A (profile, integral) pair.** If the disk integral

.. math::

    I_\star = 2\pi \int_0^1 I(\mu)\, z\, dz

has a closed form, supply it as a second function ``g(pv) -> float`` and the model uses it instead
of integrating numerically.

.. code-block:: python

    @njit
    def ldi_custom(pv):
        ...

    tm = RoadRunnerModel((ld_custom, ldi_custom))

This is how all the built-in laws are defined: each is a ``(ld_*, ldi_*)`` pair, listed in
:doc:`../api/limb_darkening`.

**An LDModel instance.** :class:`~pytransit.models.ldmodel.LDModel` is the interface for limb
darkening profiles that are not analytic laws at all but come from a stellar atmosphere model. A
subclass implements ``__call__(mu, x)`` returning the profile and its disk integral, where ``x``
holds whatever parameters the underlying model takes.

LDTk
----

:class:`~pytransit.models.ldtkldm.LDTkLDModel` is an `LDModel` backed by
`LDTk <https://github.com/hpparvi/ldtk>`_, which computes limb darkening profiles from the
Husser et al. (2013) PHOENIX specific intensity spectra for a given set of passbands and stellar
parameters.

.. code-block:: python

    from pytransit import RoadRunnerModel
    from pytransit.models.ldtkldm import LDTkLDModel
    from pytransit.contamination import sdss_g, sdss_r, sdss_i, sdss_z

    ldm = LDTkLDModel(pbs=(sdss_g, sdss_r, sdss_i, sdss_z),
                      teff=(5500, 100), logg=(4.5, 0.1), metal=(0.0, 0.1))

    tm = RoadRunnerModel(ldm)

The profile then follows the stellar atmosphere models directly, with no analytic law in between,
and its free parameters are the stellar parameters rather than abstract coefficients. This is the
most physically motivated option, at the cost of depending on the accuracy of the model
atmospheres.

.. note::

    LDTk is an optional dependency. The import in ``pytransit/__init__.py`` is guarded, so
    `LDTkLDModel` is simply unavailable if LDTk is not installed.

.. seealso::

    :doc:`../api/limb_darkening` for the full list of profile functions and their signatures.
