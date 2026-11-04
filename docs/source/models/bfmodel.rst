Brute-force model
=================

:class:`~pytransit.models.bfmodel.BFModel` computes a transit by integrating the stellar brightness
numerically over the exact region the planet covers. It makes no approximation beyond the
integration tolerance, which makes it a reference for testing the other models, but it is far too
slow to fit data with.

What it can model
-----------------

The model separates the shapes of the planet and the star from the brightness of the star.

**The planet** is an ellipse with the semi-major axis ``k``, flattening ``f``, and orientation
``alpha``, as in :doc:`OblatePlanetModel <opmodel>`. With ``f = 0`` it is the usual circular planet.

**The stellar disk** is an ellipse with an equatorial radius of one, flattening ``fstar``, and
orientation ``astar``. Both angles are measured from the sky-plane x-axis, the direction of the
planet's motion at mid-transit, to the ellipse's semi-major axis.

**The stellar brightness** is given in one of two ways:

- as a radial profile, either the name of one of RoadRunner's limb darkening laws or a callable
  ``f(mu, pv)`` with the same convention as RoadRunner's custom profiles. For an oblate star,
  ``mu = sqrt(1 - r**2)``, where `r` is the elliptical radius: the distance from the centre in units
  of the disk's radius in that direction;
- as a callable ``g(x, y, pv)`` that returns the brightness at the sky-plane coordinates
  :math:`(x, y)`, in units of the stellar equatorial radius. This describes any brightness
  distribution, such as that of a gravity-darkened star or a spotted one.

In both cases ``pv`` is the ``ldc`` argument of `evaluate`.

**The orbit** comes from an exact solution of Kepler's equation by default, so the model can also
test the orbit approximations of the other models. With ``orbit='taylor'`` it uses the same Taylor
series expansion around mid-transit as RoadRunner and the models built on it, which isolates the
error of a model's flux integration from that of its orbit.

How it integrates
-----------------

The region the planet covers is cut into horizontal scanlines. On each scanline, the planet and the
star cover an interval whose ends follow analytically from the two ellipses, and the brightness is
integrated over their intersection with adaptive Gauss-Kronrod quadrature. The integrals along the
scanlines are integrated in turn over the planet's height.

Two details keep the integration accurate:

- The outer integral is split where the planet's outline crosses the stellar limb. There the ends
  of the scanline interval switch from following one outline to following the other, and the
  integrand has a corner that adaptive quadrature would otherwise converge on slowly.
- Along each scanline, the position is written as :math:`x = m + w \sin\phi`, where :math:`m` and
  :math:`w` are the midpoint and half-width of the stellar disk's chord. This gives :math:`\mu`
  without the loss of precision of :math:`\sqrt{1 - r^2}` near the limb, and removes the
  inverse square root singularity of a profile that diverges at the limb, such as that of an
  optically thin shell.

The stellar flux that normalises the transit is integrated in the same way for a brightness
function, and as a one-dimensional integral for a radial profile.

Accuracy and speed
------------------

Against independent references -- the exact lens area for a uniform star, the exact ellipse-circle
intersection area for a uniform oblate star, and one-dimensional integrals over stellar rings for
radial profiles -- the model agrees to :math:`10^{-13}` or better with its default tolerances,
including for a profile that diverges at the limb. By comparison, the analytic quadratic and
chromosphere models in PyTransit are accurate to between :math:`10^{-9}` and :math:`10^{-7}`.

An exposure in transit takes about 10 to 30 ms, depending on the brightness description, while
exposures where the planet does not overlap the star cost almost nothing. A light curve of a few
dozen exposures across a transit is therefore enough for most tests.

The tolerances can be changed with the ``epsabs`` and ``epsrel`` arguments.

Testing a model
---------------

Comparing RoadRunner with the brute-force model on the same orbit, so that the difference is
RoadRunner's flux integration error alone:

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import BFModel, RoadRunnerModel

    time = linspace(-0.12, 0.12, 49)
    pv = dict(k=0.1, ldc=[0.4, 0.25], t0=0.0, p=4.0, a=13.0, i=0.49 * pi)

    reference = BFModel('quadratic', orbit='taylor')
    reference.set_data(time)

    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)

    error = abs(tm.evaluate(**pv) - reference.evaluate(**pv)).max()

``evaluate_xy`` computes the flux for given planet positions, which tests a model's flux
integration without any orbit at all. A brightness function describes a star the radial profiles
cannot, here a flattened star whose brightness falls linearly from the poles to the equator:

.. code-block:: python

    def brightness(x, y, pv):
        return 1.0 - pv[0] * (1.0 - abs(y) / 0.9)

    tm = BFModel(brightness=brightness)
    flux = tm.evaluate_xy(0.3, 0.5, k=0.1, ldc=[0.2], fstar=0.1)

API
---

.. autoclass:: pytransit.models.bfmodel.BFModel
    :members: evaluate, evaluate_xy
    :special-members: __init__
