PyTransit
=========

Contents:

.. toctree::
   :maxdepth: 2

=====
 API
=====

.. module:: mandelagol
.. class:: MandelAgol(nldc=2, nthr=0, interpolate=False, supersampling=0, exptime=0.02, eclipse=False, klims=(0.07,0.13), nk=128, nz=256) 

  :param int nldc: Number of limb darkening coefficients.
  :param int nthr: Number of threads.
  :param bool interpolate: Use interpolated transit model.
  :param int supersampling: Number of subsamples to calculate for each light curve point.
  :param float exptime: Integration time for a single exposure, used in supersampling.

  .. method:: evaluate(t, k, u, t0, p, a, i[, e=0., w=0., c=0., update=True, interpolate_z=False])

  Evaluates the transit model given a time array, planet-star radius ratio, and orbital parameters.

  :param array_like t: Array of mid-observation times.
  :param float k: Planet-star radius ratio.
  :param array_like u: Array of limb darkening coefficients.
  :param float t0: Zero epoch.
  :param float p: Orbital period.
  :param float a: Scaled semi-major axis.
  :param float i: Orbital inclination.
  :param float e: Eccentricity.
  :param float w: Argument of periastron.
  :param array_like c: Contamination factor.
  :return: Array of model flux values for each time sample.
  :rtype: ndarray
  

  .. method:: __call__(z, k, u[, c=0])

.. module:: gimenez
.. class:: Gimenez(nldc=2, nthr=0, interpolate=False, supersampling=0, exptime=0.02, eclipse=False) 

  :param int nldc: Number of limb darkening coefficients.
  :param int nthr: Number of threads.
  :param bool interpolate: Use interpolated transit model.
  :param int supersampling: Number of subsamples to calculate for each light curve point.
  :param float exptime: Integration time for a single exposure, used in supersampling.


  .. method:: evaluate(t, k, u, t0, p, a, i[, e=0., w=0., c=0., update=True, interpolate_z=False])

  Evaluates the transit model given a time array, planet-star radius ratio, and orbital parameters.

  :param array_like t: Array of mid-observation times.
  :param float k: Planet-star radius ratio.
  :param array_like u: Array of limb darkening coefficients.
  :param float t0: Zero epoch.
  :param float p: Orbital period.
  :param float a: Scaled semi-major axis.
  :param float i: Orbital inclination.
  :param float e: Eccentricity.
  :param float w: Argument of periastron.
  :param array_like c: Contamination factor.
  :return: Array of model flux values for each time sample.
  :rtype: ndarray

  .. method:: __call__(z, k, u[, c=0])


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

