PyTransit
=========

[![Travis](http://img.shields.io/travis/hpparvi/PyTransit/development.svg?style=flat)](https://travis-ci.org/hpparvi/PyTransit)
[![Licence](http://img.shields.io/badge/license-GPLv2-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![MNRAS](https://img.shields.io/badge/MNRAS-10.1093%2Fmnras%2Fstv894-blue.svg)](http://mnras.oxfordjournals.org/content/450/3/3233)
[![arXiv](http://img.shields.io/badge/arXiv-1504.07433-blue.svg?style=flat)](http://arxiv.org/abs/1504.07433)
[![ASCL](https://img.shields.io/badge/ASCL-A1505.024-blue.svg?style=flat)](http://ascl.net/1505.024)
[![DOI](https://zenodo.org/badge/5871/hpparvi/PyTransit.svg)](https://zenodo.org/badge/latestdoi/5871/hpparvi/PyTransit)

Fast and easy-to-use tools for exoplanet transit light curve modelling with Python. PyTransit offers optimised CPU and GPU implementations of popular exoplanet transit models with a unified interface. The Mandel & Agol and Gimenez models come with specialised optimisations for transmission spectroscopy that allow a transit model to be calculated in multiple passbands with only a minor additional computational cost to a single passband.

![](notebooks/model_example_1.png)

The package has been used in research since 2010, and is described in [Parviainen (2015)](http://arxiv.org/abs/1504.07433), which also details the model-specific optimisations and model performance.

## What's new in PyTransit v2.0 beta (2019)

**Freedom from Fortran**
- PyTransit v2.0 replaces all the old Fortran code with numba-accelerated Python versions!

**Mature OpenCL implementations**
- The OpenCL versions of the models are now mature, and can be swapped with the Numba-accelerated Python versions 
  without modifications.
- The OpenCL implementations evaluated in a GPU can offer 10-20 x acceleration compared to the Python versions. 
- Simultaneous model computation for a set of parameter vectors accelerates population-based sampling and optimisation methods, 
  such as *Affine Invariant Sampling (emcee)* and *Differential Evolution*.

**Two new transit models**
- Power-2 transit model by [Maxted & Gill](ArXiv:1812.01606)
- Optically thin shell model by [Schlawin et al. (ApJL 722, 75--79, 2010)](http://adsabs.harvard.edu/abs/2010ApJ...722L..75S)
  to model a transit over a chromosphere.

**Flux contamination module**
- Introduced a physics-based module to model flux contamination (blending).
- Detailed in Parviainen et al.  (a, submitted, 2019), and used in Parviainen et al. (b, in prep. 2019)

**Example notebooks**
- All (well, most of, but this'll be improved) the functionality is now documented in Jupyter notebooks available in
 [GitHub](https://github.com/hpparvi/PyTransit/tree/master/notebooks).

**Utility modules and functions**
- The `pytransit.modelling.lpf` (LogPosteriorFunction) module contains classes that can be used as a starting point for a transit analysis.

## Features

**Transit models**
  - Series-expansion based transit model by [A. Gimenez (A&A 450, 1231--1237, 2006)](http://adsabs.harvard.edu/abs/2006A&A...450.1231G).
  - Quadratic limb-darkening and uniform disk transit models by [Mandel & Agol (ApJ 580, L171–L175, 2002)](http://adsabs.harvard.edu/abs/2002ApJ...580L.171M).
  - Power-2 transit model by [Maxted & Gill](ArXiv:1812.01606)
  - Optically thin shell model by [Schlawin et al. (ApJL 722, 75--79, 2010)](http://adsabs.harvard.edu/abs/2010ApJ...722L..75S) to model narrow-band transits observations of chromospheric emission lines.

**Common features**
  - Efficient model evaluation for multicolour observations and transmission spectroscopy.
  - Built-in model interpolation for the modelling of large datasets.
  - Built-in supersampling to account for extended exposure times.

**Utilities**
  - Routines to calculate the projected planet-to-star distance for circular and eccentric orbits.
  - Routines to calculate transit durations, etc.

Installation
------------


### GitHub

Clone the repository from github and do the normal python package installation

    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit
    python setup.py install


Citing
------

If you use PyTransit in your reserach, please cite

Parviainen, H. MNRAS 450, 3233–3238 (2015) (DOI:10.1093/mnras/stv894).

or use this ready-made BibTeX entry

    @article{Parviainen2015,
      author = {Parviainen, Hannu},
      doi = {10.1093/mnras/stv894},
      journal = {MNRAS},
      number = {April},
      pages = {3233--3238},
      title = {{PYTRANSIT: fast and easy exoplanet transit modelling in PYTHON}},
      url = {http://mnras.oxfordjournals.org/cgi/doi/10.1093/mnras/stv894},
      volume = {450},
      year = {2015}
    }


Notes
-----

 - The interpolated (quadratic) Mandel & Agol model offers the best performance at the moment, but needs to be initialised with the minimum and maximum allowed radius ratio.  
 - Please use the [Issue tracker](https://github.com/hpparvi/PyTransit/issues) to report bugs and ideas for improvement.


Examples
--------
### Basics
Basic usage is simple, and the API is the same for all the models (with very minor model-specific exceptions). The transit model is first initialised and given the array containing the mid-exposure times

```Python
from pytransit import QuadraticModel

tm = QuadraticModel()
tm.set_data(times)
```

after which it can be evaluated using either a set of scalar arguments (radius-ratio `k`, zero-epoch `t0`, orbital 
period `p`, scaled semi-major axis `a`, orbital inclination `i`, eccentricity `e`, and argument of periastron `w`) and
an array of limb darkening coefficients `ldc`

```Python
flux = tm.evaluate_ps(k, ldc, t0, p, a, i, e, w)
```

or using either a parameter array

```Python
flux = tm.evaluate_pv(pv, ldc)
```

where pv is either a 1d array `[k, t0, p, a, i, e, w]` or a 2d array with a shape `(npv, 7)` where `npv` is the
number of parameter vectors to evaluate simultaneously. Now, `flux` will be either a 1d array of model values evaluated
for each mid-exposure time, or a 2d array with a shape `(npv, npt)` where `npv` is the number of parameter vectors and 
`npt` the number of mid-exposure points. In the case of a 2d parameter array, also the limb darkening coefficients 
should be given as a 2d array.

### OpenCL
The OpenCL versions of the models work identically to the Python version, except 
that the OpenCL context and queue can be given as arguments in the initialiser, and the model evaluation method can be 
told to not to copy the model from the GPU memory. If the context and queue are not given, the model creates a default 
context using `cl.create_some_context()`.

```Python
import pyopencl as cl
from src import QuadraticModelCL

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

tm = QuadraticModelCL(cl_ctx=ctx, cl_queue=queue)
```

### Supersampling
The transit models offer built-in *supersampling* for accurate modelling of long-cadence observations. The number of 
samples and the exposure time can be given when setting up the model

    tm.set_data(times, nsamples=10, exptimes=0.02)

### Heterogeneous time series

PyTransit allows for heterogeneous time series, that is, a single time series can contain several individual light curves 
(with, e.g., different time cadences and required supersampling rates) observed (possibly) in different passbands.

If a time series contains several light curves, it also needs the light curve indices for each exposure. These are given 
through `lcids` argument, which should be an array of integers. If the time series contains light curves observed in 
different passbands, the passband indices need to be given through `pbids` argument as an integer array, one per light 
curve. Supersampling can also be defined on per-light curve basis by giving the `nsamples`and `exptimes` as arrays with 
one value per light curve. 

For example, a set of three light curves, two observed in one passband and the third in another passband

    times_1 (lc = 0, pb = 0, sc) = [1, 2, 3, 4]
    times_2 (lc = 1, pb = 0, lc) = [3, 4]
    times_3 (lc = 2, pb = 1, sc) = [1, 5, 6]
    
Would be set up as

    tm.set_data(time  = [1, 2, 3, 4, 3, 4, 1, 5, 6], 
                lcids = [0, 0, 0, 0, 1, 1, 2, 2, 2], 
                pbids = [0, 0, 1],
                nsamples = [  1,  10,   1],
                exptimes = [0.1, 1.0, 0.1])
                
Further, each passband requires two limb darkening coefficients, so the limb darkening coefficient array for a single parameter set should now be

    ldc = [u1, v1, u2, v2]

where u and v are the passband-specific quadratic limb darkening model coefficients.

Author
------
  - [Hannu Parviainen](mailto:hpparvi@gmail.com), Instituto de Astrofísica de Canarias

Publications using the code
----------------------------
  - Parviainen, H. et al. "The GTC exoplanet transit spectroscopy survey II: An overly-large Rayleigh-like feature for exoplanet TrES-3b." A&A 585, 1–12 (2016).
  - Parviainen, H. et al. "Exoplanet Transmission Spectroscopy using KMOS." MNRAS 4, 3875–3885 (2015).
  - Gandolfi, D. et al. "Kepler-423b: a half-Jupiter mass planet transiting a very old solar-like star." A&A 576, A11 (2015).
  - Tingley, Brandon, H. Parviainen, et al. "Confirmation of an exoplanet using the transit color signature: Kepler-418b, a blended giant planet in a multiplanet system." A&A 567, (2014).
  - Gandolfi, Davide, H. Parviainen, et al. "Kepler-423b: a half-Jupiter mass planet transiting a very old solar-like star." A&A 576, A11 (2015).
  - Parviainen, H. et al. "Transiting exoplanets from the CoRoT space mission." A&A 562, A140 (2014).
  - Gandolfi, Davide, H. Parviainen, et al. "Kepler-77b: a very low albedo, Saturn-mass transiting planet around a metal-rich solar-like star." A&A 557, A74 (2013).
  - Parviainen, Hannu, H.J. Deeg, and J.A. Belmonte. "Secondary Eclipses in the CoRoT Light Curves: A Homogeneous Search Based in Bayesian Model Selection." A&A (2012)
  - Rouan, D., H. Parviainen, C. Moutou, Magali Deleuil, M. Fridlund, A Ofir, M. Havel, et al. "Transiting Exoplanets from the CoRoT Space Mission." A&A 537 (January 9, 2012): A54.
  - Murgas, F., E. Pallé, A. Cabrera-Lavers, K. D. Colón, E. L. Martín, and H. Parviainen. "Narrow Band H α Photometry of the super-Earth GJ 1214b with GTC/OSIRIS Tunable Filters." A&A 544 (July 24, 2012): A41.
  - Tingley, Brandon, E. Palle, H. Parviainen, H. J. Deeg, M. R. Zapatero Osorio, A. Cabrera-Lavers, J. a. Belmonte, P. M. Rodriguez, F. Murgas, and I. Ribas. "Detection of Transit Timing Variations in Excess of One Hour in the Kepler Multi-planet Candidate System KOI 806 with the GTC." A&A 536 (December 12, 2011): L9.
