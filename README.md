PyTransit
=========

[![Travis](http://img.shields.io/travis/hpparvi/PyTransit/development.svg?style=flat)](https://travis-ci.org/hpparvi/PyTransit)
[![Licence](http://img.shields.io/badge/license-GPLv2-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![MNRAS](https://img.shields.io/badge/MNRAS-10.1093%2Fmnras%2Fstv894-blue.svg)](http://mnras.oxfordjournals.org/content/450/3/3233)
[![arXiv](http://img.shields.io/badge/arXiv-1504.07433-blue.svg?style=flat)](http://arxiv.org/abs/1504.07433)
[![ASCL](https://img.shields.io/badge/ASCL-A1505.024-blue.svg?style=flat)](http://ascl.net/1505.024)
[![DOI](https://zenodo.org/badge/5871/hpparvi/PyTransit.svg)](https://zenodo.org/badge/latestdoi/5871/hpparvi/PyTransit)

Fast and easy-to-use tools for exoplanet transit light curve modelling with Python. PyTransit implements the quadratic Mandel & Agol and the Gimenéz transit models with various optimisations, and offers both a simple interface for model evaluation and a lower-level access for fine-tuning the model.   

```Python
from pytransit import MandelAgol
m = MandelAgol()
f = m.evaluate(t, *pv)
```

![](notebooks/model_example_1.png)


The package is described in [Parviainen (2015)](http://arxiv.org/abs/1504.07433). Also, take a look at the [Bayesian parameter estimation tutorial](http://nbviewer.ipython.org/github/hpparvi/exo_tutorials/blob/master/01_broadband_parameter_estimation.ipynb) for an example on how to use the model in a basic exoplanet transit modelling situation.

Modules
-------
**Transit models**
  - Series-expansion based transit model by [A. Gimenez (A&A 450, 1231--1237, 2006)](http://adsabs.harvard.edu/abs/2006A&A...450.1231G).
  - Quadratic limb-darkening transit model by [Mandel & Agol (ApJ 580, L171–L175, 2002)](http://adsabs.harvard.edu/abs/2002ApJ...580L.171M).
 
  - Common features
    - Optimized and parallelized Fortran implementations. 
    - Efficient model evaluation for multicolour observations and transmission spectroscopy.
    - Built-in model interpolation for the modelling of large datasets.
    - Built-in supersampling to account for extended exposure times.

**Utilities**
  - Routines to calculate the projected planet-to-star distance for circular and eccentric orbits.
  - Routines to calculate transit durations, etc.

Installation
------------

First clone the repository from github

    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit

and then do the normal python package build & installation. 

#### Intel & AMD

    python setup.py config_fc --fcompiler=gnu95 --opt="-Ofast" --f90flags="-cpp -fopenmp -march=native" build
    python setup.py install --user

#### Mac

    python setup.py config_fc --fcompiler=gnu95 --opt="-Ofast" --f90flags="-cpp -fopenmp -march=native -mno-avx" build
    python setup.py install --user

The code has been tested with gfortran and Intel compilers, but it should compile with others as well (if it doesn't, please let me know).

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
Basic usage is simple:

```Python
from pytransit import MandelAgol

m = MandelAgol()
f = m.evaluate(t, *pv)
```
or

```Python
from pytransit import MandelAgol

m = MandelAgol(interpolate=True, klims=(0.10,0.13))
f = m.evaluate(t, *pv)
```

or
```Python
from pytransit import Gimenez

m = Gimenez()
f = m.evaluate(t, *pv)
```

Here we first initialize the model accepting the defaults (quadratic limb darkening law, no supersampling, 
and the use of all available cores), and then calculate the model for times in the time array `t`, `pv` being 
a list containing the system parameters.

For a slightly more useful example, we can do:
```Python
import numpy as np
from pytransit import MandelAgol

t = np.linspace(0.8,1.2,500)
k, t0, p, a, i, e, w = 0.1, 1.01, 4, 8, 0.48*np.pi, 0.2, 0.5*np.pi
u = [0.25,0.10]

m = MandelAgol()
f = m.evaluate(t, k, u, t0, p, a, i, e, w)
```
where `k` is the planet-star radius ratio, `t0` the transit center, `p` the orbital period, `a` the scaled
semi-major axis, `i` the orbital inclination, `e` the orbital eccentricity, `w` the argument of periastron,
and `u` contains the quadratic limb darkening coefficients.

### Multiple limb darkening coefficient sets

The model can also be evaluated for several limb darkening coefficient (ldc) sets simultaneously (much faster than
evaluating the model several times for different coefficient sets):

    ...
    u = [[0.25, 0.1],[0.35,0.2],[0.45,0.3],[0.55,0.4]]

    m = MandelAgol()
    f = m.evaluate(t, k, u, t0, p, a, i, e, w)
    
In this case, the model returns several light curve models, each corresponding to a single ldc set.

### Supersampling
The transit model offers built-in *supersampling* for transit fitting to transit photometry with poor time 
sampling (such as *Kepler*'s long cadence data):

    m = MandelAgol(supersampling=8, exptime=0.02)
    ...

### Tweaking the Gimenéz model
The Gimenéz model accuracy and the number of limb darkening coefficients can be set in the initialization. 
Finally, for fitting to large datasets, the model can be evaluated using interpolation. 

Basic transit model usage with linear limb darkening law, lower accuracy, and four cores:

    m = Gimenez(npol=50, nldc=1, nthr=4)
    ...
      
Transit model using linear interpolation:

    m = Gimenez(interpolate=True)
    ...


### Advanced

Calculate projected distance for a circular or eccentric orbit given time t, transit center time t0, period p, 
scaled semi-major axis a, inclination i, eccentricity e, and argument of periastron w:

    import numpy as np
    from pytransit.orbits_f import orbits as of

    t0, p, a, i, e, w = 1.01, 4, 8, 0.48*np.pi, 0.2, 0.5*np.pi

    t   = np.linspace(0.8,1.2,500)
    zc  = of.z_circular( t, t0, p, a, i, nthreads=0)                  
    zen = of.z_eccentric_newton(t, t0, p, a, i, e, w, nthreads=0) # Calculated using Newton's method
    ze3 = of.z_eccentric_s3(t, t0, p, a, i, e, w, nthreads=0)     # Calculated using series expansion (ok for e<0.15)
    ze5 = of.z_eccentric_s(t, t0, p, a, i, e, w, nthreads=0)      # Calculated using series expansion (ok for e<0.25)
    zel = of.z_eccentric_ip(t, t0, p, a, i, e, w, nthreads=0, update=False) # Faster for large LCs, uses linear interpolation

Transit model using linear interpolation, two different sets of z:

    m  = Gimenez(interpolate=True)      # Initialize the model
    I1 = m(z1,k,u)               # Evaluate the model for z1, update the interpolation table
    I2 = m(z2,k,u, update=False) # Evaluate the model for z2, don't update the interpolation table
    
Author
------
  - Hannu Parviainen <hpparvi@gmail.com>, University of Oxford

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
