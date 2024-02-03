PyTransit
=========

[![Licence](http://img.shields.io/badge/license-GPLv2-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![MNRAS](https://img.shields.io/badge/MNRAS-10.1093%2Fmnras%2Fstv894-blue.svg)](http://mnras.oxfordjournals.org/content/450/3/3233)
[![arXiv](http://img.shields.io/badge/arXiv-1504.07433-blue.svg?style=flat)](http://arxiv.org/abs/1504.07433)
[![ASCL](https://img.shields.io/badge/ASCL-A1505.024-blue.svg?style=flat)](http://ascl.net/1505.024)
[![DOI](https://zenodo.org/badge/5871/hpparvi/PyTransit.svg)](https://zenodo.org/badge/latestdoi/5871/hpparvi/PyTransit)

Fast and easy-to-use tools for exoplanet transit light curve modelling with Python. PyTransit offers optimised CPU and 
GPU implementations of popular exoplanet transit models with a unified interface, and thrives to be *the fastest and the 
most versatile* tool for transit modelling in Python. 

PyTransit makes transit model evaluation trivial whether modelling straightforward single-passband transit light curves or
more complex science-cases, such as transmission spectroscopy. Further, the model can be evaluated for a large set of
parameter sets simultaneously in parallel to optimize the evaluation speed with population-based MCMC samplers and 
optimization methods, such as emcee and  DE-MCMC.

```Python
from pytransit import RoadRunnerModel

tm = RoadRunnerModel('quadratic')
tm.set_data(times)

tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

tm.evaluate(k=[0.10, 0.12], ldc=[[0.2, 0.1, 0.5, 0.1]], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

tm.evaluate(k=[[0.10, 0.12], [0.11, 0.13]], ldc=[[0.2, 0.1, 0.5, 0.1],[0.4, 0.2, 0.75, 0.1]],
            t0=[0.0, 0.01], p=[1, 1], a=[3.0, 2.9], i=[.5*pi, .5*pi])
```

![](doc/source/basic_example_1.svg)
![](doc/source/basic_example_2.svg)
![](doc/source/basic_example_3.svg)

The package has been used in research since 2010, and is described in [Parviainen (2015)](http://arxiv.org/abs/1504.07433), 
[Parviainen (2020a)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1633P/abstract),
and [Parviainen (2020b)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P/abstract).

  
## Examples and tutorials

### RoadRunner transit model

RoadRunner [(Parviainen, 2020a)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1633P/abstract) is a fast exoplanet transit model that can use any radially symmetric function to model stellar limb darkening 
while still being faster to evaluate than the analytical transit model for quadratic limb darkening.

- [RRModel example 1](https://github.com/hpparvi/PyTransit/blob/master/notebooks/roadrunner/roadrunner_model_example_1.ipynb) 
  shows how to use RoadRunner with the included limb darkening models.
- [RRModel example 2](https://github.com/hpparvi/PyTransit/blob/master/notebooks/roadrunner/roadrunner_model_example_2.ipynb)
  shows how to use RoadRunner with your own limb darkening model.
- [RRModel example 3](https://github.com/hpparvi/PyTransit/blob/master/notebooks/roadrunner/roadrunner_model_example_3.ipynb) 
  shows how to use an LDTk-based limb darkening model LDTkM with RoadRunner.

### Transmission spectroscopy transit model

Transmission spectroscopy transit model (TSModel) is a special version of the RoadRunner model dedicated to modelling 
transmission spectrum light curves. 
 
 - [TSModel Example 1](https://github.com/hpparvi/PyTransit/blob/dev/notebooks/roadrunner/tsmodel_example_1.ipynb)


## Documentation

Read the docs at [pytransit.readthedocs.io](https://pytransit.readthedocs.io).

Installation
------------
### PyPI

The easiest way to install PyTransit is by using `pip`

    pip install pytransit

### GitHub

Clone the repository from github and do the normal python package installation

    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit
    pip install .


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

Author
------
  - [Hannu Parviainen](mailto:hpparvi@gmail.com), Instituto de Astrofísica de Canarias
