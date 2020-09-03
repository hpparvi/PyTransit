# Changelog

## [2.2.0]

PyTransit version 2.2 now calculates the normalized planet-star distances using a Taylor series expansion
of the x, and y positions in the sky plane (Parviainen and Korth, 2020, submitted to MNRAS). This gives a 
significant speed boost in transit model evaluation that is especially noticeable for eccentric orbits. 

## [2.1.0] - 2020-07-07

PyTransit version 2.1 adds a new transit model named *swift* that can use any Python callable to model the stellar
limb darkening while still giving equal or better performance than the analytical quadratic transit
model.

### Added

- *Swift* transit model (Parviainen, submitted) to allow fast and flexible transit modelling with
  any radially symmetric limb darkening model.

## [2.0.0] - 2020-07-07

PyTransit Version 2 removes the Fortran dependencies in v1 by implementing all the transit models
in *numba*-accelerated Python. Version 2 also adds a number of new transit models, and implements
most models both in CPU and GPU versions.

### Added
- New API that is consistent across all transit models.

### Changed
- Nearly everything.

## [1.0.0 beta] - 2014-09-02

PyTransit 1.0 implements the quadratic model by Mandel & Agol and the general model by 
Giménez with special optimisation that significantly improve the model evaluation speed.

## Early history - 2010

The first version of PyTransit saw light in 2010. It implemented the transit model for general
limb darkening law by Giménez and was used in 20-30 papers before the public release of v1.0.