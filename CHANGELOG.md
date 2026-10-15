# Changelog

## [2.9.3] - 2026-10-15

### Changed

- Updated the RoadRunner documentation for the quadrature-based model, with new figures of the quadrature nodes and
  the mean intensity table.

## [2.9.2] - 2026-09-08

### Added

- `LCDataGroup.plot` for plotting light curves in a filterable, annotated grid.
- Robust running-median outlier removal: `LCData.outlier_mask`, `LCData.remove_outliers`, and
  `LCDataGroup.remove_outliers`.
- Interactive marking and removal of bad light curves in `LCDataGroup` (`mark_for_removal`, `remove_marked`, etc.).
- `add_time_covariates` for `LCData` and `LCDataGroup` to model polynomial time trends via the covariates.
- `LCData.linear_model` and `LCDataGroup.lcslices`.
- `ncores` and `start_method` arguments to `optimize_global` and `sample_mcmc` for automatic, fork-safe process pools.

### Changed

- The RoadRunner models now use Gauss quadrature instead of stellar disk annuli. The new accuracy parameter is `nq`;
  `nz`, `nzin`, `nzlimb`, `zcut`, `precompute_weights`, `klims`, and `nk` are deprecated and ignored.
- `RoadRunnerModelCL` uses the same quadrature and the MeepMeep orbit as the Numba model, and accepts a `precision`
  argument (`'single'` or `'double'`).
- 1D radius ratio arrays are interpreted consistently by the Numba and OpenCL RoadRunner models.
- `optimize_global` and `sample_mcmc` raise a `ValueError` when a pool is combined with `vectorize=True`, and attach
  the pool only for the duration of the call.

### Fixed

- RoadRunner models now return NaN instead of raising for invalid radius ratios.
- RoadRunner population evaluation with 1D radius ratio arrays, scalar `e` and `w`, and 1D `t0` vectors.
- RoadRunner radius ratio weight table indexing and small-planet profile lookup.
- Single sample counts and exposure times are now broadcast to all light curves.
- The OpenCL models defaulted the exposure times to one day instead of zero.
- Importing an OpenCL model no longer silences PyOpenCL compiler warnings process-wide.
- NumPy 2 incompatibilities in `pytransit.lpf`.
- `ParameterSet` and `LogPosteriorFunction` pickling, which made pool-based runs hang.

## [2.9.1] - 2026-08-24

### Added

- `LCData` and `LCDataGroup` light curve containers in `pytransit.utils.io`.
- `RVData` and `RVDataGroup` radial velocity containers in `pytransit.utils.io`.

## [2.9.0] - 2026-08-18

### Added

- `LSTSQBaseline`, a parameterless baseline fitted by linear least squares at every evaluation.
- Analytic disk-integrated intensity functions for all limb darkening models.
- `exact_areas`, `exact_ld`, and `nannuli` options to `OblatePlanetModel`.
- Exact and θ-sampled ellipse-circle intersection area routines.
- Support for non-transiting planets in `RVLPF`.

### Changed

- **Breaking**: `RVLPF` and `RVModel` require `rvis` and `is_transiting` arguments, `tc_{i}` is renamed `t0_{i}`,
  and the reference time is now `floor(min(times))`.
- The limb darkening models moved from `pytransit.models.numba.ldmodels` to `pytransit.models.limb_darkening`.
- The `RoadRunnerModel` `small_planet_limit` default changed from 0.05 to 0.01.
- Improved the RoadRunner model parallelization and thread count handling.
- Improved the `OblatePlanetModel` weight calculations and bounding box handling.

### Fixed

- Incorrect scanline ellipse-circle intersection areas for some geometries.
- `BaseLPF.remove_outliers` left the covariates inconsistent with the clipped data.
- `BaseLPF` covariate standardisation.

### Deprecated

- `pytransit.models.numba.ldmodels`, in favour of `pytransit.models.limb_darkening`.


## [2.8.1] - 2026-06-21

### Changed

- Fixed broken tests.


## [2.8.0] - 2026-06-19

### Changed

- Changed to use MeepMeep 1.0.0.

## [2.7.1] - 2026-05-07

### Added

- Added pytransit.models.new_eclise_model.EclipseModel to model secondary eclipses. 

## [2.7.0] - 2026-04-27

### Added

- Added a new EclipseSpectroscopyModel to model eclipse spectroscopy.

## [2.6.19] - 2026-03-18

### Changed

- Changed NumPy version requirement to >= 2.0.

## [2.6.18] - 2026-01-28

### Changed

- Fixed a RoadRunnerModel bug introduced in v2.6.17.

## [2.6.17] - 2025-11-27

### Changed

- Improved RoadRunnerModel transit bounding box calculation for eccentric orbits.

## [2.6.16] - 2025-10-29

### Changed

 - Nympy 2.0 compatibility fixes.

## [2.6.9] - 2025-07-22

### Changed

- Improved `TSModel` accuracy by using a first-order Taylor series expansion of the planet-star intersection area. The
  error in transit depth estimates should now be ~1 ppm for Jupiter-sized planets and below this for smaller planets.
- Fixed issues with Numpy 2.0.

## [2.6.0] - 2024-02-01

### Added

- *TSModel:* An extremely fast transit model for transmission spectroscopy. The model is based on the RoadRunnerModel and can use any 
  rotationally symmetric function to model stellar limb darkening.
- *LDTkLDM:* A RoadRunnerModel limb darkening model that uses directly stellar intensity profiles created by the LDTk. The model is
  parameterised by the effective stellar temperature, surface gravity, and metallicity. This should be quite valuable for transmission
  spectroscopy since the number of limb darkening parameters is independent of the number of passbands.

## [2.4.0] - 2020-10-14

### Added

- *EclipseModel:* an easy-to-use secondary eclipse model.

## [2.3.0] - 2020-09-16

### Added

 - *OblateStarModel:* transit model to model transits over rapidly rotating gravity-darkened stars by Barnes (2009).
   This is an initial release of the model (only a CPU version works at the moment) but I expect to 
   have it on-par with the rest of the models by v2.4.
 
### Changed

 - Renamed the *Swift* transit model introduced in v2.1 to *RoadRunner* transit model.
 - Fixed several *RoadRunner* model issues caused by the transition to calculating the
   projected distances using Taylor series expansion.
 - Changed several of the OpenCL models use the Taylor series expansion approach to calculate
   the projected distances.
 - Lots of minor bug fixes.

## [2.2.0] - 2020-09-13

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