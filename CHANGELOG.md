# Changelog

## [Unreleased]

### Added

- Added `LCData.running_median`, `LCData.outlier_mask`, `LCData.remove_outliers`, and
  `LCDataGroup.remove_outliers` for weeding out outlying flux points. The points are clipped against a running median
  computed with `scipy.signal.medfilt`, using a robust MAD estimate of the residual scatter so that the outliers cannot
  inflate the threshold meant to catch them, and the removal methods return the number of points removed.
- Added `LCData.marked` and `LCDataGroup.mark_for_removal`, `unmark`, `remove_marked`, `marked`, and `n_marked` for
  weeding out bad light curves interactively: plot the group, mark the bad light curves by the index shown in their
  panels, plot again to check, and remove them. Marking only sets a flag, and `remove_marked` modifies the group in
  place.
- Added `show_index` and `show_xticks` to `LCDataGroup.plot`. Switching the x axis ticks and labels off packs more
  light curves onto the screen when eyeballing the data.
- Added `LCDataGroup.lcslices`, a list of slices splitting an array of concatenated per-light-curve values back into
  a list of per-light-curve arrays, matching the slices `BaseLPF` stores under the same name.
- Added `LCDataGroup.plot`, a utility method that plots the light curves in a grid of subplots sharing their y limits.
  The number of columns and the figure size are given by `ncols` and `figsize`, the light curves can be filtered by
  passband, instrument, sector, and transiting planet, and each panel can be annotated with its instrument name and
  passband.
- Added `ncores` and `start_method` arguments to `LogPosteriorFunction.optimize_global` and
  `LogPosteriorFunction.sample_mcmc`. Setting `ncores` creates a multiprocessing pool for the duration of the call and
  closes it afterwards, also if the run raises or is interrupted, while a pool given via `pool` is used as-is and left
  for the caller to close. The pool is created using the 'forkserver' or 'spawn' start method rather than 'fork', which
  is unsafe after multithreaded Numba code has been run or an OpenCL context has been initialized, and each worker is
  restricted to a single Numba thread to avoid oversubscribing the machine.

### Changed

- `LCDataGroup.plot` takes `show_median`, `median_width`, and `nsigma` for overlaying the running median of the flux
  with its n-sigma limits, for spotting the points `remove_outliers` would clip. `nsigma` accepts either a single
  number or a sequence of them, in which case one band is drawn per value.
- `LCDataGroup.plot` now shows each light curve's index in the group in the upper left corner of its panel and draws
  the light curves marked for removal on a light gray background.
- `LCDataGroup.select` and `RVDataGroup.select` now accept a sequence of values for any criterion, selecting the
  datasets matching any of them, so `lcs.select(passband=['g', 'r'])` works as expected. A sequence used to be compared
  as a single value, which silently selected nothing.
- `optimize_global` and `sample_mcmc` now raise a `ValueError` if `pool` or `ncores` is combined with `vectorize=True`.
  Both `DiffEvol` and `emcee` bypass the pool when the log posterior function is vectorised, so the combination used to
  run everything in a single process without any indication that the pool was left unused.
- `optimize_global` and `sample_mcmc` now attach the pool to the DE optimiser and the MCMC sampler only for the
  duration of the call. Previously the pool was stored permanently when the optimiser or the sampler was created, which
  left them holding a reference to a pool the caller had already closed, and made the `pool` argument silently
  ineffective on all the subsequent calls.
- `DiffEvol.pool` is now a property that also updates the mapping function when set, so the pool can be attached and
  detached between the optimisation runs.

### Fixed

- Fixed `ParameterSet` unpickling. Pickle reconstructs `list` subclasses by calling `extend` before restoring the
  instance dictionary, so the overridden `extend` failed on the missing `frozen` attribute. This made every log
  posterior function unpicklable, and any run using a multiprocessing pool hung indefinitely because the worker died
  while unpickling the task.
- `LogPosteriorFunction` no longer includes the DE optimiser and the MCMC sampler in its pickled state. Both hold a
  reference to an unpicklable pool while running, which made the log posterior function impossible to send to the pool
  workers.


## [2.9.1] - 2026-08-24

### Added

- Added `pytransit.utils.io.LCData` and `LCDataGroup` light curve data containers. `LCData` holds a single light curve
  with its time, flux, uncertainty, covariate, passband, instrument, and transiting planet (`pids`) metadata, and adding
  light curves together (`lc1 + lc2`, `sum([lc1, lc2])`) gives an `LCDataGroup` that exposes the per-light-curve lists
  and arrays `BaseLPF` expects.
- Added `pytransit.utils.io.RVData` and `RVDataGroup`, the radial velocity counterparts of the light curve containers.


## [2.9.0] - 2026-08-18

### Added

- Added `pytransit.lpf.baselines.lstsqbaseline.LSTSQBaseline`, a parameterless baseline model. Rather than adding free
  coefficients to the parameter set, it fits the baseline to the relative residuals `flux_obs / flux_mod` by linear
  least squares at every evaluation.
- Added analytic disk-integrated intensity functions `ldi_nonlinear`, `ldi_general`, `ldi_square_root`,
  `ldi_logarithmic`, `ldi_exponential`, and `ldi_power_2_pm`. All limb darkening models now have one.
- Added `exact_areas`, `exact_ld`, and `nannuli` options to `OblatePlanetModel`, settable in the initializer and
  overridable per call in `evaluate`. `exact_areas` computes the planet-star intersection areas with an exact analytic
  algorithm instead of the θ-sampled scanline approximation, and `exact_ld` integrates the limb darkening over the
  planet's exact elliptical footprint.
- Added exact and θ-sampled ellipse-circle and ellipse-disk intersection area routines to
  `pytransit.models.roadrunner.ecintersection`.
- Added support for non-transiting planets in `RVLPF` via a new `is_transiting` argument.

### Changed

- **Breaking**: `RVLPF.__init__` and `RVModel.__init__` now require `rvis` and a new `is_transiting` sequence with one
  entry per planet. The zero epoch parameter `tc_{i}` is renamed `t0_{i}`, non-transiting planets are parameterised by a
  reference mean anomaly `m0_{i}` instead, and the reference time is now `floor(min(times))` rather than the mean time.
- The limb darkening models moved from `pytransit.models.numba.ldmodels` to `pytransit.models.limb_darkening`, which
  now holds one module per model and exports them all from its `__init__`. 
- Changed the `RoadRunnerModel` `small_planet_limit` default from 0.05 to 0.01. 
- Capped the RoadRunner model thread count to numba's launch-time maximum (`NUMBA_NUM_THREADS`) and set the thread count
  explicitly with `set_num_threads`. 
- Improved the RoadRunner model parallelization and updated the single-light-curve model.
- Improved the `OblatePlanetModel` weight calculations and bounding box handling.

### Fixed

- Fixed a bug in the scanline ellipse-circle intersection area routine that gave incorrect areas for some geometries,
  affecting `OblatePlanetModel` accuracy.
- Fixed `BaseLPF.remove_outliers` passing the unmasked covariates to `_init_data`, which left the covariates
  inconsistent with the clipped data.
- `BaseLPF` now standardises the covariates it stores correctly.

### Deprecated

- Deprecated `pytransit.models.numba.ldmodels` in favour of `pytransit.models.limb_darkening`.


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