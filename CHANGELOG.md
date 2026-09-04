# Changelog

## [Unreleased]

### Added

- `LCDataGroup.plot`, which plots the light curves in a grid of subplots sharing their y limits. The panels can be
  filtered by passband, instrument, sector and transiting planet, and annotated with the instrument name, the
  passband, and the light curve's index in the group. Takes `ncols`, `figsize`, `show_index` and `show_xticks`.
- `LCData.running_median`, `LCData.outlier_mask`, `LCData.remove_outliers` and `LCDataGroup.remove_outliers` for
  weeding out outlying flux points. The points are clipped against a running median using a robust MAD estimate of
  the scatter, so that the outliers cannot inflate the threshold meant to catch them, and the removal methods
  return the number of points removed.
- `LCData.marked` and `LCDataGroup.mark_for_removal`, `unmark`, `remove_marked`, `marked` and `n_marked` for weeding
  out bad light curves interactively: plot the group, mark the bad ones by the index shown in their panels, plot
  again to check, then remove them. Marking only sets a flag; `remove_marked` modifies the group in place.
- `LCData.add_time_covariates` and `LCDataGroup.add_time_covariates`, which append the time normalised to -1 ... 1
  and its powers to the covariates, so that a linear-in-covariates baseline can absorb a polynomial trend in time.
- `LCData.linear_model`, the least-squares linear model of the flux in terms of the covariates.
- `LCDataGroup.lcslices`, a list of slices splitting an array of concatenated per-light-curve values back into
  per-light-curve arrays, matching the slices `BaseLPF` stores under the same name.
- `ncores` and `start_method` arguments to `LogPosteriorFunction.optimize_global` and `sample_mcmc`. Setting
  `ncores` creates a pool for the duration of the call and closes it afterwards, also if the run raises or is
  interrupted. The pool uses 'forkserver' or 'spawn' rather than 'fork', which is unsafe after multithreaded Numba
  code or an OpenCL context, and restricts each worker to a single Numba thread.

### Changed

- The Numba RoadRunner models (`RoadRunnerModel`, `TransmissionSpectroscopyModel`, `OblatePlanetModel`) no longer
  discretise the stellar disk into annuli. The mean intensity under the planet is computed by Gauss quadrature
  matched to the transit geometry, tabulated against the grazing parameter in two segments split at the limb
  contact, and read with a per-interval cubic. The intensity profile is tabulated once per evaluation on a fixed
  grid uniform in the square root of mu, which is what lets limb darkening models such as `LDTkLDModel` work
  unchanged. The new accuracy parameter is `nq`, the number of quadrature nodes
  (default 8), alongside `ng` (default 100); `nz`, `nzin`, `nzlimb`, `zcut`, `precompute_weights`, `klims` and `nk`
  are accepted and ignored with a `FutureWarning`. The defaults are 3-5 times more accurate than before at every
  radius ratio from 0.02 to 0.3, at about the same cost.
- The OpenCL RoadRunner model (`RoadRunnerModelCL`) uses the same quadrature, evaluated in two kernels that run
  before the flux kernel, so the host work per evaluation is unchanged. It takes the Numba model's `nq` and `ng`,
  and `init_integration(nq, ng)` replaces `init_siwft_arrays`. The error at the defaults drops from 27 to 2.5 ppm at
  a radius ratio of 0.1 and from 120 to 10 ppm at 0.3. Each passband now uses its own radius ratio, where the old
  model integrated with the mean over the passbands, invalid parameter vectors give NaN fluxes as in the Numba
  model, and a radius ratio array with a shape other than `[npv, 1]` or `[npv, npb]` raises a `ValueError`.
  `interpolate`, `klims`, `nk`, `nz`, `nzin`, `nzlimb` and `zcut` are accepted and ignored with a `FutureWarning`.
- The OpenCL RoadRunner model takes the projected star-planet separation from the same MeepMeep Taylor series
  expansion as the Numba model instead of solving the orbit itself, so the two now share the orbit. The backends
  differed by up to 2e-6 in flux at a scaled semi-major axis of 4, all of it the expansion's truncation error, which
  only the Numba model carried; a double precision build now agrees to about 7e-9 whatever the geometry. Since the
  expansion is only valid near the transit, samples outside the transit bounding box are rejected rather than
  evaluated. `RoadRunnerModelCL` now needs a MeepMeep with the OpenCL backend (`meepmeep.backends.opencl`).
- The OpenCL RoadRunner model takes a `precision` argument, `'single'` (the default, and the previous behaviour) or
  `'double'`. The kernel's floating point type is a build option, so the precision is fixed when the model is
  created, and it also sets the dtype of the returned flux. Double precision requires `cl_khr_fp64`, which is
  checked against the device, and costs 1.1 to 7.1 times the single precision evaluation time on an RTX 5070.
- The OpenCL RoadRunner model evaluates up to 3.9 times faster, and no slower anywhere, by binding the kernels once
  when the program is built and setting the kernel arguments only when they change. The fixed per-call overhead
  drops from about 570 to about 150 microseconds, which makes the model faster than the sixteen-thread Numba model
  for a single light curve of 1e4 points or more, where it used to be slower for every single-light-curve
  evaluation.
- A one-dimensional radius ratio array is interpreted consistently by the Numba and OpenCL RoadRunner models,
  following the convention already used for the limb darkening coefficients: as the radius ratios per passband for a
  single parameter vector, and as one radius ratio per parameter vector for a population. Give a population several
  radius ratios per parameter vector as an `(npv, nk)` array. An array whose leading dimension matches neither the
  number of parameter vectors nor one raises a `ValueError` naming the expected shape.
- The OpenCL models share `set_data` through a new `OpenCLTransitModel` base class instead of each carrying its own
  copy. `TransitModel.set_data` does the validation and the bookkeeping, and the base class casts the arrays to the
  types the kernels index and uploads them, so the OpenCL models now validate the light curve and passband indices
  as the Numba models do, accept `epids`, and skip the work when `set_data` is called again with the array they
  already hold. `TransitModel.set_data` returns whether it did anything, which is what lets a subclass skip the work
  it derives from the data.
- `LCDataGroup.plot` takes `show_median`, `median_width` and `nsigma` for overlaying the running median with its
  n-sigma limits, and `show_linear_model` for overlaying the linear model of the flux in terms of the covariates.
  `median_kwargs` and `linear_model_kwargs` set the line properties, `nsigma` accepts a sequence to draw one band
  per value, and the overlays are drawn on top of the flux points rather than behind them. Light curves marked for
  removal are drawn on a light gray background.
- `LCDataGroup.select` and `RVDataGroup.select` accept a sequence of values for any criterion, selecting the
  datasets matching any of them, so `lcs.select(passband=['g', 'r'])` works as expected. A sequence used to be
  compared as a single value, which silently selected nothing.
- `optimize_global` and `sample_mcmc` raise a `ValueError` if `pool` or `ncores` is combined with `vectorize=True`.
  Both `DiffEvol` and `emcee` bypass the pool for a vectorised log posterior function, so the combination used to
  run in a single process with no indication that the pool was unused.
- `optimize_global` and `sample_mcmc` attach the pool to the DE optimiser and the MCMC sampler only for the duration
  of the call. The pool used to be stored permanently, which left them holding a reference to a pool the caller had
  closed and made the `pool` argument silently ineffective on every subsequent call.
- `DiffEvol.pool` is a property that also updates the mapping function when set, so the pool can be attached and
  detached between runs.

### Fixed

- The RoadRunner-family models raised a `ZeroDivisionError` for a radius ratio that was NaN, zero, negative or above
  one, which a differential evolution population can propose. Such a parameter vector is now treated as invalid,
  like one with a bad semi-major axis or eccentricity, and evaluates to NaN fluxes in every model including the
  OpenCL one.
- Evaluating a RoadRunner model for a population with the radius ratios given as a one-dimensional array of one
  radius ratio per parameter vector, which the documentation allows, silently returned a single light curve computed
  from the first radius ratio in the Numba model and raised an `IndexError` in the OpenCL one.
- Every OpenCL model reimplements `set_data`, and all of them defaulted the exposure times to one day instead of
  zero as `TransitModel.set_data` does, so supersampling without an explicit exposure time spread the samples of one
  exposure over a whole day. `set_data(time, nsamples=10)` gave a transit up to 5 times too shallow:
  `RoadRunnerModelCL` returned a depth of 0.0021 where the Numba model gave 0.0114, and `QuadraticModelCL`,
  `QPower2ModelCL` and `UniformModelCL` were wrong by about 1e-2 in flux. A scalar exposure time or sample count was
  also stored as a zero-dimensional array rather than broadcast to one dimension.
- Importing any OpenCL model disabled PyOpenCL's `CompilerWarning` for the whole process, including for OpenCL code
  the caller builds itself, because each model silenced it with a module-level `filterwarnings`. The filter is now
  scoped to the model's own program build, and the kernels are checked to build without any compiler output at all,
  so nothing of ours is hidden by it. Two kernel sources were also read without closing the file.
- A single sample count or exposure time was left as a length-one array rather than applied to every light curve as
  documented. Every model indexes them per light curve, so all but the first read past the end: the Numba models
  raised a `ZeroDivisionError` from a garbage sample count and the OpenCL ones read out of bounds on the device.
  Both are now broadcast to the light curves, and a count that matches neither one nor the number of light curves
  raises a `ValueError`.
- `TransitModel.set_data` stored the default exposure times as an integer array, which would silently truncate an
  exposure time assigned into it afterwards. They are now floats.
- The Numba RoadRunner and oblate planet models built the quadrature nodes of the mean intensity table with the
  first passband's radius ratio for every passband, so with passband-dependent radius ratios the other passbands
  read a table built for the wrong planet size: 0.100 next to 0.114 was off by 140 ppm.
- The RoadRunner radius ratio weight table was indexed with the wrong node spacing, `(kmax - kmin) / nk` instead of
  `(kmax - kmin) / (nk - 1)`, so the interpolation was systematically misplaced and the model read one row past the
  end of the table at `kmax`.
- The small-planet profile lookup in the RoadRunner model walked off the start of the node array for a planet
  centred inside the innermost annulus.
- Evaluating a RoadRunner-family model for a population read past the ends of the eccentricity and argument of
  periastron arrays when they were left at their scalar defaults, and past the end of the zero epoch array when it
  was given as a one-dimensional vector, as documented. The results were silently wrong or raised a
  `ZeroDivisionError`. The scalars are now broadcast to the population, a one-dimensional zero epoch vector is
  treated as one zero epoch per parameter vector, and the oblate planet model also broadcasts scalar flattening and
  obliquity.
- Fixed the NumPy 2 incompatibilities in `pytransit.lpf`. The `ndarray.ptp()` calls in `BaseLPF`, `TransitAnalysis`,
  `LegendreBaseline`, `TDVLPF` and `OCLTDVLPF` are replaced with `numpy.ptp`, and the removed `numpy.int` alias is
  dropped from the `TDVLPF`, `OCLTDVLPF` and `OCLTTVLPF` imports, which made those three modules impossible to
  import.
- `BaseLPF.plot_light_curves` looked the zero epoch and the period up as `tc_1` and `p_1`, which only the
  multiplanet LPFs define, and raised a `KeyError` for a plain `BaseLPF` whose parameters are named `tc` and `p`.
  Both namings are now accepted.
- Fixed `ParameterSet` unpickling. Pickle reconstructs `list` subclasses by calling `extend` before restoring the
  instance dictionary, so the overridden `extend` failed on the missing `frozen` attribute. This made every log
  posterior function unpicklable, and any run using a pool hung indefinitely because the worker died while
  unpickling the task.
- `LogPosteriorFunction` no longer includes the DE optimiser and the MCMC sampler in its pickled state. Both hold a
  reference to an unpicklable pool while running, which made the log posterior function impossible to send to the
  pool workers.

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