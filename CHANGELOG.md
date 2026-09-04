# Changelog

## [Unreleased]

### Changed

- The Numba RoadRunner models (`RoadRunnerModel`, `TransmissionSpectroscopyModel`, `OblatePlanetModel`) no longer
  discretise the stellar disk into annuli. The mean intensity under the planet is now computed for every radius
  ratio by Gauss quadrature matched to the transit geometry (Gauss-Jacobi rules for the square-root zeros of the
  planet's angular extent at its contacts, substitutions that regularise the limb, and the integration variable
  chosen per regime), tabulated against the grazing parameter in two segments split at the limb contact, and read during the
  evaluation with a per-interval cubic. The intensity profile is tabulated once per evaluation on a fixed grid
  uniform in the square root of mu, which is also what lets limb darkening models such as `LDTkLDModel` work
  unchanged. The new accuracy parameter is `nq`, the number of quadrature nodes (default 8), alongside `ng` (default
  100). Against the analytic Mandel & Agol model the defaults are 3-5 times more accurate than before at every
  radius ratio from 0.02 to 0.3, the per-evaluation setup costs about the same, and the per-sample cost rises by a
  few percent. `nz`, `nzin`, `nzlimb`, `zcut`, `precompute_weights`, `klims` and `nk` are accepted and ignored
  with a `FutureWarning`.
- The OpenCL RoadRunner model (`RoadRunnerModelCL`) uses the same quadrature. The intensity profile is tabulated on
  the host on the same fixed grid as in the Numba model, and the device integrates it over the planet's footprint,
  tabulates the mean intensity under the planet against the grazing parameter and fits the per-interval cubics in
  two kernels that run before the flux kernel, so the host work per evaluation is unchanged. The model takes the
  Numba model's `nq` and `ng` (defaults 8 and 100) and `init_integration(nq, ng)` replaces `init_siwft_arrays`;
  `interpolate`, `klims`, `nk`, `nz`, `nzin`, `nzlimb` and `zcut` are accepted and ignored with a `FutureWarning`.
  Against the analytic Mandel & Agol model the error at the defaults drops from 27 ppm to 2.5 ppm at a radius ratio
  of 0.1 and from 120 ppm to 10 ppm at 0.3, and the evaluation time is unchanged to within 0.2 ms. The radius ratio of each passband is now used for that
  passband's table (the old model integrated with the mean over the passbands), parameter vectors with a NaN
  radius ratio, a semi-major axis at or below one, or a negative eccentricity give NaN fluxes as in the Numba
  model, and a radius ratio array with a shape other than `[npv, 1]` or `[npv, npb]` raises a `ValueError`.

- The OpenCL RoadRunner model takes the projected star-planet separation from the same MeepMeep Taylor series
  expansion as the Numba model instead of solving the orbit itself, so the two now share the orbit exactly.
  MeepMeep's OpenCL backend ships its evaluators as device functions, and `meepmeep.backends.opencl.point2d.cl` is
  prepended to the model source; the coefficient solvers stay on the host, as MeepMeep's consumer contract requires,
  and `solve2d` and `bounding_box` are evaluated in a compiled loop mirroring the Numba model's precomputation. The
  difference between the two backends was up to 2e-6 in flux at a scaled semi-major axis of 4, all of it the
  expansion's truncation error, which the Numba model carried and the OpenCL model did not; a double precision build
  now agrees with the Numba model to about 7e-9 regardless of the transit geometry. Because the expansion is only
  valid near the transit, where the Keplerian solver it replaces was valid everywhere, samples outside the transit
  bounding box are now rejected rather than evaluated. `RoadRunnerModelCL` now needs a MeepMeep with the OpenCL
  backend (`meepmeep.backends.opencl`).
- The OpenCL RoadRunner model takes a `precision` argument, either `'single'` (the default, and the previous
  behaviour) or `'double'`. The kernel's floating point type is a `-DREAL=` build option, so the precision is fixed
  when the model is created, and it sets the dtype of the returned flux as well. Double precision requires
  `cl_khr_fp64`, which is checked against the device. It costs between 1.1 and 7.1 times the single precision
  evaluation time on an RTX 5070, depending on how much of the time is the fixed per-call overhead.
- The OpenCL RoadRunner model evaluates up to 3.9 times faster, and no slower anywhere, by binding the kernels once
  when the program is built rather than looking them up on every evaluation, and by setting the kernel arguments
  only when they change rather than letting PyOpenCL re-marshal all seventeen of them at every launch. The gain is
  largest for small models and vanishes for the largest ones, where the device-to-host transfer of the fluxes
  dominates everything else. Each `Program`
  attribute lookup builds a new `Kernel` and regenerates its invoker, which consults PyOpenCL's on-disk cache, so
  the three lookups were more than half the cost of a small evaluation. The fixed per-call overhead drops from about
  570 to about 150 microseconds, which makes the OpenCL model faster than the sixteen-thread Numba model for a
  single light curve of 1e4 points or more, where it used to be slower for every single-light-curve evaluation.
- A one-dimensional radius ratio array is now interpreted consistently by the Numba and OpenCL RoadRunner models,
  following the convention already used for the limb darkening coefficients: as the radius ratios per passband when
  a single parameter vector is evaluated, and as one radius ratio per parameter vector when a population is. Give a
  population several radius ratios per parameter vector as an `(npv, nk)` array. A radius ratio array whose leading
  dimension matches neither the number of parameter vectors nor one raises a `ValueError` naming the expected shape.

### Fixed

- The RoadRunner-family models raised a `ZeroDivisionError` for a radius ratio that was NaN, zero, negative or
  above one, which a differential evolution population can propose and which the annulus-based versions evaluated
  to NaN or meaningless finite fluxes. A parameter vector with a radius ratio outside (0, 1] is now treated as
  invalid, like one with a bad semi-major axis or eccentricity, and evaluates to NaN fluxes in all the models
  including the OpenCL one.
- Evaluating a RoadRunner model for a population with the radius ratios given as a one-dimensional array of one
  radius ratio per parameter vector, which the documentation allows, silently returned a single light curve computed
  from the first radius ratio in the Numba model and raised an `IndexError` in the OpenCL one. The array was read as
  a set of passband-dependent radius ratios for one parameter vector in both. It is now read as one radius ratio per
  parameter vector whenever a population is evaluated.
- `RoadRunnerModelCL.set_data` defaulted the exposure times to one day instead of zero as `TransitModel.set_data`
  does, so supersampling without an explicit exposure time spread the samples over a whole day: `set_data(time,
  nsamples=10)` gave a transit 5 times too shallow, a depth of 0.0021 where the Numba model gave 0.0114. A scalar
  exposure time or sample count was also stored as a zero-dimensional array rather than being broadcast to one
  dimension.
- The Numba RoadRunner and oblate planet models built the quadrature nodes of the mean intensity table with the
  radius ratio of the first passband for every passband, so with passband-dependent radius ratios the other
  passbands read a table built for the wrong planet size: a radius ratio of 0.100 in the second passband next to
  0.114 in the first was off by 140 ppm. The nodes are now built per passband.
- The RoadRunner radius ratio weight table was indexed with the wrong node spacing, `(kmax - kmin) / nk` instead of
  `(kmax - kmin) / (nk - 1)`, so the interpolation between table nodes was systematically misplaced: at a radius
  ratio exactly on a node the error was 3-7 times what the exact weights give, and at `kmax` the model read one row
  past the end of the table.
- The small-planet profile lookup in the RoadRunner model walked off the start of the node array for a planet
  centred inside the innermost annulus.
- Evaluating a RoadRunner-family model for a population of parameter vectors read past the ends of the eccentricity
  and argument of periastron arrays when they were left at their scalar defaults, and past the end of the zero epoch
  array when it was given as a one-dimensional vector, as documented. The results were silently wrong or, for larger
  populations, raised a `ZeroDivisionError`; this is also the likely cause of the intermittent failure of
  `test_rrmodel_batch_evaluation_matches_scalar`. The scalars are now broadcast to the population and a
  one-dimensional zero epoch vector is treated as one zero epoch per parameter vector. The oblate planet model
  additionally broadcasts scalar flattening and obliquity.

### Added

- Added `LCData.add_time_covariates` and `LCDataGroup.add_time_covariates`, which append the time normalised to
  -1 ... 1 and its powers to the existing covariates, so that a linear-in-covariates baseline can absorb a polynomial
  trend in time.
- Added `LCData.linear_model`, the least-squares linear model of the flux in terms of the covariates.
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

- `LCDataGroup.plot` takes `show_linear_model` for overlaying the least-squares linear model of the flux in terms of
  the covariates, showing how much of the variability the covariates can explain.
- `LCDataGroup.plot` takes `median_kwargs` and `linear_model_kwargs` for setting the line properties of the running
  median and the linear model overlays. Both overlays are now drawn on top of the flux points rather than behind them,
  and the n-sigma bands take their colour from `median_kwargs`.
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

- Fixed the NumPy 2 incompatibilities in `pytransit.lpf`. The `ndarray.ptp()` method calls in `BaseLPF`,
  `TransitAnalysis`, `LegendreBaseline`, `TDVLPF`, and `OCLTDVLPF` are replaced with `numpy.ptp`, and the removed
  `numpy.int` alias is dropped from the `TDVLPF`, `OCLTDVLPF`, and `OCLTTVLPF` imports, which made those three modules
  impossible to import.
- Fixed `BaseLPF.plot_light_curves` for single-planet LPFs. It looked the zero epoch and the period up as `tc_1` and
  `p_1`, which only the multiplanet LPFs define, and raised a `KeyError` for a plain `BaseLPF` whose parameters are
  named `tc` and `p`. Both namings are now accepted.
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