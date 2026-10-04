# Changelog

## [Unreleased]

### Fixed

- The JAX backend is now usable in single precision. Two hard-coded epsilons in
  `backends/jax/rrmodel.py` sat below float32 eps (1.2e-7) and so rounded away entirely, silently
  ceasing to guard: the quadrature-node margin `1 - 1e-9` in `_ldm_nodes` and the arc-cosine clip
  `1 - 1e-12` in `_planet_angular_extent`. With the latter gone, the planet's angular extent at the
  limb-tangent node collapsed to zero, every quadrature weight in that table row vanished, and the
  row became `0/0 = NaN` -- which the split-cubic build then spread across every sample with the
  grazing parameter near 1, i.e. the whole in-window, out-of-transit band. Both margins are now
  derived from the working precision and are exactly unchanged in float64.
- The JAX backend now subtracts a reference epoch from `times` in float64 before they are cast, as
  the OpenCL backend already did. Without it a float32 ULP at BJD 2458325 is ~0.25 d, so an absolute
  timestamp collapsed a whole transit onto a couple of representable values -- an error the size of
  the transit depth, raised silently. `TransitModel.set_data` stores the shift as `tref`, and the
  JAX evaluate paths subtract it from `t0`; `tref` is `0.0` for every other backend.

- A user-supplied `LDModel` now works with `return_grad=True`. `RoadRunnerModel.evaluate` read an
  `ldigmean` attribute that is only set for the built-in limb darkening laws, so the gradient path
  raised `AttributeError`, and `LDModel` had no way to supply the gradient of the disk-integrated
  intensity. `TransmissionSpectroscopyModel` could not evaluate an `LDModel` at all: it unpacked two
  values from a call that returned three. The base class's numerical disk integral, which raised
  `IndexError` for 1D coefficients and passed all passbands' coefficients to `_evaluate` at once, is
  rewritten.

- `import pytransit` works again with MeepMeep's reorganised JAX backend. The JAX uniform-disk
  kernel still imported `solve_xy_p5_d` and `pd_t15_d` from `meepmeep.backends.jax.ts2d`, which
  MeepMeep removed in favour of autodiff, so importing the package failed on every backend. The
  kernel now builds its flux from `meepmeep.jax2d.solve2d` and `pos_c` and takes the Jacobian from
  that by forward-mode autodiff. `udmodel` keeps its `@jax.custom_jvp` rule, now fed by the autodiff
  Jacobian, so `jax.grad` still works with supersampling; `udmodel` and `udmodel_grad` keep their
  signatures and return values. The Jacobian agrees with the analytic Numba one to 1e-7 relative.
- The JAX `UniformDiskModel` used the first transit centre for every epoch when the orbit was shared
  across epochs (`include_orbit_variations=False`) and `t0` was passed per epoch. Each epoch now
  uses its own transit centre, as on the Numba and OpenCL backends.

### Changed

- **Breaking:** every transit model (`RoadRunnerModel`, `UniformDiskModel`, `TransmissionSpectroscopyModel`)
  now returns the flux **deficit** `dF = F - 1` instead of normalised flux: out-of-transit is exactly
  `0.0` and mid-transit is about `-k**2`. Observed flux is `baseline + dF`. Code that plots or fits
  `model.evaluate(...)` results directly must add back a baseline (1.0 for normalised photometry) or it
  will silently be off by 1.0. `SecondaryEclipseModel` is the deliberate exception and keeps its own
  planet-surface-brightness convention (out-of-eclipse `pi*k**2`). `TransitModel.loglikelihood` and
  `WNLogLikelihood` gained a keyword-only `baseline` parameter (default `1.0`) so the residual is
  `(obs - baseline) - dF`; their gradient output gained a matching `d lnL / d baseline` column, placed
  before the trailing `d lnL / d jitter` column.
- `RoadRunnerModel` no longer discretises the stellar disk into annuli on any backend. The mean intensity
  under the planet is computed for every radius ratio by Gauss quadrature matched to the transit geometry
  (`pytransit.backends.numba.ldquad`), tabulated against the grazing parameter in two segments split at the
  limb contact, and read with a per-interval cubic. All three backends build their tables with the same
  numba code (`ldquad.rr_tables`); the numba kernels read them directly, the OpenCL host uploads them to the
  device, and the JAX backend re-implements the build in `jax.numpy` so that it can be differentiated. The
  intensity profile is tabulated once per evaluation on a fixed grid uniform in the square root of mu. The
  accuracy parameters are `nq`, the number of quadrature nodes per segment (default 8), and `ng` (default
  100); the annulus parameters `nzin`, `nzlimb`, `zcut`, `precompute_weights`, `klims` and `nk` are removed
  from `RoadRunnerModel` (`TransmissionSpectroscopyModel` keeps its own). Against exact integration of the
  limb-darkened disk the defaults are accurate to 0.5-5 ppm for radius ratios from 0.02 to 0.3. The
  per-evaluation setup costs more than the old weight-table interpolation (about 25 us per parameter vector
  for the flux and 60 us for the gradient at the defaults), while the per-sample cost is unchanged.
- The numba and OpenCL `RoadRunnerModel` gradients now include the dependence of the mean intensity under the
  planet on the radius ratio, which the weight-table version neglected. The derivative with respect to the
  grazing parameter is the analytic derivative of the tabulated cubic. The JAX backend differentiates through
  the table build, so `jax.grad` of the JAX kernel agrees with the numba gradient.

- **Breaking:** `LDModel.__call__` takes a `return_grad` argument and returns the 4-tuple
  `(ldp, ldg, ldi, dldi)` instead of `(ldp, ldg, ldi)`; `ldg` and `dldi` are `None` unless gradients
  are requested, so a forward-only model no longer needs a `_gradient` method. A subclass that
  overrides `__call__` must return the four values. The methods now always receive the coefficients
  as `(npv, npb, nldc)`: `__call__` reads a 1D array as one parameter set for one passband and a 2D
  array as `(npb, nldc)`, as the built-in laws do. The new `_integrate_gradient` method returns the
  derivatives of the disk-integrated intensity; it and `_integrate` default to Gauss-Legendre
  quadrature in mu of `_gradient` and `_evaluate`, and can be overridden with analytic expressions.
  The quadrature replaces a trapezoidal rule in z, whose integrand has a square-root endpoint, and
  reproduces the analytic integrals of the quadratic law to 1e-12 at the default `niz`.

### Fixed

- The RoadRunner radius ratio weight table (still used by the transmission spectroscopy model) was indexed with the wrong node spacing, `(kmax - kmin) / nk` instead of `(kmax - kmin) / (nk - 1)`,
  which misplaced the interpolation between table nodes and read one row past the end of the table at `kmax`.
- A NaN radius ratio, routine from a sampler, no longer relies on undefined behaviour to yield a NaN flux. The
  numba `isnan` guards were compiled with `fastmath=True`, whose no-NaN assumption lets LLVM fold them away;
  the old kernel only returned NaN because `int(nan)` happened to wrap the table read back to element 0. The
  lookup and profile functions now keep fast math without the no-NaN and no-infinity flags, and the table
  builders check the radius ratio before building anything, so a NaN radius ratio gives NaN tables and an
  all-NaN row on every backend.

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