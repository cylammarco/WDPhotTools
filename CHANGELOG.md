# Changelog

All notable changes to WDPhotTools are documented in this file.

## [Unreleased] - 2026-07-07

### New Features
- Added canonical photometry API support in `WDfitter.fit`:
  - `photometry`
  - `photometry_errors`
  - `photometry_space`
- Added fitting in flux space alongside magnitude space.
- Added `self.best_fit_photometry` in the selected `photometry_space`.

### API Changes (Breaking)
- Removed legacy fit inputs:
  - `mags`, `mag_errors`
  - `fluxes`, `flux_errors`
- Standardized `self.fitting_params` to:
  - `photometry`
  - `photometry_errors`
  - `photometry_space`
- Removed legacy output aliases:
  - `best_fit_mag`
  - `best_fit_flux`

### Logging
- Replaced touched `print` calls with structured `logging`.
- Added INFO/WARNING logs at fitter orchestration boundaries.

### Documentation
- Updated README/examples/docstrings to canonical photometry API.
- Updated RTD config/build and migration guidance.

### Tests
- Updated fitter tests to canonical API calls.
- Added deterministic flux-vs-magnitude parity tests.
- Added deterministic regression checks against `origin/main`.
- Reduced runtime for the new parity/regression matrix.

## [v0.0.13] - 2026-01-14

### Release
- Versioned release `v0.0.13` (tag commit `e81da92`).

## [v0.0.12] - 2025-12-27

### Fixes
- Fixed conversion constant usage in fitter-related numerical paths.
- Ensured `_integrand` returns `float` for improved compatibility.

### Compatibility
- Improved compatibility posture for NumPy 1/2 transition.
- Marked Windows+Python 3.13/3.14 CI as allowed failures during transition.

### CI and Tooling
- Updated GitHub Actions workflow and checkout handling.
- Updated setup metadata and test-pypi workflow in `dev`.
- General housekeeping before release.

## [v0.0.11] - 2025-09-22

### Performance
- Significant performance improvement in WDLF computation.
- Refactored basis interpolation/integration helper paths.

### Fixes
- Fixed wrong-array update bug and additional minor defects.
- Cleaned redundant `.keys()` usage.

### Dependency and Packaging
- Replaced `pkg_resources` with `importlib`.

### Documentation and Tooling
- Updated RTD configuration/docs.
- Updated formatting and CI setup.

### Tests
- Updated tests to use `Agg` backend where needed.
- Fixed test filename issues.

## [v0.0.10] - 2025-05-13

### New Features
- Added support for user-provided priors in fitting workflows.

### Validation and Fixes
- Added/updated tests for prior handling.
- Fixed missing `z_min`/`z_max` handling in reddening cases.
- Decoupled objective functions from fitter orchestration.
- Avoided interpolation-grid breakage in log-scale age grids.

### CI and Tooling
- Updated test environments and weekly build checks.
- Updated pre-commit line-length configuration to 120 chars.
- Multiple housekeeping/style/markdown fixes.

## [v0.0.9] - 2023-08-11

### Fixes
- Fixed example script syntax issues.
- Fixed interpolation variable-allowlist omissions.
- Fixed plotting type/element-wise comparison edge cases.
- Added `> 0` safety check in density computation paths.

## [v0.0.8] - 2023-05-07

### New Features
- Added/validated fitter mass-estimation test coverage.

### Fixes
- Fixed fitting-mass bug.
- Fixed independent-variable name comparison robustness.
- Migrated from deprecated `interp2d` to RBF interpolation.
- Standardized fitter method outputs to scalar floats.

### Documentation
- Updated README/RTD content and citation references.
- Added clarifications for uncertainty/error estimation behavior.

## [v0.0.7] - 2022-12-04

### New Features
- Stored `number_density` as object properties before CSV export.

### Fixes
- Fixed sign bug in `dL/dt` number-density computation.
- Added protections for log-scale y-axis with non-positive data.
- Fixed negative-zero and additional plotting/cleanup issues.
- Corrected default `Rv` from `0.0` to `3.1`.

### CI and Tooling
- Added Python 3.11 in tests.
- Added code analysis workflow and broad tidying.

## [v0.0.6] - 2022-10-24

### New Features
- Added linear fractional extinction model support.
- Added RA/Dec-aware fitter updates for extinction workflows.
- Extended reddening API for linearly interpolated extinction.

### Compatibility
- Lowered minimum `astropy` version to support Python 3.7.
- Added Python 3.10 and Windows CI/test coverage.

### Tests
- Extended tests for new reddening/extinction paths.
- Reduced MCMC test walkers/steps/burn-in and relaxed tolerance for stability.

### Fixes
- Fixed dependent-variable distance handling edge cases.

## [v0.0.5] - 2022-09-12

### Refactor
- Moved package code to `src/` layout.
- Removed `autograd` usage.

### Fixes
- Fixed minimization-function bug(s).
- Updated outdated examples.

### Documentation
- Added docs/README notes for retrieving fitted solutions.
- Added uncertainty notes and plotting updates.

## [v0.0.4] - 2022-08-09

### Compatibility
- Adapted to SciPy 1.9 stricter data-type requirements.
- Added SciPy-version handling for RegularGridInterpolator behavior.

### Fixes
- Fixed NaN-extinction behavior when reddening is not fitted.
- Fixed K01 IMF normalization bug.
- Corrected README/docs typos and fitter docstrings.
