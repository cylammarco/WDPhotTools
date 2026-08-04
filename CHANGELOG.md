# CHANGELOG

All notable changes to WDPhotTools are documented in this file.

The 0.0.x line is treated as the beta series. Entries below are listed newest
to oldest and aligned to the 10 beta tags (`0.0.4` to `v0.0.13`).

## [Unreleased] - 2026-07-07

- New feature: Added canonical photometry API support in `WDfitter.fit` with
  `photometry`, `photometry_errors`, and `photometry_space`.
- New feature: Added fitting in flux space alongside magnitude space.
- New feature: Added `self.best_fit_photometry` in the selected
  `photometry_space`.
- New feature: Added opt-in interpolator extrapolation controls for atmosphere
  and cooling model readers (default remains disabled).
- New feature: Added sanitisation for extrapolated values to avoid aphysical
  outputs (`NaN`, `inf`, and impossible negatives) in affected paths.
- Behaviour: Kept in-range interpolation and boundary behaviour unchanged when
  extrapolation is disabled.
- API change (breaking): Removed legacy fit inputs `mags`, `mag_errors`,
  `fluxes`, and `flux_errors`.
- API change (breaking): Standardised `self.fitting_params` to only
  `photometry`, `photometry_errors`, and `photometry_space`.
- API change (breaking): Removed legacy output aliases `best_fit_mag` and
  `best_fit_flux`.
- Logging: Replaced touched `print` calls with structured `logging`.
- Logging: Added INFO/WARNING logs at fitter orchestration boundaries.
- Documentation: Updated README, examples, and docstrings to the canonical
  photometry API.
- Documentation: Updated RTD configuration/build and migration guidance.
- Documentation: Added extrapolation guidance and a success-rate table across
  10% to 50% extrapolation levels.
- Tests: Updated fitter tests to canonical API calls.
- Tests: Added deterministic flux-vs-magnitude parity tests.
- Tests: Added deterministic regression checks against `origin/main`.
- Tests: Reduced runtime for the new parity/regression matrix.
- Tests: Added extrapolation safety and boundary-behaviour coverage for
  atmosphere/cooling interpolators.
- Examples: Updated example scripts for the new API/workflow expectations and
  checked near-identical behaviour against `v0.0.13` where deterministic.
- Documentation: Added interpolation scheme explanation

## [v0.0.13] - 2026-01-14

- Release: Versioned release `v0.0.13` (tag commit `e81da92`).

## [v0.0.12] - 2025-12-27

- Fix: Corrected conversion constant usage in fitter-related numerical paths.
- Fix: Ensured `_integrand` returns `float` for improved compatibility.
- Compatibility: Improved posture for the NumPy 1/2 transition.
- Compatibility: Marked Windows with Python 3.13/3.14 CI as allowed failures
  during transition.
- CI/tooling: Updated GitHub Actions workflow and checkout handling.
- CI/tooling: Updated setup metadata and test-pypi workflow on `dev`.
- CI/tooling: Performed general housekeeping before release.

## [v0.0.11] - 2025-09-22

- Performance: Significantly improved WDLF computation runtime.
- Performance: Refactored basis interpolation and integration helper paths.
- Fix: Corrected wrong-array update bug and additional minor defects.
- Fix: Cleaned redundant `.keys()` usage.
- Dependency/packaging: Replaced `pkg_resources` with `importlib`.
- Documentation/tooling: Updated RTD configuration/docs plus formatting and CI.
- Tests: Updated tests to use `Agg` backend where needed.
- Tests: Fixed test filename issues.

## [v0.0.10] - 2025-05-13

- New feature: Added support for user-provided priors in fitting workflows.
- Validation/fix: Added and updated tests for prior handling.
- Validation/fix: Fixed missing `z_min` and `z_max` handling in reddening
  cases.
- Validation/fix: Decoupled objective functions from fitter orchestration.
- Validation/fix: Avoided interpolation-grid breakage in log-scale age grids.
- CI/tooling: Updated test environments and weekly build checks.
- CI/tooling: Updated pre-commit line-length configuration to 120 chars.
- CI/tooling: Applied multiple housekeeping, style, and markdown fixes.

## [v0.0.9] - 2023-08-11

- Fix: Corrected example script syntax issues.
- Fix: Corrected interpolation variable-allowlist omissions.
- Fix: Corrected plotting type and element-wise comparison edge cases.
- Fix: Added `> 0` safety checks in density computation paths.

## [v0.0.8] - 2023-05-07

- New feature: Added and validated fitter mass-estimation test coverage.
- Fix: Corrected fitting-mass bug.
- Fix: Corrected independent-variable name comparison robustness.
- Fix: Migrated from deprecated `interp2d` to RBF interpolation.
- Fix: Standardised fitter method outputs to scalar floats.
- Documentation: Updated README and RTD content plus citation references.
- Documentation: Added clarifications on uncertainty/error estimation behaviour.

## [v0.0.7] - 2022-12-04

- New feature: Stored `number_density` as object properties before CSV export.
- Fix: Corrected sign bug in `dL/dt` number-density computation.
- Fix: Added protections for log-scale y-axis with non-positive data.
- Fix: Corrected negative-zero and additional plotting/cleanup issues.
- Fix: Corrected default `Rv` from `0.0` to `3.1`.
- CI/tooling: Added Python 3.11 in tests.
- CI/tooling: Added code-analysis workflow and broad tidying.

## [v0.0.6] - 2022-10-24

- New feature: Added linear fractional extinction model support.
- New feature: Added RA/Dec-aware fitter updates for extinction workflows.
- New feature: Extended reddening API for linearly interpolated extinction.
- Compatibility: Lowered minimum `astropy` version to support Python 3.7.
- Compatibility: Added Python 3.10 and Windows CI/test coverage.
- Tests: Extended tests for new reddening/extinction paths.
- Tests: Reduced MCMC test walkers/steps/burn-in and relaxed tolerance for
  stability.
- Fix: Corrected dependent-variable distance handling edge cases.

## [v0.0.5] - 2022-09-12

- Refactor: Moved package code to `src/` layout.
- Refactor: Removed `autograd` usage.
- Fix: Corrected minimization-function bug(s).
- Fix: Updated outdated examples.
- Documentation: Added docs and README notes for retrieving fitted solutions.
- Documentation: Added uncertainty notes and plotting updates.

## [v0.0.4] - 2022-08-09

- Compatibility: Adapted to SciPy 1.9 stricter data-type requirements.
- Compatibility: Added SciPy-version handling for RegularGridInterpolator
  behaviour.
- Fix: Corrected NaN-extinction behaviour when reddening is not fitted.
- Fix: Corrected K01 IMF normalization bug.
- Fix: Corrected README/docs typos and fitter docstrings.
