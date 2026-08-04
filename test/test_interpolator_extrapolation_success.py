#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extrapolation quality checks for atmosphere and cooling interpolators."""

import inspect

import numpy as np
import pytest

from WDPhotTools.atmosphere_model_reader import AtmosphereModelReader
from WDPhotTools.cooling_model_reader import CoolingModelReader


def _sample_with_fraction(rng, value_min, value_max, fraction, size):
    span = value_max - value_min
    return rng.uniform(value_min - span * fraction, value_max + span * fraction, size=size)


def _estimate_atmosphere_success_rates(interpolator, sample_size=200, seed=12345):
    rng = np.random.default_rng(seed)
    atm = AtmosphereModelReader()
    itp = atm.interp_am(
        dependent="Teff",
        atmosphere="H",
        independent=["logg", "Mbol"],
        interpolator=interpolator,
        allow_extrapolation=True,
    )

    logg_grid = np.asarray(atm.model_da["logg"], dtype=float)
    mbol_grid = np.asarray(atm.model_da["Mbol"], dtype=float)
    logg_min, logg_max = float(np.nanmin(logg_grid)), float(np.nanmax(logg_grid))
    mbol_min, mbol_max = float(np.nanmin(mbol_grid)), float(np.nanmax(mbol_grid))

    rates = {}
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5):
        logg = _sample_with_fraction(rng, logg_min, logg_max, fraction, sample_size)
        mbol = _sample_with_fraction(rng, mbol_min, mbol_max, fraction, sample_size)
        extrapolated = (logg < logg_min) | (logg > logg_max) | (mbol < mbol_min) | (mbol > mbol_max)
        teff = np.asarray(itp(logg, mbol), dtype=float).reshape(-1)
        valid = np.isfinite(teff) & (teff > 0.0)
        rates[fraction] = float(np.mean(valid[extrapolated])) if np.any(extrapolated) else 1.0

    return rates


def _estimate_cooling_success_rates(interpolator, sample_size=200, seed=12345):
    rng = np.random.default_rng(seed)
    cmr = CoolingModelReader()
    cmr.compute_cooling_age_interpolator(
        interpolator=interpolator,
        allow_extrapolation=True,
    )

    logl_grid = np.log10(np.asarray(cmr.luminosity, dtype=float))
    mass_grid = np.asarray(cmr.mass, dtype=float)
    logl_min, logl_max = float(np.nanmin(logl_grid)), float(np.nanmax(logl_grid))
    mass_min, mass_max = float(np.nanmin(mass_grid)), float(np.nanmax(mass_grid))

    rates = {}
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5):
        logl = _sample_with_fraction(rng, logl_min, logl_max, fraction, sample_size)
        mass = _sample_with_fraction(rng, mass_min, mass_max, fraction, sample_size)
        extrapolated = (logl < logl_min) | (logl > logl_max) | (mass < mass_min) | (mass > mass_max)
        cooling_age = np.asarray(cmr.cooling_interpolator(logl, mass), dtype=float).reshape(-1)
        cooling_rate = np.asarray(cmr.cooling_rate_interpolator(logl, mass), dtype=float).reshape(-1)
        valid = np.isfinite(cooling_age) & np.isfinite(cooling_rate) & (cooling_age > 0.0) & (cooling_rate <= 0.0)
        rates[fraction] = float(np.mean(valid[extrapolated])) if np.any(extrapolated) else 1.0

    return rates


def test_extrapolation_defaults_remain_disabled():
    """Ensure extrapolation defaults remain disabled for public APIs."""
    interp_am_sig = inspect.signature(AtmosphereModelReader.interp_am)
    compute_sig = inspect.signature(CoolingModelReader.compute_cooling_age_interpolator)
    assert interp_am_sig.parameters["allow_extrapolation"].default is False
    assert compute_sig.parameters["allow_extrapolation"].default is False


@pytest.mark.parametrize("interpolator", ["CT", "RBF"])
def test_atmosphere_extrapolation_success_rate(interpolator):
    """10% atmosphere extrapolation must return finite and physically valid Teff."""
    rates = _estimate_atmosphere_success_rates(interpolator)
    assert rates[0.1] >= 0.99
    assert rates[0.5] >= 0.95


@pytest.mark.parametrize("interpolator", ["CT", "RBF"])
def test_cooling_extrapolation_success_rate(interpolator):
    """10% cooling extrapolation must return finite positive age and finite non-positive dL/dt."""
    rates = _estimate_cooling_success_rates(interpolator)
    assert rates[0.1] >= 0.99
    assert rates[0.5] >= 0.95


@pytest.mark.parametrize("interpolator", ["CT", "RBF"])
def test_atmosphere_boundary_behaviour_unchanged(interpolator):
    """Boundary and out-of-bounds behaviour stays unchanged when extrapolation is disabled."""
    atm = AtmosphereModelReader()
    interp_no_extrap = atm.interp_am(
        dependent="Teff",
        atmosphere="H",
        independent=["logg", "Mbol"],
        interpolator=interpolator,
        allow_extrapolation=False,
    )
    interp_with_extrap = atm.interp_am(
        dependent="Teff",
        atmosphere="H",
        independent=["logg", "Mbol"],
        interpolator=interpolator,
        allow_extrapolation=True,
    )

    logg_grid = np.asarray(atm.model_da["logg"], dtype=float)
    mbol_grid = np.asarray(atm.model_da["Mbol"], dtype=float)
    logg_min, logg_max = float(np.nanmin(logg_grid)), float(np.nanmax(logg_grid))
    mbol_min, mbol_max = float(np.nanmin(mbol_grid)), float(np.nanmax(mbol_grid))

    boundary_logg = np.array([logg_min, logg_min, logg_max, logg_max], dtype=float)
    boundary_mbol = np.array([mbol_min, mbol_max, mbol_min, mbol_max], dtype=float)
    no_extrap_boundary = np.asarray(interp_no_extrap(boundary_logg, boundary_mbol), dtype=float).reshape(-1)
    with_extrap_boundary = np.asarray(interp_with_extrap(boundary_logg, boundary_mbol), dtype=float).reshape(-1)
    finite = np.isfinite(no_extrap_boundary) & np.isfinite(with_extrap_boundary)
    assert np.allclose(no_extrap_boundary[finite], with_extrap_boundary[finite], rtol=1e-10, atol=1e-6)

    out_of_bounds_logg = np.array([logg_min - 0.1, logg_max + 0.1], dtype=float)
    out_of_bounds_mbol = np.array([mbol_min - 0.1, mbol_max + 0.1], dtype=float)
    no_extrap_oob = np.asarray(interp_no_extrap(out_of_bounds_logg, out_of_bounds_mbol), dtype=float).reshape(-1)
    assert np.all(np.isneginf(no_extrap_oob))


@pytest.mark.parametrize("interpolator", ["CT", "RBF"])
def test_cooling_boundary_behaviour_unchanged(interpolator):
    """Cooling interpolator boundary behaviour stays unchanged when extrapolation is disabled."""
    cmr_no_extrap = CoolingModelReader()
    cmr_no_extrap.compute_cooling_age_interpolator(interpolator=interpolator, allow_extrapolation=False)
    cmr_with_extrap = CoolingModelReader()
    cmr_with_extrap.compute_cooling_age_interpolator(interpolator=interpolator, allow_extrapolation=True)

    lum_grid = np.log10(np.asarray(cmr_no_extrap.luminosity, dtype=float))
    mass_grid = np.asarray(cmr_no_extrap.mass, dtype=float)
    lum_min, lum_max = float(np.nanmin(lum_grid)), float(np.nanmax(lum_grid))
    mass_min, mass_max = float(np.nanmin(mass_grid)), float(np.nanmax(mass_grid))

    boundary_lum = np.array([lum_min, lum_min, lum_max, lum_max], dtype=float)
    boundary_mass = np.array([mass_min, mass_max, mass_min, mass_max], dtype=float)
    no_extrap_boundary = np.asarray(
        cmr_no_extrap.cooling_interpolator(boundary_lum, boundary_mass),
        dtype=float,
    ).reshape(-1)
    with_extrap_boundary = np.asarray(
        cmr_with_extrap.cooling_interpolator(boundary_lum, boundary_mass),
        dtype=float,
    ).reshape(-1)
    finite = np.isfinite(no_extrap_boundary) & np.isfinite(with_extrap_boundary)
    assert np.allclose(no_extrap_boundary[finite], with_extrap_boundary[finite], rtol=1e-10, atol=1e-6)

    out_of_bounds_lum = np.array([lum_min - 0.05, lum_max + 0.05], dtype=float)
    out_of_bounds_mass = np.array([mass_min - 0.05, mass_max + 0.05], dtype=float)
    no_extrap_oob = np.asarray(
        cmr_no_extrap.cooling_interpolator(out_of_bounds_lum, out_of_bounds_mass),
        dtype=float,
    ).reshape(-1)
    assert np.all(np.isneginf(no_extrap_oob))
