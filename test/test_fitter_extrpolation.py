#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from WDPhotTools.fitter import WDfitter


# testing with logg=7.5 and Teff=13000.
mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
mag_errors = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
filters = np.array(["G3", "G3_BP", "G3_RP", "FUV", "NUV"])


# The most basic cases
@pytest.mark.parametrize("allow_extrapolation", [False, True])
def test_fitter_extrapolation_rbf(allow_extrapolation):
    f = WDfitter()
    f.fit(
        atmosphere=["H"],
        filters=filters,
        mags=mags,
        mag_errors=mag_errors,
        distance=10.0,
        distance_err=1.0,
        independent=["Mbol", "logg"],
        initial_guess=[12.0, 8.0],
        atmosphere_interpolator="RBF",
        method="least_squares",
        allow_extrapolation=allow_extrapolation,
    )
    assert np.isclose(
        f.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        f.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


@pytest.mark.parametrize("allow_extrapolation", [False, True])
def test_fitter_extrapolation_ct(allow_extrapolation):
    f = WDfitter()
    f.fit(
        atmosphere=["H"],
        filters=filters,
        mags=mags,
        mag_errors=mag_errors,
        distance=10.0,
        distance_err=1.0,
        independent=["Mbol", "logg"],
        initial_guess=[12.0, 8.0],
        atmosphere_interpolator="CT",
        method="least_squares",
        allow_extrapolation=allow_extrapolation,
    )
    assert np.isclose(
        f.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        f.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


# Test for actual extrapolation behavior with Mbol initial guess significantly outside grid
@pytest.mark.parametrize(
    "allow_extrapolation", [pytest.param(False, marks=pytest.mark.xfail(reason="intial guess outside of grid")), True]
)
def test_fitter_extrapolation_rbf_initial_guess_outisde_grid(allow_extrapolation):
    f = WDfitter()
    f.fit(
        atmosphere=["H"],
        filters=filters,
        mags=mags,
        mag_errors=mag_errors,
        distance=10.0,
        distance_err=1.0,
        independent=["Mbol", "logg"],
        initial_guess=[5.0, 8.0],
        atmosphere_interpolator="RBF",
        method="least_squares",
        allow_extrapolation=allow_extrapolation,
    )
    assert np.isclose(
        f.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        f.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


@pytest.mark.parametrize(
    "allow_extrapolation", [pytest.param(False, marks=pytest.mark.xfail(reason="intial guess outside of grid")), True]
)
def test_fitter_extrapolation_ct_initial_guess_outisde_grid(allow_extrapolation):
    f = WDfitter()
    f.fit(
        atmosphere=["H"],
        filters=filters,
        mags=mags,
        mag_errors=mag_errors,
        distance=10.0,
        distance_err=1.0,
        independent=["Mbol", "logg"],
        initial_guess=[5.0, 8.0],
        atmosphere_interpolator="CT",
        method="least_squares",
        allow_extrapolation=allow_extrapolation,
    )
    assert np.isclose(
        f.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        f.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )
