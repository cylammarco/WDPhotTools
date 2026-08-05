#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for WD fitter in relative flux space."""

import numpy as np
import pytest

from WDPhotTools.fitter import WDfitter

FIVE_FILTERS = np.array(["G3", "G3_BP", "G3_RP", "FUV", "NUV"])
MAGS = np.array([10.882, 10.853, 10.946, 11.301, 11.183], dtype=float)
MAG_ERRORS = np.ones(FIVE_FILTERS.size, dtype=float) * 0.02
RELATIVE_FLUX = 10.0 ** (-0.4 * MAGS)
RELATIVE_FLUX_ERRORS = RELATIVE_FLUX * np.log(10.0) / 2.5 * MAG_ERRORS


def _run_fit(space, **kwargs):
    ftr = WDfitter()
    if space == "magnitude":
        ftr.fit(
            atmosphere="H",
            filters=FIVE_FILTERS,
            photometry=MAGS,
            photometry_errors=MAG_ERRORS,
            photometry_space=space,
            **kwargs,
        )
    else:
        ftr.fit(
            atmosphere="H",
            filters=FIVE_FILTERS,
            photometry=RELATIVE_FLUX,
            photometry_errors=RELATIVE_FLUX_ERRORS,
            photometry_space=space,
            **kwargs,
        )
    return ftr


def test_flux_minimize_teff():
    ftr = _run_fit(
        "flux",
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


def test_flux_least_squares_teff_logg():
    ftr = _run_fit(
        "flux",
        independent=["Teff", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_flux_allow_none_and_plot_axis():
    photometry = [RELATIVE_FLUX[0], RELATIVE_FLUX[1], RELATIVE_FLUX[2], None, RELATIVE_FLUX[4]]
    photometry_errors = [
        RELATIVE_FLUX_ERRORS[0],
        RELATIVE_FLUX_ERRORS[1],
        RELATIVE_FLUX_ERRORS[2],
        1.0,
        RELATIVE_FLUX_ERRORS[4],
    ]
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=FIVE_FILTERS,
        photometry=photometry,
        photometry_errors=photometry_errors,
        photometry_space="flux",
        allow_none=True,
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    fig = ftr.show_best_fit(display=False, savefig=False, return_fig=True)
    assert fig.gca().get_ylabel() == "Relative flux"


def test_flux_emcee_smoke():
    np.random.seed(42)
    ftr = _run_fit(
        "flux",
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        nwalkers=20,
        nsteps=80,
        nburns=20,
        progress=False,
    )
    assert np.isfinite(ftr.best_fit_params["H"]["Teff"])


def test_flux_input_validation():
    with pytest.raises(ValueError):
        WDfitter().fit(
            filters=FIVE_FILTERS,
            photometry=RELATIVE_FLUX,
            photometry_errors=RELATIVE_FLUX_ERRORS,
            photometry_space="unknown",
            independent=["Teff"],
            initial_guess=[13000.0],
            distance=10.0,
            distance_err=0.1,
        )

    with pytest.raises(ValueError):
        WDfitter().fit(
            filters=FIVE_FILTERS,
            photometry_space="flux",
            independent=["Teff"],
            initial_guess=[13000.0],
            distance=10.0,
            distance_err=0.1,
        )


def test_canonical_storage_fields():
    ftr = _run_fit(
        "magnitude",
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    assert "photometry" in ftr.fitting_params
    assert "photometry_errors" in ftr.fitting_params
    assert "mags" not in ftr.fitting_params
    assert "mag_errors" not in ftr.fitting_params
    assert "fluxes" not in ftr.fitting_params
    assert "flux_errors" not in ftr.fitting_params
    assert len(ftr.best_fit_photometry["H"]) == FIVE_FILTERS.size


def test_legacy_keyword_arguments_removed():
    with pytest.raises(TypeError):
        WDfitter().fit(
            filters=FIVE_FILTERS,
            mags=MAGS,
            photometry_errors=MAG_ERRORS,
            independent=["Teff"],
            initial_guess=[13000.0],
            distance=10.0,
            distance_err=0.1,
        )


PARITY_CASES = [
    {
        "name": "ct_minimize_teff",
        "kwargs": {
            "independent": ["Teff"],
            "method": "minimize",
            "atmosphere_interpolator": "CT",
            "initial_guess": [13000.0],
            "logg": 7.5,
            "distance": 10.0,
            "distance_err": 0.1,
        },
    },
    {
        "name": "ct_lsq_teff_logg",
        "kwargs": {
            "independent": ["Teff", "logg"],
            "method": "least_squares",
            "atmosphere_interpolator": "CT",
            "initial_guess": [13000.0, 7.5],
            "distance": 10.0,
            "distance_err": 0.1,
        },
    },
    {
        "name": "ct_lsq_teff_reddening",
        "kwargs": {
            "independent": ["Teff"],
            "method": "least_squares",
            "atmosphere_interpolator": "CT",
            "initial_guess": [13000.0],
            "logg": 7.5,
            "distance": 10.0,
            "distance_err": 0.1,
            "rv": 3.1,
            "ebv": 0.123,
        },
    },
    {
        "name": "rbf_minimize_teff_logg_distance",
        "kwargs": {
            "independent": ["Teff", "logg"],
            "method": "minimize",
            "atmosphere_interpolator": "RBF",
            "initial_guess": [13000.0, 7.5, 10.0],
            "distance": None,
            "distance_err": None,
        },
    },
]


@pytest.mark.parametrize("case", PARITY_CASES, ids=[c["name"] for c in PARITY_CASES])
def test_flux_vs_magnitude_parameter_parity(case):
    fit_mag = _run_fit("magnitude", **case["kwargs"])
    fit_flux = _run_fit("flux", **case["kwargs"])

    mag = fit_mag.best_fit_params["H"]
    flux = fit_flux.best_fit_params["H"]

    assert np.isclose(mag["Teff"], flux["Teff"], rtol=5e-03, atol=0.0)
    assert np.isclose(mag["logg"], flux["logg"], atol=5e-03, rtol=0.0)
    assert np.isclose(mag["distance"], flux["distance"], rtol=5e-03, atol=0.0)
    assert np.isclose(mag["Mbol"], flux["Mbol"], atol=3e-02, rtol=0.0)
    assert np.isfinite(mag["chi2"]) and np.isfinite(flux["chi2"])

    # Avoid unstable relative comparison when chi2 values are effectively zero.
    if max(abs(mag["chi2"]), abs(flux["chi2"])) > 1e-12:
        assert abs(np.log10(abs(mag["chi2"])) - np.log10(abs(flux["chi2"]))) <= 1.0

    assert len(fit_mag.best_fit_photometry["H"]) == FIVE_FILTERS.size
    assert len(fit_flux.best_fit_photometry["H"]) == FIVE_FILTERS.size
