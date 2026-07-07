#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Regression checks against pinned origin/main fitter results.

Baseline values were generated from origin/main at commit 59f833d.
"""

import numpy as np
import pytest

from WDPhotTools.fitter import WDfitter

FILTERS = np.array(["G3", "G3_BP", "G3_RP", "FUV", "NUV"])
MAGS = np.array([10.882, 10.853, 10.946, 11.301, 11.183], dtype=float)
MAG_ERRORS = np.ones(FILTERS.size, dtype=float) * 0.02


REGRESSION_CASES = [
    {
        "name": "ct_min_teff_fixed_dist",
        "kwargs": {
            "atmosphere": "H",
            "filters": FILTERS,
            "photometry": MAGS,
            "photometry_errors": MAG_ERRORS,
            "photometry_space": "magnitude",
            "logg": 7.5,
            "independent": ["Teff"],
            "atmosphere_interpolator": "CT",
            "method": "minimize",
            "distance": 10.0,
            "distance_err": 0.1,
            "initial_guess": [13000.0],
        },
        "expected": {
            "Teff": 13000.0,
            "logg": 7.5,
            "distance": 10.0,
            "Mbol": 9.962,
            "chi2": 0.0,
        },
    },
    {
        "name": "ct_lsq_teff_logg_fixed_dist",
        "kwargs": {
            "atmosphere": "H",
            "filters": FILTERS,
            "photometry": MAGS,
            "photometry_errors": MAG_ERRORS,
            "photometry_space": "magnitude",
            "independent": ["Teff", "logg"],
            "atmosphere_interpolator": "CT",
            "method": "least_squares",
            "distance": 10.0,
            "distance_err": 0.1,
            "initial_guess": [13000.0, 7.5],
        },
        "expected": {
            "Teff": 13000.0,
            "logg": 7.5,
            "distance": 10.0,
            "Mbol": 9.962,
            "chi2": 0.0,
        },
    },
    {
        "name": "rbf_min_teff_logg_dist",
        "kwargs": {
            "atmosphere": "H",
            "filters": FILTERS,
            "photometry": MAGS,
            "photometry_errors": MAG_ERRORS,
            "photometry_space": "magnitude",
            "independent": ["Teff", "logg"],
            "atmosphere_interpolator": "RBF",
            "method": "minimize",
            "initial_guess": [13000.0, 7.5, 10.0],
        },
        "expected": {
            "Teff": 13000.000000000116,
            "logg": 7.5,
            "distance": 10.0,
            "Mbol": 9.96199999999996,
            "chi2": 2.4194140247106092e-21,
        },
    },
    {
        "name": "ct_lsq_teff_reddening",
        "kwargs": {
            "atmosphere": "H",
            "filters": FILTERS,
            "photometry": MAGS,
            "photometry_errors": MAG_ERRORS,
            "photometry_space": "magnitude",
            "logg": 7.5,
            "independent": ["Teff"],
            "atmosphere_interpolator": "CT",
            "method": "least_squares",
            "distance": 10.0,
            "distance_err": 0.1,
            "initial_guess": [13000.0],
            "rv": 3.1,
            "ebv": 0.123,
        },
        "expected": {
            "Teff": 15562.152043762782,
            "logg": 7.5,
            "distance": 10.0,
            "Mbol": 9.152973607611566,
            "chi2": 44.26036599996003,
        },
    },
]


@pytest.mark.parametrize("case", REGRESSION_CASES, ids=[c["name"] for c in REGRESSION_CASES])
def test_regression_against_origin_main(case):
    ftr = WDfitter()
    ftr.fit(**case["kwargs"])
    params = ftr.best_fit_params["H"]
    expected = case["expected"]

    assert np.isclose(params["Teff"], expected["Teff"], rtol=1e-02, atol=0.0)
    assert np.isclose(params["logg"], expected["logg"], atol=1e-02, rtol=0.0)
    assert np.isclose(params["distance"], expected["distance"], rtol=1e-02, atol=0.0)
    assert np.isclose(params["Mbol"], expected["Mbol"], atol=1e-02, rtol=0.0)

    assert np.isfinite(params["chi2"])
    if expected["chi2"] <= 1e-20:
        assert abs(params["chi2"]) <= 1e-6
    else:
        assert abs(np.log10(abs(params["chi2"])) - np.log10(abs(expected["chi2"]))) <= 1.0
