#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing some general methods in a fitter"""

from unittest.mock import patch

import numpy as np
import pytest

from WDPhotTools.fitter import WDfitter
from WDPhotTools.reddening import reddening_vector_filter
from WDPhotTools.reddening import reddening_vector_interpolated


# testing with logg=7.5 and Teff=13000.
wave_GBRFN = np.array((6218.0, 5110.0, 7769.0, 1535.0, 2301.0))

rv = 3.1
ebv = 0.123

reddening = reddening_vector_interpolated(kernel="cubic")
extinction_interpolated = reddening(wave_GBRFN, rv) * ebv

A_G3 = reddening_vector_filter("G3")([7.5, 13000.0, rv]) * ebv
A_G3_BP = reddening_vector_filter("G3_BP")([7.5, 13000.0, rv]) * ebv
A_G3_RP = reddening_vector_filter("G3_RP")([7.5, 13000.0, rv]) * ebv
A_FUV = reddening_vector_filter("FUV")([7.5, 13000.0, rv]) * ebv
A_NUV = reddening_vector_filter("NUV")([7.5, 13000.0, rv]) * ebv

mags = [10.882, 10.853, 10.946, 11.301, 11.183]
extinction = np.array([A_G3, A_G3_BP, A_G3_RP, A_FUV, A_NUV]).reshape(-1)


def test_list_everything():
    """List all atmosphere parameters"""
    ftr = WDfitter()
    ftr.list_atmosphere_parameters()


def test_fitter_get_extinction_fraction():
    """Test exitnction function"""
    ftr = WDfitter()
    ftr.set_extinction_mode(mode="linear", z_min=100.0, z_max=250.0)
    np.isclose(
        ftr._get_extinction_fraction(
            distance=250.0,
            ra=192.85949646,
            dec=27.12835323,
        ),
        1.0,
    )
    np.isclose(
        ftr._get_extinction_fraction(
            distance=75.0, ra=192.85949646, dec=27.12835323
        ),
        0.0,
    )
    np.isclose(
        ftr._get_extinction_fraction(
            distance=100.0, ra=192.85949646, dec=27.12835323
        ),
        0.1386644841777267,
    )
    np.isclose(
        ftr._get_extinction_fraction(
            distance=175.0, ra=192.85949646, dec=27.12835323
        ),
        0.6386628473110217,
    )
    np.isclose(
        ftr._get_extinction_fraction(
            distance=250.0, ra=192.85949646, dec=27.12835323
        ),
        1.0,
    )
    np.isclose(
        ftr._get_extinction_fraction(
            distance=251.0, ra=192.85949646, dec=27.12835323
        ),
        1.0,
    )
    np.isclose(
        ftr._get_extinction_fraction(
            distance=500.0, ra=240.63412385, dec=-11.01234783
        ),
        1.0,
    )


def test_fitter_change_extinction_mode():
    """Test chaning extinction mode"""
    ftr = WDfitter()
    ftr.set_extinction_mode(mode="linear", z_min=100.0, z_max=250.0)
    assert np.isclose(
        ftr._get_extinction_fraction(
            distance=175.0, ra=192.85949646, dec=27.12835323
        ),
        0.6386628473110217,
    )
    assert ftr.extinction_mode == "linear"
    ftr.set_extinction_mode(mode="total")
    assert ftr.extinction_mode == "total"


@pytest.mark.xfail
def test_fitter_change_extinction_mode_unknown():
    """Test giving unknown mode of extinction"""
    ftr = WDfitter()
    ftr.set_extinction_mode(mode="blabla")


@pytest.mark.xfail
def test_fitter_get_extinction_fraction_fail_zmin():
    """Test giving negative z-distance"""
    ftr = WDfitter()
    ftr.set_extinction_mode(mode="linear", z_min=-10.0, z_max=250.0)


@pytest.mark.xfail
def test_fitter_get_extinction_fraction_fail_zmax():
    """Test giving maximum smaller than minimum"""
    ftr = WDfitter()
    ftr.set_extinction_mode(mode="linear", z_min=1000.0, z_max=250.0)


@pytest.mark.xfail
def test_fitter_get_extinction_fraction_pass_zmin_zmax_none():
    """Test dec larger than 90"""
    ftr = WDfitter()
    ftr._get_extinction_fraction(distance=150.0, ra=10.0, dec=100.0)


@pytest.mark.xfail
def test_fitter_get_extinction_fraction_fail_ra():
    """test ra and dec are None"""
    ftr = WDfitter()
    ftr._get_extinction_fraction(distance=150.0, ra=None, dec=None)


@patch("matplotlib.pyplot.show")
def test_fitting_teff(mock_show):
    """Fitting for Teff with 5 filters for both DA and DB"""
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
    )
    ftr.show_best_fit(
        title="Testing Teff = 13000.0",
        display=True,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
        return_fig=True,
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_teff_with_none():
    """
    Fitting for Teff with 5 filters for both DA and DB with alternating None
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 10.0],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, 10.350],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 0.1],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 10.0],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_logg_and_mbol():
    """
    Fitting for logg and Teff with 5 filters for both DA and DB
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0, 7.5],
    )
    ftr.show_best_fit(
        display=False,
        folder="test_output",
        filename="test_fitting_logg_and_mbol",
        ext="png",
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance():
    """
    Fitting for logg, Teff and distance with 5 filters for both DA and DB
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
    )
    ftr.show_best_fit(display=False)
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance_nelder_mead():
    """
    Fitting for logg, Teff and distance with 8 filters for both DA and DB with
    Nelder-Mead method
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        kwargs_for_minimize={"method": "Nelder-Mead"},
    )
    ftr.show_best_fit(display=False)
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_teff_red():
    """
    Fitting for Teff with 5 filters for both DA and DB with added extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
        return_fig=True,
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_logg_and_teff_red():
    """
    Fitting for logg and Teff with 5 filters for both DA and DB with added
    extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0, 7.5],
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False,
        folder="test_output",
        filename="test_fitting_logg_and_mbol",
        ext="png",
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance_red():
    """
    Fitting for logg, Teff and distance with 5 filters for both DA and DB with
    added extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False, title="fitted (logg, Mbol, distance) and dereddend"
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance_red_best_fit_plot_colour():
    """
    Fitting for logg, Teff and distance with 5 filters for both DA and DB with
    added extinction
    Manually chaning plot colours
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        color="purple",
        atmosphere="H",
        display=False,
        title="fitted (logg, Mbol, distance) and dereddend",
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


#
#
#
# Repeat all the test with optimize.least_square
#
#
#
#
def test_fitting_teff_lsq():
    """
    Fitting for Teff with 5 filters for both DA and DB
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        logg=7.5,
        independent=["Teff"],
        distance=10.0,
        distance_err=0.1,
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
    )
    ftr.show_best_fit(
        display=False,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
        return_fig=True,
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_teff_with_none_lsq():
    """
    Fitting for Teff with 5 filters for both DA and DB with alternating None
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 10.0],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        distance=10.0,
        distance_err=0.1,
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, 10.350],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 0.1],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        distance=10.0,
        distance_err=0.1,
        method="least_squares",
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 10.0],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        distance=10.0,
        distance_err=0.1,
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_logg_and_teff_lsq():
    """
    Fitting for logg and Teff with 5 filters for both DA and DB
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        distance=10.0,
        distance_err=0.1,
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
    )
    ftr.show_best_fit(
        display=False,
        folder="test_output",
        filename="test_fitting_logg_and_mbol",
        ext="png",
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance_lsq():
    """
    Fitting for logg, Teff and distance with 5 filters for both DA and DB
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
    )
    ftr.show_best_fit(display=False)
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance_nelder_mead_lsq():
    """
    Fitting for logg, Teff and distance with 8 filters for both DA and DB with
    Nelder-Mead method
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        initial_guess=[13000.0, 7.5, 10.0],
        method="least_squares",
        atmosphere_interpolator="CT",
    )
    ftr.show_best_fit(display=False)
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_teff_red_lsq():
    """
    Fitting for Teff with 5 filters for both DA and DB with added extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        logg=7.5,
        independent=["Teff"],
        distance=10.0,
        distance_err=0.1,
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
        return_fig=True,
    )
    assert np.isclose(
        ftr.results["H"].x, np.array([13000.0]), rtol=2.5e-02, atol=2.5e-02
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_logg_and_teff_red_lsq():
    """
    Fitting for logg and Teff with 5 filters for both DA and DB with added
    extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0, 7.5],
        method="least_squares",
        atmosphere_interpolator="CT",
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False,
        folder="test_output",
        filename="test_fitting_logg_and_mbol",
        ext="png",
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


def test_fitting_logg_teff_distance_red_lsq():
    """
    Fitting for logg, Teff and distance with 5 filters for both DA and DB with
    added extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        initial_guess=[13000.0, 7.5, 10.0],
        method="least_squares",
        atmosphere_interpolator="CT",
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False, title="fitted (logg, Mbol, distance) and dereddend"
    )
    assert np.isclose(
        ftr.results["H"].x,
        np.array([13000.0, 7.5, 10.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)


#
#
# Repeat all the test with emcee
#
#
#
def test_fitting_teff_emcee():
    """
    Fitting for logg and Teff with 5 filters for both DA and DB
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        method="emcee",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
    )
    ftr.show_corner_plot(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_fitting_and_mbol_corner",
        ext="png",
    )
    ftr.show_best_fit(
        display=False,
        savefig=True,
        folder="test_output",
        filename="test_fitting_and_mbol",
        ext="png",
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        np.array([13000.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_teff_with_none_emcee():
    """
    Fitting for Teff with 5 filters for both DA and DB with alternating None
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 10.0],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        method="emcee",
        distance=10.0,
        distance_err=0.1,
        refine=True,
        refine_bounds=[0.1, 99.9],
        initial_guess=[13000.0],
    )
    ftr.show_corner_plot(
        display=False, folder="test_output", ext=["png", "pdf"]
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        np.array([13000.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, 10.350],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 0.1],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        method="emcee",
        distance=10.0,
        distance_err=0.1,
        refine=True,
        refine_bounds=[0.1, 99.9],
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        np.array([13000.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV", "U"],
        mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02, 10.0],
        allow_none=True,
        atmosphere="H",
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        method="emcee",
        distance=10.0,
        distance_err=0.1,
        refine=True,
        refine_bounds=[0.1, 99.9],
        initial_guess=[13000.0],
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        np.array([13000.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_fitting_teff_red_emcee():
    """
    Fitting for Teff with 5 filters for both DA and DB with added extinction
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        logg=7.5,
        independent=["Teff"],
        atmosphere_interpolator="CT",
        method="emcee",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0],
        refine=True,
        refine_bounds=[0.1, 99.9],
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False,
        savefig=True,
        folder="test_output",
        ext=["png", "pdf"],
        return_fig=True,
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        np.array([13000.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert np.isclose(
        ftr.best_fit_params["H"]["logg"],
        np.array([7.5]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)


def test_chi2_minimization_red_interpolated():
    """
    Testing _chi2_minimization_red_interpolated()
    """
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction_interpolated,
        mag_errors=[0.02, 0.02, 0.02, 0.02, 0.02],
        independent=["Teff", "logg"],
        atmosphere_interpolator="CT",
        method="least_squares",
        distance=10.0,
        distance_err=0.1,
        initial_guess=[13000.0, 7.5],
        refine=True,
        refine_bounds=[0.1, 99.9],
        extinction_convolved=False,
        Rv=rv,
        ebv=ebv,
    )
    ftr.show_best_fit(
        display=False,
        folder="test_output",
        filename="test_chi2_minimization_red_interpolated",
        ext="png",
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        np.array([13000.0]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert np.isclose(
        ftr.best_fit_params["H"]["logg"],
        np.array([7.5]),
        rtol=2.5e-02,
        atol=2.5e-02,
    ).all()
    assert isinstance(ftr.best_fit_params["H"]["Teff"], float)
    assert isinstance(ftr.best_fit_params["H"]["Mbol"], float)
    assert isinstance(ftr.best_fit_params["H"]["Teff_err"], float)
    assert isinstance(ftr.best_fit_params["H"]["logg_err"], float)
