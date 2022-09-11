from WDPhotTools.fitter import WDfitter
import numpy as np
from unittest.mock import patch
import os  # For testing file existence with assert
import time  # For testing file existence with assert

from WDPhotTools.reddening import reddening_vector_filter
from WDPhotTools.reddening import reddening_vector_interpolated
from numpy import False_

# testing with logg=7.5 and Teff=13000.
wave_GBRFN = np.array((6218.0, 5110.0, 7769.0, 1535.0, 2301.0))

rv = 3.1
ebv = 0.123

reddening = reddening_vector_interpolated(kind="cubic")
extinction_interpolated = reddening(wave_GBRFN, rv) * ebv

A_G3 = reddening_vector_filter("G3")([7.5, 13000.0, rv]) * ebv
A_G3_BP = reddening_vector_filter("G3_BP")([7.5, 13000.0, rv]) * ebv
A_G3_RP = reddening_vector_filter("G3_RP")([7.5, 13000.0, rv]) * ebv
A_FUV = reddening_vector_filter("FUV")([7.5, 13000.0, rv]) * ebv
A_NUV = reddening_vector_filter("NUV")([7.5, 13000.0, rv]) * ebv

mags = [10.882, 10.853, 10.946, 11.301, 11.183]
extinction = np.array([A_G3, A_G3_BP, A_G3_RP, A_FUV, A_NUV]).reshape(-1)


# List all atmosphere parameters
def test_list_everything():
    ftr = WDfitter()
    ftr.list_atmosphere_parameters()


# Fitting for Teff with 5 filters for both DA and DB
@patch("matplotlib.pyplot.show")
def test_fitting_Teff(mock_show):
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


# Fitting for Teff with 5 filters for both DA and DB with alternating None
def test_fitting_Teff_with_None():
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


# Fitting for logg and Teff with 5 filters for both DA and DB
def test_fitting_logg_and_mbol():
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


# Fitting for logg, Teff and distance with 5 filters for both DA and DB
def test_fitting_logg_Teff_distance():
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


# Fitting for logg, Teff and distance with 8 filters for both DA and DB with
# Nelder-Mead method
def test_fitting_logg_Teff_distance_nelder_mead():
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


# Fitting for Teff with 5 filters for both DA and DB with added extinction
def test_fitting_Teff_red():
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


# Fitting for logg and Teff with 5 filters for both DA and DB with added
# extinction
def test_fitting_logg_and_Teff_red():
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


# Fitting for logg, Teff and distance with 5 filters for both DA and DB with
# added extinction
def test_fitting_logg_Teff_distance_red():
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


# Fitting for logg, Teff and distance with 5 filters for both DA and DB with
# added extinction
# Manually chaning plot colours
def test_fitting_logg_Teff_distance_red_best_fit_plot_colour():
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


#
#
#
# Repeat all the test with optimize.least_square
#
#
#
#


# Fitting for Teff with 5 filters for both DA and DB
def test_fitting_Teff_lsq():
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


# Fitting for Teff with 5 filters for both DA and DB with alternating None
def test_fitting_Teff_with_None_lsq():
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


# Fitting for logg and Teff with 5 filters for both DA and DB
def test_fitting_logg_and_Teff_lsq():
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


# Fitting for logg, Teff and distance with 5 filters for both DA and DB
def test_fitting_logg_Teff_distance_lsq():
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


# Fitting for logg, Teff and distance with 8 filters for both DA and DB with
# Nelder-Mead method
def test_fitting_logg_Teff_distance_nelder_mead_lsq():
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


# Fitting for Teff with 5 filters for both DA and DB with added extinction
def test_fitting_Teff_red_lsq():
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


# Fitting for logg and Teff with 5 filters for both DA and DB with added
# extinction
def test_fitting_logg_and_Teff_red_lsq():
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


# Fitting for logg, Teff and distance with 5 filters for both DA and DB with
# added extinction
def test_fitting_logg_Teff_distance_red_lsq():
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


#
#
# Repeat all the test with emcee
#
#
#

# Fitting for logg and Teff with 5 filters for both DA and DB
def test_fitting_Teff_emcee():
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


# Fitting for Teff with 5 filters for both DA and DB with alternating None
def test_fitting_Teff_with_None_emcee():
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


# Fitting for Teff with 5 filters for both DA and DB with added extinction
def test_fitting_Teff_red_emcee():
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


# Testing interp_reddening()
def test_interp_reddening():
    ftr = WDfitter()
    ftr.interp_reddening(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        extinction_convolved=True,
    )


# Testing _chi2_minimization_red_interpolated()
def test_chi2_minimization_red_interpolated():
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
        extinction_convolved=False_,
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
