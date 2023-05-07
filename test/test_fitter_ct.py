#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing the fitter using CT interpolator"""

import numpy as np

from WDPhotTools.fitter import WDfitter
from WDPhotTools.reddening import reddening_vector_filter
from WDPhotTools.reddening import reddening_vector_interpolated

# testing with logg=7.5 and Teff=13000.
wave_GBRFN = np.array((6218.0, 5110.0, 7769.0, 1535.0, 2301.0))
wave_grizyJHK = np.array(
    (4849.0, 6201.0, 7535.0, 8674.0, 9628.0, 12350.00, 16460.00, 21600.00)
)

RV = 3.1
EBV = 0.123

reddening = reddening_vector_interpolated(kernel="cubic")
extinction_interpolated = reddening(wave_GBRFN, RV) * EBV
extinction_grizyJHK_interpolated = reddening(wave_grizyJHK, RV) * EBV

A_G3 = reddening_vector_filter("G3")([7.5, 13000.0, RV]) * EBV
A_G3_BP = reddening_vector_filter("G3_BP")([7.5, 13000.0, RV]) * EBV
A_G3_RP = reddening_vector_filter("G3_RP")([7.5, 13000.0, RV]) * EBV
A_FUV = reddening_vector_filter("FUV")([7.5, 13000.0, RV]) * EBV
A_NUV = reddening_vector_filter("NUV")([7.5, 13000.0, RV]) * EBV
A_g = reddening_vector_filter("g_ps1")([7.5, 13000.0, RV]) * EBV
A_r = reddening_vector_filter("r_ps1")([7.5, 13000.0, RV]) * EBV
A_i = reddening_vector_filter("i_ps1")([7.5, 13000.0, RV]) * EBV
A_z = reddening_vector_filter("z_ps1")([7.5, 13000.0, RV]) * EBV
A_y = reddening_vector_filter("y_ps1")([7.5, 13000.0, RV]) * EBV
A_J = reddening_vector_filter("J")([7.5, 13000.0, RV]) * EBV
A_H = reddening_vector_filter("H")([7.5, 13000.0, RV]) * EBV
A_Ks = reddening_vector_filter("Ks")([7.5, 13000.0, RV]) * EBV

mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
extinction = np.array([A_G3, A_G3_BP, A_G3_RP, A_FUV, A_NUV]).reshape(-1)

mags_grizyJHK = np.array(
    [10.764, 11.006, 11.262, 11.482, 11.633, 11.106, 11.139, 11.187]
)
extinction_grizyJHK = np.array(
    [A_g, A_r, A_i, A_z, A_y, A_J, A_H, A_Ks]
).reshape(-1)

five_filters_name_list = np.array(["G3", "G3_BP", "G3_RP", "FUV", "NUV"])
thirteen_filters_name_list = np.array(
    [
        "G3",
        "G3_BP",
        "G3_RP",
        "FUV",
        "NUV",
        "g_ps1",
        "r_ps1",
        "i_ps1",
        "z_ps1",
        "y_ps1",
        "J",
        "H",
        "Ks",
    ]
)


def test_minimize_teff_logg():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: No
    """
    ftr1 = WDfitter()
    ftr1.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr1.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr1.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: No
    """
    ftr2 = WDfitter()
    ftr2.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr2.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr2.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: Yes
    """
    ftr3 = WDfitter()
    ftr3.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr3.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr3.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr4 = WDfitter()
    ftr4.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="minimize",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr4.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr4.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_logg_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: Yes
    """
    ftr5 = WDfitter()
    ftr5.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr5.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr5.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_logg_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr6 = WDfitter()
    ftr6.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="minimize",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr6.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr6.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_logg_distance():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: No
    """
    ftr7 = WDfitter()
    ftr7.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
    )
    assert np.isclose(
        ftr7.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr7.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_distance():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: No
    """
    ftr8 = WDfitter()
    ftr8.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        logg=7.5,
    )
    assert np.isclose(
        ftr8.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr8.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr9 = WDfitter()
    ftr9.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        logg=7.5,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr9.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr9.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr10 = WDfitter()
    ftr10.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="minimize",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        logg=7.5,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr10.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr10.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_logg_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr11 = WDfitter()
    ftr11.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr11.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr11.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_logg_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr12 = WDfitter()
    ftr12.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="minimize",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr12.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr12.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


# least_squares


def test_lsq_teff_logg():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: No
    """
    ftr13 = WDfitter()
    ftr13.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr13.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr13.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: No
    """
    ftr14 = WDfitter()
    ftr14.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr14.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr14.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: Yes
    """
    ftr15 = WDfitter()
    ftr15.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr15.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr15.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr16 = WDfitter()
    ftr16.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="least_squares",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr16.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr16.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_logg_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: Yes
    """
    ftr17 = WDfitter()
    ftr17.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr17.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr17.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_logg_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr18 = WDfitter()
    ftr18.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="least_squares",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr18.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr18.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_logg_distance():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: No
    """
    ftr19 = WDfitter()
    ftr19.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
    )
    assert np.isclose(
        ftr19.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr19.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_distance():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: No
    """
    ftr20 = WDfitter()
    ftr20.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        logg=7.5,
    )
    assert np.isclose(
        ftr20.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr20.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr21 = WDfitter()
    ftr21.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        logg=7.5,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr21.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr21.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr22 = WDfitter()
    ftr22.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="least_squares",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        logg=7.5,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr22.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr22.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_logg_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr23 = WDfitter()
    ftr23.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr23.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr23.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_logg_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr24 = WDfitter()
    ftr24.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="least_squares",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr24.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr24.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


#
# emcee
#


def test_emcee_teff_logg():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: No
    """
    ftr25 = WDfitter()
    ftr25.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr25.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr25.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: No
    """
    ftr26 = WDfitter()
    ftr26.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr26.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr26.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: Yes
    """
    ftr27 = WDfitter()
    ftr27.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr27.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr27.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr28 = WDfitter()
    ftr28.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="CT",
        extinction_convolved=False,
        initial_guess=[13000.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr28.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr28.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_logg_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: Yes
    """
    ftr29 = WDfitter()
    ftr29.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr29.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr29.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_logg_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr30 = WDfitter()
    ftr30.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        distance=10.0,
        distance_err=0.1,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr30.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr30.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_logg_distance():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: No
    """
    ftr31 = WDfitter()
    ftr31.fit(
        atmosphere="H",
        filters=thirteen_filters_name_list,
        mags=np.concatenate((mags, mags_grizyJHK)),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
    )
    assert np.isclose(
        ftr31.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr31.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_distance():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: No
    """
    ftr32 = WDfitter()
    ftr32.fit(
        atmosphere="H",
        filters=thirteen_filters_name_list,
        mags=np.concatenate((mags, mags_grizyJHK)),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
    )
    assert np.isclose(
        ftr32.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr32.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr33 = WDfitter()
    ftr33.fit(
        atmosphere="H",
        filters=thirteen_filters_name_list,
        mags=np.concatenate(
            (mags + extinction, mags_grizyJHK + extinction_grizyJHK)
        ),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr33.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr33.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: No
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr34 = WDfitter()
    ftr34.fit(
        atmosphere="H",
        filters=thirteen_filters_name_list,
        mags=np.concatenate(
            (
                mags + extinction_interpolated,
                mags_grizyJHK + extinction_grizyJHK_interpolated,
            )
        ),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr34.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr34.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_logg_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr35 = WDfitter()
    ftr35.fit(
        atmosphere="H",
        filters=thirteen_filters_name_list,
        mags=np.concatenate(
            (mags + extinction, mags_grizyJHK + extinction_grizyJHK)
        ),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr35.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr35.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )


def test_emcee_teff_logg_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr36 = WDfitter()
    ftr36.fit(
        atmosphere="H",
        filters=thirteen_filters_name_list,
        mags=np.concatenate(
            (
                mags + extinction_interpolated,
                mags_grizyJHK + extinction_grizyJHK_interpolated,
            )
        ),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 7.5, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        Rv=RV,
        ebv=EBV,
    )
    assert np.isclose(
        ftr36.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr36.best_fit_params["H"]["logg"],
        7.5,
        rtol=1e-01,
        atol=1e-01,
    )
