#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing the fitter using CT interpolator"""

import numpy as np

from WDPhotTools.fitter import WDfitter
from WDPhotTools.reddening import reddening_vector_filter
from WDPhotTools.reddening import reddening_vector_interpolated

# testing with logg=7.5 and Teff=13000. (=> M = 0.368)
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


def test_minimize_teff_mass_reddening():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: No
    Reddenning: Yes
    """
    ftr5 = WDfitter()
    ftr5.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.6],
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
        ftr5.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_mass_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr6 = WDfitter()
    ftr6.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="minimize",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.6],
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
        ftr6.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_mass_distance():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: Yes
    Reddenning: No
    """
    ftr7 = WDfitter()
    ftr7.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.4, 10.0],
    )
    assert np.isclose(
        ftr7.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr7.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_mass_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr11 = WDfitter()
    ftr11.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.4, 10.0],
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
        ftr11.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_minimize_teff_mass_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr12 = WDfitter()
    ftr12.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="minimize",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.4, 10.0],
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
        ftr12.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


# least_squares


def test_lsq_teff_mass():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: No
    Reddenning: No
    """
    ftr13 = WDfitter()
    ftr13.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.6],
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
        ftr13.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_mass_reddening():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: No
    Reddenning: Yes
    """
    ftr17 = WDfitter()
    ftr17.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.6],
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
        ftr17.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_mass_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: No
    Reddenning: Yes (interpolated)
    """
    ftr18 = WDfitter()
    ftr18.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="least_squares",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.6],
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
        ftr18.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_mass_distance():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: Yes
    Reddenning: No
    """
    ftr19 = WDfitter()
    ftr19.fit(
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.4, 10.0],
    )
    assert np.isclose(
        ftr19.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr19.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_mass_distance_reddening():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: Yes
    Reddenning: Yes
    """
    ftr23 = WDfitter()
    ftr23.fit(
        filters=five_filters_name_list,
        mags=mags + extinction,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.4, 10.0],
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
        ftr23.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )


def test_lsq_teff_mass_distance_reddening_interpolated():
    """
    Fitting Teff: Yes
    Fitting mass: Yes
    Fitting distance: Yes
    Reddenning: Yes (interpolated)
    """
    ftr24 = WDfitter()
    ftr24.fit(
        filters=five_filters_name_list,
        mags=mags + extinction_interpolated,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "mass"],
        method="least_squares",
        extinction_convolved=False,
        atmosphere_interpolator="CT",
        initial_guess=[13000.0, 0.4, 10.0],
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
        ftr24.best_fit_params["H"]["mass"],
        0.368,
        rtol=1e-01,
        atol=1e-01,
    )
