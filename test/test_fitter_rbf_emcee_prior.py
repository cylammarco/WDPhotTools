#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing the fitter using CT interpolator"""

import numpy as np

from WDPhotTools.fitter import WDfitter
from WDPhotTools.reddening import reddening_vector_filter
from WDPhotTools.reddening import reddening_vector_interpolated

# testing with logg=7.5 and Teff=13000.
wave_GBRFN = np.array((6218.0, 5110.0, 7769.0, 1535.0, 2301.0))
wave_grizyJHK = np.array((4849.0, 6201.0, 7535.0, 8674.0, 9628.0, 12350.00, 16460.00, 21600.00))

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

mags_grizyJHK = np.array([10.764, 11.006, 11.262, 11.482, 11.633, 11.106, 11.139, 11.187])
extinction_grizyJHK = np.array([A_g, A_r, A_i, A_z, A_y, A_J, A_H, A_Ks]).reshape(-1)

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


def prior_teff(*args):
    # setting a prior for Teff within 500 and logg within 0.1 from the ground truth
    Teff = args[0]
    if Teff < 13000 or Teff > 14000:
        return -np.inf
    else:
        return 0.0


def prior_teff_logg(*args):
    # setting a prior for Teff within 500 and logg within 0.1 from the ground truth
    Teff = args[0]
    logg = args[1]
    if Teff < 13000 or Teff > 14000 or logg < 7.4 or logg > 7.6:
        return -np.inf
    else:
        return 0.0


def prior_teff_logg_distance(*args):
    # setting a prior for Teff within 500 and logg within 0.1 from the ground truth
    Teff = args[0]
    logg = args[1]
    distance = args[2]
    if Teff < 13000 or Teff > 14000 or logg < 7.4 or logg > 7.6 or distance < 9.5 or distance > 10.5:
        return -np.inf
    else:
        return 0.0


def prior_teff_distance(*args):
    # setting a prior for Teff within 500 and logg within 0.1 from the ground truth
    Teff = args[0]
    distance = args[1]
    if Teff < 13000 or Teff > 14000 or distance < 9.5 or distance > 10.5:
        return -np.inf
    else:
        return 0.0


def test_emcee_teff_logg():
    """
    Fitting Teff: Yes
    Fitting logg: Yes
    Fitting distance: No
    Reddenning: No
    """
    ftr01 = WDfitter()
    ftr01.fit(
        atmosphere="H",
        filters=five_filters_name_list,
        mags=mags,
        mag_errors=np.ones(five_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 7.5],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        distance=10.0,
        distance_err=0.1,
        prior=prior_teff_logg,
    )
    assert np.isclose(
        ftr01.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )
    assert np.isclose(
        ftr01.best_fit_params["H"]["logg"],
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        prior=prior_teff,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        rv=RV,
        ebv=EBV,
        prior=prior_teff,
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
        atmosphere_interpolator="RBF",
        extinction_convolved=False,
        initial_guess=[13000.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        rv=RV,
        ebv=EBV,
        prior=prior_teff,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 7.5],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        distance=10.0,
        distance_err=0.1,
        rv=RV,
        ebv=EBV,
        prior=prior_teff_logg,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 7.5],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        distance=10.0,
        distance_err=0.1,
        rv=RV,
        ebv=EBV,
        prior=prior_teff_logg,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 7.5, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        prior=prior_teff_logg_distance,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        prior=prior_teff_distance,
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
        mags=np.concatenate((mags + extinction, mags_grizyJHK + extinction_grizyJHK)),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff"],
        method="emcee",
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        rv=RV,
        ebv=EBV,
        prior=prior_teff_distance,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        logg=7.5,
        rv=RV,
        ebv=EBV,
        prior=prior_teff_distance,
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
        mags=np.concatenate((mags + extinction, mags_grizyJHK + extinction_grizyJHK)),
        mag_errors=np.ones(thirteen_filters_name_list.size) * 0.02,
        independent=["Teff", "logg"],
        method="emcee",
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 7.5, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        rv=RV,
        ebv=EBV,
        prior=prior_teff_logg_distance,
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
        atmosphere_interpolator="RBF",
        initial_guess=[13000.0, 7.5, 10.0],
        refine=False,
        nwalkers=20,
        nsteps=1000,
        nburns=100,
        rv=RV,
        ebv=EBV,
        prior=prior_teff_logg_distance,
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
