from WDPhotTools.fitter import WDfitter
import numpy as np
from unittest.mock import patch
import os  # For testing file existence with assert
import time  # For testing file existence with assert

from WDPhotTools.reddening import reddening_vector_filter
from WDPhotTools.reddening import reddening_vector_interpolated

# testing with logg=7.5 and Teff=13000.
wave_GBRFN = np.array((6218.0, 5110.0, 7769.0, 1535.0, 2301.0))

rv = 3.1
ebv = 0.123

reddening = reddening_vector_interpolated(kind="cubic")
extinction_interpolated = reddening(wave_GBRFN, rv) * ebv

ftr = WDfitter()

A_G3 = reddening_vector_filter("G3")([7.5, 13000.0, rv]) * ebv
A_G3_BP = reddening_vector_filter("G3_BP")([7.5, 13000.0, rv]) * ebv
A_G3_RP = reddening_vector_filter("G3_RP")([7.5, 13000.0, rv]) * ebv
A_FUV = reddening_vector_filter("FUV")([7.5, 13000.0, rv]) * ebv
A_NUV = reddening_vector_filter("NUV")([7.5, 13000.0, rv]) * ebv

mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
extinction = np.array([A_G3, A_G3_BP, A_G3_RP, A_FUV, A_NUV]).reshape(-1)


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: No
# Reddenning: No
def test_minimize_Mbol_logg():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5],
        distance=10.0,
        distance_err=0.1,
    )
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: No
# Reddenning: No
def test_minimize_Mbol():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: No
# Reddenning: Yes
def test_minimize_Mbol_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: No
# Reddenning: Yes
def test_minimize_Mbol_logg_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5],
        distance=10.0,
        distance_err=0.1,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: Yes
# Reddenning: No
def test_minimize_Mbol_logg_distance():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5, 10.0],
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: Yes
# Reddenning: No
def test_minimize_Mbol_distance():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 10.0],
        logg=7.5,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: Yes
# Reddenning: Yes
def test_minimize_Mbol_distance_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 10.0],
        logg=7.5,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: Yes
# Reddenning: Yes
def test_minimize_Mbol_logg_distance_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="minimize",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5, 10.0],
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# least_squares

# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: No
# Reddenning: No
def test_lsq_Mbol_logg():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5],
        distance=10.0,
        distance_err=0.1,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: No
# Reddenning: No
def test_lsq_Mbol():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: No
# Reddenning: Yes
def test_lsq_Mbol_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0],
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: No
# Reddenning: Yes
def test_lsq_Mbol_logg_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5],
        distance=10.0,
        distance_err=0.1,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: Yes
# Reddenning: No
def test_lsq_Mbol_logg_distance():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5, 10.0],
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: Yes
# Reddenning: No
def test_lsq_Mbol_distance():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 10.0],
        logg=7.5,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: Yes
# Reddenning: Yes
def test_lsq_Mbol_distance_logg():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 10.0],
        logg=7.5,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: Yes
# Reddenning: Yes
def test_lsq_Mbol_logg_distance_reddening():
    ftr = WDfitter()
    ftr.fit(
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="least_squares",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5, 10.0],
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


#
# emcee
#


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: No
# Reddenning: No
def test_emcee_Mbol_logg():
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5],
        nwalkers=100,
        nsteps=2000,
        nburns=200,
        distance=10.0,
        distance_err=0.1,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: No
# Reddenning: No
def test_emcee_Mbol():
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[10.0],
        nwalkers=100,
        nsteps=2000,
        nburns=200,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: No
# Reddenning: Yes
def test_emcee_Mbol_reddening():
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[10.0],
        nwalkers=100,
        nsteps=2000,
        nburns=200,
        logg=7.5,
        distance=10.0,
        distance_err=0.1,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: Yes
# Fitting distance: No
# Reddenning: Yes
def test_emcee_Mbol_logg_reddening():
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol", "logg"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 7.5],
        nwalkers=100,
        nsteps=2000,
        nburns=200,
        distance=10.0,
        distance_err=0.1,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# """ Not working yet
# # Fitting Mbol: Yes
# # Fitting logg: Yes
# # Fitting distance: Yes
# # Reddenning: No
# def test_emcee_Mbol_logg_distance():
#     ftr = WDfitter()
#     ftr.fit(
#         atmosphere="H",
#         filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
#         mags=mags,
#         mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
#         independent=["Mbol", "logg"],
#         method="emcee",
#         atmosphere_interpolator="CT",
#         initial_guess=[10.0, 7.5, 10.0],
#         nwalkers=100,
#         nsteps=2000,
#         nburns=200,
#     )
#     ftr.best_fit_params["H"]["Mbol"]
#     ftr.best_fit_params["H"]["Teff"]
#     assert np.isclose(
#         ftr.best_fit_params["H"]["Teff"],
#         13000.0,
#         rtol=1e-01,
#         atol=1e-01,
#     )
# """

# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: Yes
# Reddenning: No
def test_emcee_Mbol_distance():
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 10.0],
        nwalkers=100,
        nsteps=2000,
        nburns=200,
        logg=7.5,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# Fitting Mbol: Yes
# Fitting logg: No
# Fitting distance: Yes
# Reddenning: Yes
def test_minimize_Mbol_distance_reddening():
    ftr = WDfitter()
    ftr.fit(
        atmosphere="H",
        filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
        mags=mags + extinction,
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=["Mbol"],
        method="emcee",
        atmosphere_interpolator="CT",
        initial_guess=[10.0, 10.0],
        nwalkers=100,
        nsteps=2000,
        nburns=200,
        logg=7.5,
        Rv=rv,
        ebv=ebv,
    )
    ftr.best_fit_params["H"]["Mbol"]
    ftr.best_fit_params["H"]["Teff"]
    assert np.isclose(
        ftr.best_fit_params["H"]["Teff"],
        13000.0,
        rtol=1e-01,
        atol=1e-01,
    )


# """ Not working yet
# # Fitting Mbol: Yes
# # Fitting logg: Yes
# # Fitting distance: Yes
# # Reddenning: Yes
# def test_minimize_Mbol_logg_distance_reddening():
#     ftr = WDfitter()
#     ftr.fit(
#         atmosphere='H',
#         filters=["G3", "G3_BP", "G3_RP", "FUV", "NUV"],
#         mags=mags + extinction,
#         mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
#         independent=["Mbol", 'logg'],
#         method="emcee",
#         atmosphere_interpolator='CT',
#         initial_guess=[10.0, 7.5, 10.0],
#         nwalkers=100,
#         nsteps=2000,
#         nburns=200,
#         Rv=rv,
#         ebv=ebv,
#     )
#     ftr.best_fit_params['H']['Mbol']
#     ftr.best_fit_params['H']['Teff']
#     assert np.isclose(ftr.best_fit_params['H']['Teff'], 13000.,
#     rtol=1e-01,
#     atol=1e-01,)
# """
