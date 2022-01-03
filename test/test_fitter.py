from WDPhotTools.fitter import WDfitter
import numpy as np
import os

ftr = WDfitter()


# List all atmosphere parameters
def test_list_everything():
    ftr.list_atmosphere_parameters()


# Fitting for Mbol with 5 filters for both DA and DB
def test_fitting_Mbol():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.0,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
    ftr.show_best_fit(display=False,
                      savefig=True,
                      folder='test_output',
                      ext=['png', 'pdf'],
                      return_fig=True)
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB
def test_fitting_logg_and_mbol():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.0])
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421, 7.0]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 5 filters for both DA and DB
def test_fitting_logg_Mbol_distance():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.0])
    ftr.show_best_fit(display=False)
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421, 7.0, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 8 filters for both DA and DB with Nelder-Mead
def test_fitting_logg_mbol_nelder_mead():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.0],
            kwargs_for_minimization={'method': 'Nelder-Mead'})
    ftr.show_best_fit(display=False)
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421, 7.0, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()
