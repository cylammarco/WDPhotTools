from WDPhotTools.fitter import WDfitter
import numpy as np
from WDPhotTools.reddening import reddening_vector_filter, reddening_vector_interpolated

# testing with logg=7.5 and Teff=13000.
wave_GBRFN = np.array((6218., 5110., 7769., 1535., 2301.))

rv = 3.1
ebv = 1.23

reddening = reddening_vector_interpolated(kind='cubic')
extinction_interpolated = reddening(wave_GBRFN, rv) * ebv

ftr = WDfitter()

A_G3 = reddening_vector_filter('G3')([7.5, 13000., rv]) * ebv
A_G3_BP = reddening_vector_filter('G3_BP')([7.5, 13000., rv]) * ebv
A_G3_RP = reddening_vector_filter('G3_RP')([7.5, 13000., rv]) * ebv
A_FUV = reddening_vector_filter('FUV')([7.5, 13000., rv]) * ebv
A_NUV = reddening_vector_filter('NUV')([7.5, 13000., rv]) * ebv

extinction = np.array([A_G3, A_G3_BP, A_G3_RP, A_FUV, A_NUV]).reshape(-1)


# List all atmosphere parameters
def test_list_everything():
    ftr.list_atmosphere_parameters()


# Fitting for Mbol with 5 filters for both DA and DB
def test_fitting_Mbol():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.5,
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
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for Mbol with 5 filters for both DA and DB with alternating None
def test_fitting_Mbol_with_None():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, 10.350],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB
def test_fitting_logg_and_mbol():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.5])
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 5 filters for both DA and DB
def test_fitting_logg_Mbol_distance():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.5])
    ftr.show_best_fit(display=False)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 8 filters for both DA and DB with
# Nelder-Mead method
def test_fitting_logg_Mbol_distance_nelder_mead():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.5],
            kwargs_for_minimize={'method': 'Nelder-Mead'})
    ftr.show_best_fit(display=False)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for Mbol with 5 filters for both DA and DB with added extinction
def test_fitting_Mbol_red():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      savefig=True,
                      folder='test_output',
                      ext=['png', 'pdf'],
                      return_fig=True)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB with added
# extinction
def test_fitting_logg_and_Mbol_red():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.5],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 5 filters for both DA and DB with
# added extinction
def test_fitting_logg_Mbol_distance_red():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.5],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      title='fitted (logg, Mbol, distance) and dereddend')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


#
#
#
# Repeat all the test with optimize.least_square
#
#
#
#


# Fitting for Mbol with 5 filters for both DA and DB
def test_fitting_Mbol_lsq():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            method='least_square',
            initial_guess=[10.0])
    ftr.show_best_fit(display=False,
                      savefig=True,
                      folder='test_output',
                      ext=['png', 'pdf'],
                      return_fig=True)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for Mbol with 5 filters for both DA and DB with alternating None
def test_fitting_Mbol_with_None_lsq():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            method='least_square',
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, 10.350],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            method='least_square',
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            method='least_square',
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB
def test_fitting_logg_and_Mbol_lsq():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            distance=10.,
            distance_err=0.1,
            method='least_square',
            initial_guess=[10.0, 7.5])
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 5 filters for both DA and DB
def test_fitting_logg_Mbol_distance_lsq():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='least_square',
            initial_guess=[10.0, 7.5])
    ftr.show_best_fit(display=False)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 8 filters for both DA and DB with
# Nelder-Mead method
def test_fitting_logg_Mbol_distance_nelder_mead_lsq():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.5],
            method='least_square')
    ftr.show_best_fit(display=False)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for Mbol with 5 filters for both DA and DB with added extinction
def test_fitting_Mbol_red_lsq():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.5,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            method='least_square',
            initial_guess=[10.0],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      savefig=True,
                      folder='test_output',
                      ext=['png', 'pdf'],
                      return_fig=True)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB with added
# extinction
def test_fitting_logg_and_Mbol_red_lsq():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.5],
            method='least_square',
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 5 filters for both DA and DB with
# added extinction
def test_fitting_logg_Mbol_distance_red_lsq():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.5],
            method='least_square',
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      title='fitted (logg, Mbol, distance) and dereddend')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()


#
#
# Repeat all the test with emcee
#
#
#


# Fitting for logg and Mbol with 5 filters for both DA and DB
def test_fitting_Mbol_emcee():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.5,
            independent=['Mbol'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            refine_bounds=[0.1, 99.9],
            initial_guess=[10.0])
    ftr.show_corner_plot(display=False,
                         folder='test_output',
                         filename='test_fitting_and_mbol',
                         ext='png')
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for Mbol with 5 filters for both DA and DB with alternating None
def test_fitting_Mbol_with_None_emcee():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            refine_bounds=[0.1, 99.9],
            initial_guess=[10.0])
    ftr.show_corner_plot(display=False,
                         folder='test_output',
                         ext=['png', 'pdf'])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, 10.350],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            refine_bounds=[0.1, 99.9],
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.5,
            independent=['Mbol'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            refine_bounds=[0.1, 99.9],
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB
def test_fitting_logg_and_Mbol_emcee():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=[10.882, 10.853, 10.946, 11.301, 11.183],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            refine_bounds=[0.1, 99.9],
            initial_guess=[10.0, 7.5])
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for Mbol with 5 filters for both DA and DB with added extinction
def test_fitting_Mbol_red_emcee():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.5,
            independent=['Mbol'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0],
            refine_bounds=[0.1, 99.9],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      savefig=True,
                      folder='test_output',
                      ext=['png', 'pdf'],
                      return_fig=True)
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB with added
# extinction
def test_fitting_logg_and_Mbol_red_emcee():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='emcee',
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Testing the interp_reddening() by YKW on 12Jan2022
def test_interp_reddening():
    ftr.interp_reddening(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
                         interpolated=True)


# Testing the _chi2_minimization_red_interpolated() by YKW on 13Jan2022
def test_chi2_minimization_red_interpolated():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction_interpolated
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='least_square',
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            interpolated=True,
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_chi2_minimization_red_interpolated',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([9.962, 7.5]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Testing the _chi2_minimization_distance_red_interpolated() by YKW on 17Jan2022
def test_chi2_minimization_distance_red_interpolated():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction_interpolated
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='emcee',
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            interpolated=True,
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(
        display=False,
        folder='test_output',
        filename='test_chi2_minimization_distance_red_interpolated',
        ext='png')


# Testing the _chi2_minimization_distance_red_filter_fixed_logg() by YKW on 17Jan2022
def test_chi2_minimization_distance_red_filter_fixed_logg():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='emcee',
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(
        display=False,
        folder='test_output',
        filename='test_chi2_minimization_distance_red_filter_fixed_logg',
        ext='png')


# YKW Test 1 23Jan2022
def test_shower_corner_plot_savefig_true():
    ftr.show_corner_plot(display=False,
                         savefig=True,
                         folder=None,
                         filename=None,
                         ext='png')


# YKW Test 2 23Jan2022
# Testing the _chi2_minimization_distance_red_filter_fixed_logg()
def test_chi2_minimization_distance_red_filter_fixed_logg_emcee():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='emcee',
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            interpolated=False,
            Rv=rv,
            ebv=ebv,
            logg=None)


# YKW Test 3 23Jan2022
# Testing the _chi2_minimization_distance_red_filter_fixed_logg()
def test_chi2_minimization_distance_red_filter_fixed_logg_minimize():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='minimize',
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            interpolated=False,
            Rv=rv,
            ebv=ebv,
            logg=None)


# YKW Test 4 23Jan2022
# Testing the _chi2_minimization_distance_red_filter_fixed_logg()
def test_chi2_minimization_distance_red_filter_fixed_logg_least_square():
    mags = np.array([10.882, 10.853, 10.946, 11.301, 11.183])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            method='least_square',
            initial_guess=[10.0, 7.5],
            refine_bounds=[0.1, 99.9],
            interpolated=False,
            Rv=rv,
            ebv=ebv,
            logg=None)


# YKW Test 5 23Jan2022
def test_shower_corner_plot_savefig_true_folder_not_None():
    ftr.show_corner_plot(display=False,
                         savefig=True,
                         folder='test_output_folder',
                         filename=None,
                         ext='png')
