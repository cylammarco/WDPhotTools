from WDPhotTools.fitter import WDfitter
import numpy as np
from WDPhotTools.reddening import reddening_vector

wave_GBRFN = np.array((6218., 5110., 7769., 1535., 2301.))

rv = 3.1
ebv = 1.23

reddening = reddening_vector(kind='cubic')
extinction = reddening(wave_GBRFN, rv) * ebv

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


# Fitting for Mbol with 5 filters for both DA and DB with alternating None
def test_fitting_Mbol_with_None():
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.0,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738, 10.438],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            allow_none=True,
            atmosphere='H',
            logg=7.0,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421]),
                      rtol=1e-03,
                      atol=1e-03).all()
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV', 'U'],
            mags=[10.744, 10.775, 10.681, 13.940, 11.738, None],
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1, 10.],
            allow_none=True,
            atmosphere='H',
            logg=7.0,
            independent=['Mbol'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0])
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


# Fitting for logg, Mbol and distance with 8 filters for both DA and DB with
# Nelder-Mead method
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


# Fitting for Mbol with 5 filters for both DA and DB with added extinction
def test_fitting_Mbol_red():
    mags = np.array([10.744, 10.775, 10.681, 13.940, 11.738])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            logg=7.0,
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
                      np.array([10.421]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg and Mbol with 5 filters for both DA and DB with added
# extinction
def test_fitting_logg_and_mbol_red():
    mags = np.array([10.744, 10.775, 10.681, 13.940, 11.738])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            distance=10.,
            distance_err=0.1,
            initial_guess=[10.0, 7.0],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      folder='test_output',
                      filename='test_fitting_logg_and_mbol',
                      ext='png')
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421, 7.0]),
                      rtol=1e-03,
                      atol=1e-03).all()


# Fitting for logg, Mbol and distance with 5 filters for both DA and DB with
# added extinction
def test_fitting_logg_Mbol_distance_red():
    mags = np.array([10.744, 10.775, 10.681, 13.940, 11.738])
    mags = mags + extinction
    ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
            mags=mags,
            mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
            independent=['Mbol', 'logg'],
            initial_guess=[10.0, 7.0],
            Rv=rv,
            ebv=ebv)
    ftr.show_best_fit(display=False,
                      title='fitted (logg, Mbol, distance) and dereddend')
    assert np.isclose(ftr.results['H'].x,
                      np.array([10.421, 7.0, 10.]),
                      rtol=1e-03,
                      atol=1e-03).all()
