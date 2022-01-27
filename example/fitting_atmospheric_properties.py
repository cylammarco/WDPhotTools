from WDPhotTools.fitter import WDfitter

ftr = WDfitter()

# Fitting for logg and Mbol with 5 filters for both DA and DB
ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
        mags=[10.744, 10.775, 10.681, 13.940, 11.738],
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=['Mbol', 'logg'],
        distance=10.,
        distance_err=0.1,
        initial_guess=[10.0, 7.0])
ftr.results['H']
ftr.results['He']
ftr.show_best_fit()

# Fitting for Mbol with 5 filters for both DA and DB
ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
        mags=[10.744, 10.775, 10.681, 13.940, 11.738],
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        logg=7.05,
        independent=['Mbol'],
        distance=10.,
        distance_err=0.1,
        initial_guess=[10.0])
ftr.results['H']
ftr.results['He']
ftr.show_best_fit()

# Fitting for logg, Mbol and distance with 5 filters for both DA and DB
ftr.fit(filters=['G3', 'G3_BP', 'G3_RP', 'FUV', 'NUV'],
        mags=[10.744, 10.775, 10.681, 13.940, 11.738],
        mag_errors=[0.1, 0.1, 0.1, 0.1, 0.1],
        independent=['Mbol', 'logg'],
        initial_guess=[10.0, 7.0])
ftr.results['H']
ftr.results['He']
ftr.show_best_fit()

# Fitting for logg and Mbol with 8 filters for both DA and DB
ftr.fit(filters=[
    'g_ps1', 'r_ps1', 'i_ps1', 'z_ps1', 'y_ps1', 'G3', 'G3_BP', 'G3_RP'
],
        mags=[
            21.1437, 19.9678, 19.4993, 19.2981, 19.1478, 20.0533, 20.7883,
            19.1868
        ],
        mag_errors=[
            0.0321, 0.0229, 0.0083, 0.0234, 0.0187, 0.006322, 0.118615,
            0.070880
        ],
        independent=['Mbol', 'logg'],
        initial_guess=[15.0, 7.5],
        kwargs_for_minimize={'method': 'Nelder-Mead'})
ftr.results['H']
ftr.results['He']
ftr.show_best_fit(savefig=True,
                  folder='example_output',
                  filename='PSOJ1801p6254')
