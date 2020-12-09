import numpy as np
from WDLFBuilder import theoretical_lf

wdlf = theoretical_lf.WDLF()

wdlf.set_ifmr_model('C08')

# Burst SFR
wdlf.set_sfr_model(mode='burst', duration=1e9)
wdlf.compute_cooling_age_interpolator()

Mag = np.arange(0, 20., 0.2)
age = 1E9 * np.arange(8, 15, 1)
num = np.zeros((len(age), len(Mag)))

wdlf.compute_density(Mag=Mag,
                     passband='G3',
                     T0=age,
                     mass_interval=0.1,
                     save_csv=True)

wdlf.plot_wdlf(display=False, savefig=True)

# Exponential decay SFR
wdlf.set_sfr_model(mode='decay')
wdlf.compute_cooling_age_interpolator()

num = np.zeros((len(age), len(Mag)))

wdlf.compute_density(Mag=Mag, T0=age, mass_interval=0.1, save_csv=True)

wdlf.plot_wdlf(display=True, savefig=True)
