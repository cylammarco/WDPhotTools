import numpy as np
from WDLFBuilder import theoretical_lf

wdlf = theoretical_lf.WDLF()

wdlf.compute_cooling_age_interpolator()

Mag = np.arange(4, 20., 0.2)
age = 1E9 * np.arange(8, 15, 1)
num = np.zeros((len(age), len(Mag)))

wdlf.compute_density(Mag=Mag, T0=age)

wdlf.plot_cooling_model(display=False)
wdlf.plot_wdlf(display=False)

wdlf.set_low_mass_cooling_model('montreal_co_da_20')
wdlf.set_intermediate_mass_cooling_model('basti_co_da_10')
wdlf.set_high_mass_cooling_model('basti_co_da_10')
wdlf.compute_cooling_age_interpolator()

wdlf.compute_density(Mag=Mag, T0=age)

wdlf.plot_cooling_model(display=False)
wdlf.plot_wdlf(display=True)
