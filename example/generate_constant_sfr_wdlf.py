import numpy as np
from matplotlib import pyplot as plt

from theoretical_lf import WDLF

wdlf = WDLF()

#wdlf.set_imf_model('K01')
wdlf.compute_cooling_age_interpolator()


L = 10.**np.arange(28, 34., 0.01)
age = 1E9 * np.arange(8, 15, 2)
num = np.zeros((len(age), len(L)))

wdlf.compute_density(L=L, T0=age)

wdlf.plot_cooling_model(display=False)
wdlf.plot_wdlf(display=False)

wdlf.set_low_mass_cooling_model('montreal_co_da_20')
wdlf.set_intermediate_mass_cooling_model('basti_co_da_10')
wdlf.set_high_mass_cooling_model('basti_co_da_10')
wdlf.compute_cooling_age_interpolator()

wdlf.compute_density(L=L, T0=age)

wdlf.plot_cooling_model(display=False)
wdlf.plot_wdlf(display=True)

