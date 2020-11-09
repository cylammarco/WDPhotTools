import numpy as np
from matplotlib import pyplot as plt

from theoretical_lf import WDLF

wdlf = WDLF()

wdlf.set_sfr_model(mode='burst')
wdlf.compute_cooling_age_interpolator()


L = 10.**np.arange(28, 34., 0.1)
age = 1E9 * np.arange(8, 15, 2)
num = np.zeros((len(age), len(L)))

wdlf.compute_density(L=L, T0=age)

wdlf.plot_cooling_model(display=False)
wdlf.plot_wdlf(display=False)


wdlf.set_sfr_model(mode='decay')
wdlf.compute_cooling_age_interpolator()


L = 10.**np.arange(28, 34., 0.1)
age = 1E9 * np.arange(8, 15, 2)
num = np.zeros((len(age), len(L)))

wdlf.compute_density(L=L, T0=age)

wdlf.plot_wdlf(display=True)

